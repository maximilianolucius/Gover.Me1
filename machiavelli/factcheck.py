"""Fact-checking engine that splits text into claims, retrieves context via RAG,
and classifies each claim using an LLM.

Outputs structured JSON with per-claim verdicts, character-level discourse
annotations, source references, and headline summaries. Can run as a CLI
or be imported for programmatic use.
"""

from __future__ import annotations

import argparse
import datetime as _dt
import json
import re
import sys
import traceback
from typing import Any, Dict, List, Tuple, Optional, Set


def _print_tb_to_stderr(prefix: str = "") -> str:
    """Prints full traceback to stderr and returns it as string"""
    tb_str = "".join(traceback.format_exception(*sys.exc_info()))
    banner = f"\n===== TRACEBACK {prefix}=====\n"
    sys.stderr.write(banner)
    traceback.print_exc()
    sys.stderr.write("===== END TRACEBACK =====\n\n")
    sys.stderr.flush()
    return tb_str


def _debug_print(message: str, debug: bool = False) -> None:
    """Print debug message to stderr if debug is enabled"""
    if debug:
        sys.stderr.write(f"[DEBUG] {message}\n")
        sys.stderr.flush()


# --------------------------------- Split utils ---------------------------------

# Fixed corrupted characters in regex pattern
_SENT_SPLIT_REGEX = re.compile(
    r"""
    (?<= [\.\!\?] )
    \s+
    (?= [A-ZÃÃ‰ÃÃ"ÃšÃ'Ãœ0-9Â¿Â¡] )
    """,
    re.VERBOSE,
)


def _norm_ws(s: str) -> str:
    """Collapse all whitespace runs into a single space and strip."""
    return re.sub(r"\s+", " ", s).strip()


def split_into_claims_with_positions(
        text: str,
        min_claim_len: int = 40,
        max_claim_len: int = 420,
) -> List[Tuple[str, int, int]]:
    """Split text into claims and return (claim_text, start_pos, end_pos) tuples"""
    raw = [_norm_ws(x) for x in _SENT_SPLIT_REGEX.split(text) if _norm_ws(x)]
    claims: List[Tuple[str, int, int]] = []

    current_pos = 0
    buf = ""
    buf_start = 0

    for s in raw:
        s_start = text.find(s, current_pos)
        if s_start == -1:
            s_start = current_pos
        s_end = s_start + len(s)
        current_pos = s_end

        if not buf:
            buf = s
            buf_start = s_start
            continue

        if len(buf) < min_claim_len:
            buf = f"{buf} {s}"
        else:
            buf_end = text.find(buf, buf_start) + len(buf)
            if buf_end == -1 + len(buf):
                buf_end = buf_start + len(buf)
            claims.append((buf, buf_start, buf_end))
            buf = s
            buf_start = s_start

    if buf:
        buf_end = text.find(buf, buf_start) + len(buf)
        if buf_end == -1 + len(buf):
            buf_end = buf_start + len(buf)
        claims.append((buf, buf_start, buf_end))

    final: List[Tuple[str, int, int]] = []
    for claim_text, start, end in claims:
        if len(claim_text) <= max_claim_len:
            final.append((claim_text, start, end))
            continue

        # Handle long claims by splitting them
        parts = re.split(r"(?<=,)\s+|\s{2,}", claim_text)
        chunk = ""
        chunk_start = start
        current_offset = 0

        for p in parts:
            cand = f"{chunk} {p}".strip()
            if len(cand) > max_claim_len and chunk:
                chunk_end = start + current_offset + len(chunk)
                final.append((chunk, chunk_start, chunk_end))
                chunk = p
                chunk_start = start + current_offset + len(chunk) + 1
            else:
                chunk = cand
            current_offset += len(p) + 1

        if chunk:
            chunk_end = start + current_offset + len(chunk)
            final.append((chunk, chunk_start, min(chunk_end, end)))

    # Filter short claims
    final = [(c, s, e) for c, s, e in final if len(c) >= max(10, min_claim_len // 2)]
    return final or [(_norm_ws(text), 0, len(text))]


def split_into_claims(
        text: str,
        min_claim_len: int = 40,
        max_claim_len: int = 420,
) -> List[str]:
    """Original function for backward compatibility"""
    return [claim for claim, _, _ in split_into_claims_with_positions(text, min_claim_len, max_claim_len)]


# ------------------------------- Milvus helpers --------------------------------

def _extract_hit_fields(hit: Any) -> Tuple[Optional[str], Optional[int], Optional[str]]:
    """Extract (document_id, chunk_index, text) from a Milvus search hit.

    Handles multiple hit formats: dict with various key conventions,
    nested entity/metadata dicts, and positional list/tuple results.
    """
    if isinstance(hit, dict):
        doc_id = hit.get("document_id") or hit.get("doc_id") or hit.get("id")
        chunk_idx = hit.get("chunk_index") or hit.get("chunk_idx") or hit.get("chunk_id")
        text = hit.get("text") or hit.get("content") or hit.get("chunk") or hit.get("page_content")
        meta = hit.get("metadata") or {}
        if doc_id is None:
            doc_id = meta.get("document_id") or meta.get("doc_id") or meta.get("id")
        if chunk_idx is None:
            chunk_idx = meta.get("chunk_index") or meta.get("chunk_idx")
        if text is None:
            text = meta.get("text") or meta.get("content") or meta.get("page_content")
        ent = hit.get("entity") or {}
        if doc_id is None and isinstance(ent, dict):
            doc_id = ent.get("document_id") or ent.get("doc_id") or ent.get("id")
        return (str(doc_id) if doc_id is not None else None,
                int(chunk_idx) if chunk_idx is not None else None,
                text)
    if isinstance(hit, (list, tuple)) and len(hit) >= 3:
        try:
            return str(hit[0]), int(hit[1]), str(hit[2])
        except Exception:
            pass
    return None, None, None


def _fmt_date(d: Any) -> str:
    """Convert a date-like value to an ISO 8601 string, or empty string if None."""
    if hasattr(d, "isoformat"):
        try:
            return d.isoformat()  # type: ignore[attr-defined]
        except Exception:
            pass
    if isinstance(d, (int, float)):
        try:
            return _dt.datetime.fromtimestamp(d).isoformat()
        except Exception:
            pass
    return str(d) if d is not None else ""


def _build_context_and_sources_for_claim(
        chatbot: Any,
        claim_text: str,
        top_k: int,
        max_context_chars: int,
        debug: bool = False,
) -> Tuple[str, List[Dict[str, Any]]]:
    """Embed a claim, search Milvus for similar chunks, and build a context string.

    Returns:
        Tuple of (concatenated context text, list of source dicts with metadata).
    """
    _debug_print(f"Building context for claim: {claim_text[:100]}...", debug)

    try:
        query_vec = chatbot.embeddings.embed_query(claim_text)
        _debug_print(
            f"Generated embedding vector of length: {len(query_vec) if hasattr(query_vec, '__len__') else 'unknown'}",
            debug)
    except Exception as e:
        _debug_print(f"Error generating embedding: {e}", debug)
        raise

    try:
        hits = chatbot.milvus_manager.search_similar(query_vec, top_k, dim=chatbot.vector_dim)
        _debug_print(f"Retrieved {len(hits)} hits from Milvus", debug)
    except Exception as e:
        _debug_print(f"Error searching Milvus: {e}", debug)
        raise

    seen_chunks: Set[Tuple[str, int]] = set()
    context_parts: List[str] = []
    doc_first_ref: Dict[str, str] = {}

    for i, h in enumerate(hits):
        did, cidx, text = _extract_hit_fields(h)
        _debug_print(f"Hit {i}: doc_id={did}, chunk_idx={cidx}, text_len={len(text) if text else 0}", debug)

        if did is None or cidx is None or not text:
            _debug_print(f"Skipping hit {i} due to missing fields", debug)
            continue
        key = (did, cidx)
        if key in seen_chunks:
            _debug_print(f"Skipping duplicate chunk {key}", debug)
            continue
        seen_chunks.add(key)

        snippet = _norm_ws(str(text))
        if not snippet:
            _debug_print(f"Skipping empty snippet for chunk {key}", debug)
            continue

        context_parts.append(f"• [{did}:{cidx}] {snippet}")
        if did not in doc_first_ref:
            doc_first_ref[did] = snippet

        if sum(len(x) + 1 for x in context_parts) >= max_context_chars:
            _debug_print(f"Reached max context chars limit", debug)
            break

    context = "\n".join(context_parts)
    _debug_print(f"Final context length: {len(context)} chars", debug)

    fuentes: List[Dict[str, Any]] = []
    for i, did in enumerate(doc_first_ref.keys(), start=1):
        try:
            meta = chatbot.db_manager.get_document_by_id(did)
            _debug_print(f"Document {did} metadata: {meta}", debug)
        except Exception as e:
            _debug_print(f"Error getting metadata for doc {did}: {e}", debug)
            meta = {}

        author = ""
        url = ""
        pub = ""
        if isinstance(meta, dict):
            author = str(meta.get("author") or meta.get("source") or "") or ""
            url = str(meta.get("url") or meta.get("link") or "") or ""
            pub = _fmt_date(meta.get("publish_date") or meta.get("date"))
        fuente_str = author.strip()
        if url:
            fuente_str = f"{fuente_str} + {url}" if fuente_str else url

        fuentes.append(
            {
                "idx": str(i),
                "document_id": str(did),
                "fuente": fuente_str or "Desconocido",
                "publish_date": pub,
                "ref": doc_first_ref.get(did, ""),
            }
        )

    return context, fuentes

def _format_date(dt_obj) -> str:
    """Formato de fecha compatible"""
    if dt_obj is None:
        return ""
    # Soporta datetime, date o strings provenientes de psycopg2
    if hasattr(dt_obj, "isoformat"):
        try:
            return dt_obj.isoformat()
        except Exception:
            pass
    return str(dt_obj)




# ------------------------------- LLM + parsing ---------------------------------

_LLM_PROMPT_TMPL = """Eres un verificador de hechos extremadamente cauteloso.
Debes clasificar la afirmación del usuario *usando EXCLUSIVAMENTE el CONTEXTO*.
Si el contexto no es suficiente o es contradictorio, responde como "insufficient".

AFIRMACION:
{claim_text}

CONTEXTO (fragmentos recuperados; pueden ser redundantes o parciales):
{context}

INSTRUCCIONES DE CLASIFICACION:
- Devuelve un JSON **estricto** en una sola línea, sin texto adicional.
- Campos obligatorios:
  - "score": número real en [-1, 1]. Mapea: fake=-1, true=+1, insufficient=0.
    Usa magnitud para confianza (p.ej., -0.9 = muy seguro que es falso; +0.6 = moderadamente verdadero).
  - "classification": "fake" | "true" | "insufficient". Debe concordar con el signo de "score".
  - "rationale": texto corto explicando por qué, citando los elementos clave del CONTEXTO.
- No inventes datos; si no hay suficiente evidencia, usa "insufficient" con score cercano a 0.
Ejemplo válido:
{{"score": -0.8, "classification": "fake", "rationale": "El contexto indica X que contradice claramente la afirmación Y."}}
"""

_INCOHERENCIA_PROMPT_TMPL = """Analiza la siguiente afirmación y el contexto para detectar incoherencias específicas.

AFIRMACION:
{claim_text}

CONTEXTO:
{context}

INSTRUCCIONES:
- Si encuentras datos contradictorios, describe la incoherencia específica
- Si la afirmación es correcta o no hay suficiente información, responde "No se detecta incoherencia"
- Sé específico con números, fechas y datos concretos
- Responde en una sola línea, máximo 200 caracteres

Ejemplo: "Se afirma que el 20.4% de la población está en riesgo de pobreza, pero los datos más recientes del INE muestran un 20.7%"
"""

_RESULTADO_PROMPT_TMPL = """Basándote en el contexto, proporciona el resultado verificado para esta afirmación.

AFIRMACION:
{claim_text}

CONTEXTO:
{context}

INSTRUCCIONES:
- Proporciona la información correcta según el contexto
- Si no hay suficiente información, responde "Información insuficiente para verificar"
- Incluye datos específicos y fuentes cuando estén disponibles
- Responde en una sola línea, máximo 200 caracteres

Ejemplo: "Según el último informe del INE de 2024, la tasa de riesgo de pobreza es del 20.7%, no del 20.4%"
"""

_RESPUESTA_PROMPT_TMPL = """Genera una respuesta sugerida para corregir esta afirmación.

AFIRMACION:
{claim_text}

CONTEXTO:
{context}

INSTRUCCIONES:
- Proporciona una corrección clara y factual
- Si la afirmación es correcta, confirma los datos
- Si no hay suficiente información, sugiere buscar más datos
- Mantén un tono profesional y constructivo
- Responde en una sola línea, máximo 200 caracteres

Ejemplo: "Los datos más actualizados del INE indican que la tasa de riesgo de pobreza es del 20.7%"
"""


def _coerce_float(x: Any, default: float = 0.0) -> float:
    """Safely convert a value to float, returning *default* on failure."""
    try:
        if isinstance(x, (int, float)):
            return float(x)
        if isinstance(x, str):
            xs = x.strip().replace(",", ".")
            return float(xs)
    except Exception:
        pass
    return default


def _normalize_keys(d: Dict[str, Any]) -> Dict[str, Any]:
    """Lowercase and strip quotes from all dictionary keys."""
    nd: Dict[str, Any] = {}
    for k, v in d.items():
        nk = str(k).strip().strip('"\'').lower()
        nd[nk] = v
    return nd


def _invoke_llm_simple(chatbot: Any, prompt: str, debug: bool = False) -> str:
    """Invoke LLM and return simple text response"""
    _debug_print("=== INVOKING LLM (SIMPLE) ===", debug)

    try:
        # Check if chatbot and llm exist
        if not hasattr(chatbot, 'llm'):
            _debug_print("ERROR: chatbot.llm not found", True)
            return ""

        raw = chatbot.llm.invoke(prompt)
        content = getattr(raw, "content", raw)
        if not isinstance(content, str):
            content = str(content)

        if debug:
            sys.stderr.write(f"\n=== LLM SIMPLE RESPONSE ===\n{content}\n=== END RESPONSE ===\n")
            sys.stderr.flush()

        return content.strip()
    except Exception as e:
        _debug_print(f"LLM simple invocation failed: {e}", True)  # Always print this error
        if debug:
            _print_tb_to_stderr("during simple LLM invocation ")
        return ""


def _generate_additional_insights(chatbot: Any, claim_text: str, context: str, debug: bool = False) -> Dict[str, str]:
    """Generate additional insights using multiple LLM calls"""

    # Generate incoherencia_detectada
    incoherencia_prompt = _INCOHERENCIA_PROMPT_TMPL.format(claim_text=claim_text, context=context)
    incoherencia = _invoke_llm_simple(chatbot, incoherencia_prompt, debug)

    # Generate resultado_verificado
    resultado_prompt = _RESULTADO_PROMPT_TMPL.format(claim_text=claim_text, context=context)
    resultado = _invoke_llm_simple(chatbot, resultado_prompt, debug)

    # Generate respuesta_sugerida
    respuesta_prompt = _RESPUESTA_PROMPT_TMPL.format(claim_text=claim_text, context=context)
    respuesta = _invoke_llm_simple(chatbot, respuesta_prompt, debug)

    # Return the insights dictionary
    return {
        "incoherencia_detectada": incoherencia or "No se detecta incoherencia",
        "resultado_verificado": resultado or "Información insuficiente para verificar",
        "respuesta_sugerida": respuesta or "Se requiere más información para evaluar esta afirmación"
    }

def _invoke_llm_and_parse_json(chatbot: Any, prompt: str, debug: bool = False) -> Dict[str, Any]:
    """Send a prompt to the LLM and parse its response as a JSON verdict.

    Extracts score, classification, and rationale from the LLM output,
    falling back to heuristic score extraction if JSON parsing fails.
    """
    _debug_print("=== INVOKING LLM ===", debug)
    _debug_print(f"Prompt length: {len(prompt)} chars", debug)
    if debug:
        sys.stderr.write("\n--- FULL PROMPT START ---\n")
        sys.stderr.write(prompt)
        sys.stderr.write("\n--- FULL PROMPT END ---\n")
        sys.stderr.flush()

    try:
        raw = chatbot.llm.invoke(prompt)
        _debug_print("LLM invocation successful", debug)
    except Exception as e:
        _debug_print(f"LLM invocation failed: {e}", debug)
        if debug:
            _print_tb_to_stderr("during LLM invocation ")
        raise

    content = getattr(raw, "content", raw)
    if not isinstance(content, str):
        content = str(content)

    # ALWAYS print raw LLM content when there's an error or debug is enabled
    if debug:
        sys.stderr.write("\n=== RAW LLM RESPONSE START ===\n")
        sys.stderr.write(content)
        sys.stderr.write("\n=== RAW LLM RESPONSE END ===\n")
        sys.stderr.flush()

    _debug_print(f"LLM response length: {len(content)} chars", debug)

    # Try to extract JSON from response
    m = re.search(r"\{.*\}", content, flags=re.DOTALL)
    candidate = None
    if m:
        candidate = m.group(0)
        _debug_print("Found JSON candidate using direct regex", debug)
    else:
        m2 = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", content, flags=re.DOTALL | re.IGNORECASE)
        if m2:
            candidate = m2.group(1)
            _debug_print("Found JSON candidate using code block regex", debug)

    if candidate:
        _debug_print(f"JSON candidate: {candidate[:200]}...", debug)
        try:
            parsed = json.loads(candidate)
            if isinstance(parsed, list) and parsed and isinstance(parsed[0], dict):
                parsed = parsed[0]
            if isinstance(parsed, dict):
                parsed = _normalize_keys(parsed)
                score = _coerce_float(parsed.get("score") or parsed.get("puntaje") or parsed.get("confidence"))
                cls = str(parsed.get("classification") or parsed.get("class") or parsed.get("label") or parsed.get(
                    "clasificacion") or "insufficient").lower()
                rat = _norm_ws(
                    str(parsed.get("rationale") or parsed.get("reason") or parsed.get("justification") or parsed.get(
                        "justificacion") or ""))

                if cls not in {"true", "fake", "insufficient"}:
                    if score > 0.2:
                        cls = "true"
                    elif score < -0.2:
                        cls = "fake"
                    else:
                        cls = "insufficient"

                _debug_print(f"Successfully parsed JSON: score={score}, class={cls}", debug)
                return {"score": max(-1.0, min(1.0, score)), "classification": cls, "rationale": rat}
        except Exception as e:
            _debug_print(f"JSON parsing failed: {e}", debug)
            if debug:
                _print_tb_to_stderr("while parsing LLM JSON ")
                # Print the problematic JSON for debugging
                sys.stderr.write(f"\n=== PROBLEMATIC JSON ===\n{candidate}\n=== END PROBLEMATIC JSON ===\n")
                sys.stderr.flush()

    # Fallback: try to extract score from anywhere in the response
    _debug_print("Falling back to score extraction", debug)
    score_match = re.search(r"(-?\d+(?:[\.,]\d+)?)", content)
    score = _coerce_float(score_match.group(1), 0.0) if score_match else 0.0
    cls = "insufficient"
    if score > 0.2:
        cls = "true"
    elif score < -0.2:
        cls = "fake"

    _debug_print(f"Fallback result: score={score}, class={cls}", debug)

    # ALWAYS print full LLM response if we had to fall back to parsing
    if not debug:  # If debug was already enabled, we already printed it above
        sys.stderr.write("\n=== LLM RESPONSE (FALLBACK PARSING) ===\n")
        sys.stderr.write(content)
        sys.stderr.write("\n=== END LLM RESPONSE ===\n")
        sys.stderr.flush()

    return {
        "score": max(-1.0, min(1.0, score)),
        "classification": cls,
        "rationale": "Formato LLM no fue JSON estricto; clasificación aproximada a partir del puntaje detectado.",
    }
    _debug_print("=== INVOKING LLM ===", debug)
    _debug_print(f"Prompt length: {len(prompt)} chars", debug)
    if debug:
        sys.stderr.write("\n--- FULL PROMPT START ---\n")
        sys.stderr.write(prompt)
        sys.stderr.write("\n--- FULL PROMPT END ---\n")
        sys.stderr.flush()

    try:
        raw = chatbot.llm.invoke(prompt)
        _debug_print("LLM invocation successful", debug)
    except Exception as e:
        _debug_print(f"LLM invocation failed: {e}", debug)
        if debug:
            _print_tb_to_stderr("during LLM invocation ")
        raise

    content = getattr(raw, "content", raw)
    if not isinstance(content, str):
        content = str(content)

    # ALWAYS print raw LLM content when there's an error or debug is enabled
    if debug:
        sys.stderr.write("\n=== RAW LLM RESPONSE START ===\n")
        sys.stderr.write(content)
        sys.stderr.write("\n=== RAW LLM RESPONSE END ===\n")
        sys.stderr.flush()

    _debug_print(f"LLM response length: {len(content)} chars", debug)

    # Try to extract JSON from response
    m = re.search(r"\{.*\}", content, flags=re.DOTALL)
    candidate = None
    if m:
        candidate = m.group(0)
        _debug_print("Found JSON candidate using direct regex", debug)
    else:
        m2 = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", content, flags=re.DOTALL | re.IGNORECASE)
        if m2:
            candidate = m2.group(1)
            _debug_print("Found JSON candidate using code block regex", debug)

    if candidate:
        _debug_print(f"JSON candidate: {candidate[:200]}...", debug)
        try:
            parsed = json.loads(candidate)
            if isinstance(parsed, list) and parsed and isinstance(parsed[0], dict):
                parsed = parsed[0]
            if isinstance(parsed, dict):
                parsed = _normalize_keys(parsed)
                score = _coerce_float(parsed.get("score") or parsed.get("puntaje") or parsed.get("confidence"))
                cls = str(parsed.get("classification") or parsed.get("class") or parsed.get("label") or parsed.get(
                    "clasificacion") or "insufficient").lower()
                rat = _norm_ws(
                    str(parsed.get("rationale") or parsed.get("reason") or parsed.get("justification") or parsed.get(
                        "justificacion") or ""))

                if cls not in {"true", "fake", "insufficient"}:
                    if score > 0.2:
                        cls = "true"
                    elif score < -0.2:
                        cls = "fake"
                    else:
                        cls = "insufficient"

                _debug_print(f"Successfully parsed JSON: score={score}, class={cls}", debug)
                return {"score": max(-1.0, min(1.0, score)), "classification": cls, "rationale": rat}
        except Exception as e:
            _debug_print(f"JSON parsing failed: {e}", debug)
            if debug:
                _print_tb_to_stderr("while parsing LLM JSON ")
                # Print the problematic JSON for debugging
                sys.stderr.write(f"\n=== PROBLEMATIC JSON ===\n{candidate}\n=== END PROBLEMATIC JSON ===\n")
                sys.stderr.flush()

    # Fallback: try to extract score from anywhere in the response
    _debug_print("Falling back to score extraction", debug)
    score_match = re.search(r"(-?\d+(?:[\.,]\d+)?)", content)
    score = _coerce_float(score_match.group(1), 0.0) if score_match else 0.0
    cls = "insufficient"
    if score > 0.2:
        cls = "true"
    elif score < -0.2:
        cls = "fake"

    _debug_print(f"Fallback result: score={score}, class={cls}", debug)

    # ALWAYS print full LLM response if we had to fall back to parsing
    if not debug:  # If debug was already enabled, we already printed it above
        sys.stderr.write("\n=== LLM RESPONSE (FALLBACK PARSING) ===\n")
        sys.stderr.write(content)
        sys.stderr.write("\n=== END LLM RESPONSE ===\n")
        sys.stderr.flush()

    return {
        "score": max(-1.0, min(1.0, score)),
        "classification": cls,
        "rationale": "Formato LLM no fue JSON estricto; clasificación aproximada a partir del puntaje detectado.",
    }


# --------------------------------- Discurso generation -------------------------

def _generate_discurso_arrays(user_text: str, claim_results: List[Dict[str, Any]],
                              claims_with_pos: List[Tuple[str, int, int]]) -> Dict[str, Any]:
    """Generate character-level color and claim_mask arrays"""
    text_len = len(user_text)
    color = [0.0] * text_len
    claim_mask = [0] * text_len

    # Map each claim result to its position
    for i, (claim_text, start_pos, end_pos) in enumerate(claims_with_pos):
        if i < len(claim_results):
            result = claim_results[i]
            score = float(result.get("score", 0.0))
            classification = result.get("classification", "insufficient")

            # Ensure positions are within bounds
            start_pos = max(0, min(start_pos, text_len))
            end_pos = max(start_pos, min(end_pos, text_len))

            # Set color intensity based on absolute score
            color_intensity = min(1.0, abs(score))

            # Set claim_mask value based on classification
            if classification == "true":
                mask_value = 1
            elif classification == "fake":
                mask_value = -1
            else:  # insufficient
                mask_value = 0

            # Apply to character positions
            for pos in range(start_pos, end_pos):
                if pos < text_len:
                    color[pos] = color_intensity
                    claim_mask[pos] = mask_value

    return {
        "text": user_text,
        "color": color,
        "claim_mask": claim_mask
    }


# --------------------------------- Orchestration --------------------------------

def classify_paragraph(
        chatbot: Any,
        user_text: str,
        top_k: int,
        max_context_chars: int,
        debug: bool = False,
) -> Dict[str, Any]:
    """Orchestrate full fact-checking of a paragraph.

    Splits text into claims, retrieves context for each, classifies via LLM,
    generates additional insights, and assembles the final structured result
    including discourse annotations, headlines, and source references.
    """
    _debug_print(f"Starting classification for text: {user_text[:100]}...", debug)

    claims_with_pos = split_into_claims_with_positions(user_text)
    claims = [claim for claim, _, _ in claims_with_pos]
    _debug_print(f"Split into {len(claims)} claims", debug)

    ideas_de_fuerza_data: List[Dict[str, Any]] = []
    all_sources_unique: Dict[str, Dict[str, Any]] = {}
    score_line: List[float] = []

    for idx, claim in enumerate(claims, start=1):
        _debug_print(f"\n--- Processing claim {idx}/{len(claims)} ---", debug)
        _debug_print(f"Claim text: {claim}", debug)

        try:
            # context, fuentes = _build_context_and_sources_for_claim(
            #     chatbot, claim, top_k=top_k, max_context_chars=max_context_chars, debug=debug
            # )

            # context, fuentes = _build_context_and_sources_enhanced(chatbot, claim, top_k=top_k, max_context_chars=max_context_chars, debug=debug)

            from machiavelli.query import run_query_raw_for_fact_check

            context, fuentes = run_query_raw_for_fact_check(claim, top_k=top_k, max_context_chars=max_context_chars)

        except Exception as e:
            tb = _print_tb_to_stderr("during retrieval for a claim ")
            _debug_print(f"Retrieval failed for claim {idx}: {e}", debug)

            # Add score for failed retrieval
            score_line.append(0.0)

            # Create referencias with empty/default data
            referencias = [{"fuente": "Error en recuperación", "id": idx, "title": "Error", "url": ""}]

            ideas_de_fuerza_data.append(
                {
                    "claim": claim,
                    "incoherencia_detectada": f"Error en recuperación: {e}",
                    "resultado_verificado": "No se pudo verificar debido a error en recuperación",
                    "respuesta_sugerida": "Se requiere revisar la conectividad y configuración del sistema",
                    "referencias": referencias,
                }
            )
            continue

        if not context.strip():
            _debug_print(f"No context found for claim {idx}", debug)

            # Add score for no context
            score_line.append(0.0)

            # Create referencias with empty/default data
            referencias = [{"fuente": "Sin contexto", "id": idx, "title": "Sin información", "url": ""}]

            ideas_de_fuerza_data.append(
                {
                    "claim": claim,
                    "incoherencia_detectada": "No se detecta incoherencia",
                    "resultado_verificado": "Información insuficiente para verificar",
                    "respuesta_sugerida": "Se requiere más información para evaluar esta afirmación",
                    "referencias": referencias,
                }
            )
            continue

        prompt = _LLM_PROMPT_TMPL.format(claim_text=claim, context=context[:max_context_chars])

        try:
            parsed = _invoke_llm_and_parse_json(chatbot, prompt, debug=debug)
        except Exception as e:
            tb = _print_tb_to_stderr("during LLM invoke/parse ")
            _debug_print(f"LLM invocation failed for claim {idx}: {e}", debug)
            parsed = {
                "score": 0.0,
                "classification": "insufficient",
                "rationale": f"Error invocando/parsing LLM: {e}",
            }

        score = float(parsed.get("score", 0.0))
        score = max(-1.0, min(1.0, score))
        classification = str(parsed.get("classification", "insufficient")).lower()
        if classification not in {"true", "fake", "insufficient"}:
            if score > 0.2:
                classification = "true"
            elif score < -0.2:
                classification = "fake"
            else:
                classification = "insufficient"

        # Add score to score_line
        score_line.append(score)

        _debug_print(f"Claim {idx} result: {classification} (score={score})", debug)

        # Generate additional insights
        additional_insights = _generate_additional_insights(chatbot, claim, context, debug=debug)

        # Ensure we have valid responses (no None checking since function guarantees dict return)
        _debug_print(f"Additional insights result: {additional_insights}", debug)

        # Process sources and add to unique collection
        for f in fuentes:
            did = f.get("document_id") or f.get("idx")
            if did and str(did) not in all_sources_unique:
                fcopy = {k: v for k, v in f.items() if k != "document_id"}
                all_sources_unique[str(did)] = fcopy

        # Convert fuentes to referencias format
        referencias = []
        for i, f in enumerate(fuentes, start=1):
            ref = {
                "fuente": f.get("fuente", "Desconocido"),
                "id": i,
                "title": f.get("ref", "")[:100] + "..." if len(f.get("ref", "")) > 100 else f.get("ref", ""),
                "url": f.get("fuente", "").split(" + ")[-1] if " + " in f.get("fuente", "") else ""
            }
            referencias.append(ref)

        ideas_de_fuerza_data.append(
            {
                "claim": claim,
                "incoherencia_detectada": additional_insights["incoherencia_detectada"],
                "resultado_verificado": additional_insights["resultado_verificado"],
                "respuesta_sugerida": additional_insights["respuesta_sugerida"],
                "referencias": referencias,
            }
        )

    # Generate discurso arrays - need to recreate claim_results for compatibility
    claim_results = []
    for i, score in enumerate(score_line):
        # Determine classification from score
        if score > 0.2:
            classification = "true"
        elif score < -0.2:
            classification = "fake"
        else:
            classification = "insufficient"

        claim_results.append({
            "score": score,
            "classification": classification
        })

    discurso = _generate_discurso_arrays(user_text, claim_results, claims_with_pos)

    # Generate Titulares section
    afirmaciones_criticas = []
    afirmaciones_ambiguas = []

    for i, idea in enumerate(ideas_de_fuerza_data):
        if i < len(score_line):
            score = score_line[i]
            claim = idea["claim"]
            respuesta_sugerida = idea["respuesta_sugerida"]

            # Generate a title (first 50 chars of claim)
            titulo = claim[:50] + "..." if len(claim) > 50 else claim

            if score < -0.2:  # Critical/fake claims
                afirmaciones_criticas.append({
                    "Titulo": titulo,
                    "respuesta_sugerida": respuesta_sugerida
                })
            elif -0.2 <= score <= 0.2:  # Ambiguous/insufficient claims
                afirmaciones_ambiguas.append({
                    "Titulo": titulo,
                    "respuesta_sugerida": respuesta_sugerida
                })

    return {
        "code": 0,
        "data": {
            "ideas_de_fuerza": {
                "data": ideas_de_fuerza_data,
                "score_line": score_line
            },
            "fuentes": list(all_sources_unique.values()),
            "discurso": discurso,
            "Titulares": {
                "Afirmaciones Criticas": afirmaciones_criticas,
                "Afirmaciones Ambiguas": afirmaciones_ambiguas
            }
        },
        "error": "",
        "is_done": True,
    }


# -------------------------------------- CLI ------------------------------------

def _parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    """Parse command-line arguments for the fact-checking CLI."""
    p = argparse.ArgumentParser(description="Fact-check CLI por claims usando RAG + LLM.")
    p.add_argument(
        "--user_text", "-t", "--user_query", "-q",
        dest="user_text",
        type=str,
        required=True,
        help="Párrafo a evaluar (texto).",
    )
    p.add_argument("--top_k", type=int, default=6, help="Pasajes similares por claim (default: 6).")
    p.add_argument("--max_context_chars", type=int, default=12000,
                   help="Límite de caracteres del contexto (default: 12000).")
    p.add_argument("--debug", action="store_true", help="Imprime tracebacks y RAW LLM outputs a STDERR.")
    return p.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> int:
    """CLI entry point: parse args, initialize RAGChatbot, run fact-check, print JSON."""
    args = _parse_args(argv)

    _debug_print("Starting fact-check CLI", args.debug)
    _debug_print(f"Arguments: {vars(args)}", args.debug)

    # Try multiple import patterns
    chatbot = None
    import importlib
    import_attempts = [
        "rag_chatbot_core",
        "machiavelli.query",
        "rag.chatbot",
        "chatbot",
    ]

    for module_name in import_attempts:
        try:
            _debug_print(f"Attempting to import RAGChatbot from {module_name}", args.debug)
            module = importlib.import_module(module_name)
            chatbot_class = getattr(module, "RAGChatbot")
            chatbot = chatbot_class()
            _debug_print(f"Successfully imported RAGChatbot from {module_name}", args.debug)
            break
        except Exception as e:
            _debug_print(f"Failed to import from {module_name}: {e}", args.debug)
            continue
    if chatbot is None:
        try:
            from machiavelli.query import RAGChatbot

            chatbot = RAGChatbot()
        except Exception as e:
            traceback.print_exc()
            pass



    if chatbot is None:
        tb = _print_tb_to_stderr("while importing RAGChatbot ")
        error_msg = f"No se pudo importar RAGChatbot from any of: {import_attempts}"
        print(
            json.dumps(
                {
                    "code": 0,
                    "data": {
                        "ideas_de_fuerza": {
                            "data": [],
                            "score_line": []
                        },
                        "fuentes": [],
                        "discurso": {"text": args.user_text, "color": [0.0] * len(args.user_text),
                                     "claim_mask": [0] * len(args.user_text)},
                        "Titulares": {
                            "Afirmaciones Criticas": [],
                            "Afirmaciones Ambiguas": []
                        }
                    },
                    "error": error_msg,
                    "error_traceback": tb if args.debug else "",
                    "is_done": True,
                },
                ensure_ascii=False,
                indent=2,
            ),
            file=sys.stdout,
        )
        return 0

    try:
        result = classify_paragraph(
            chatbot=chatbot,
            user_text=args.user_text,
            top_k=args.top_k,
            max_context_chars=args.max_context_chars,
            debug=args.debug,
        )
    except Exception as e:
        tb = _print_tb_to_stderr("during classify_paragraph ")
        _debug_print(f"Main execution failed: {e}", True)  # Always print this error
        result = {
            "code": 0,
            "data": {
                "ideas_de_fuerza": {
                    "data": [],
                    "score_line": []
                },
                "fuentes": [],
                "discurso": {"text": args.user_text, "color": [0.0] * len(args.user_text),
                             "claim_mask": [0] * len(args.user_text)},
                "Titulares": {
                    "Afirmaciones Criticas": [],
                    "Afirmaciones Ambiguas": []
                }
            },
            "error": f"Excepción en ejecución: {e}",
            "error_traceback": tb if args.debug else "",
            "is_done": True,
        }

    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())







# python machiavelli_factcheck_cli.py --user_text "El párrafo a evaluar..." --top_k 6 --max_context_chars 12000
# python machiavelli_factcheck_cli.py --user_text "La Segunda Guerra Mundial comenzó en 1939 cuando Alemania invadió Polonia. Durante este conflicto, Estados Unidos lanzó bombas nucleares sobre las ciudades japonesas de Hiroshima y Nagasaki en 1945. El presidente estadounidense durante toda la guerra fue Franklin D. Roosevelt, quien murió en 1944. La guerra terminó oficialmente el 2 de septiembre de 1945 con la rendición de Japón." --top_k 6 --max_context_chars 12000
# python machiavelli_factcheck_cli.py --user_text "El Monte Everest es la montaña más alta del mundo con 8.848 metros de altura. Se encuentra en la frontera entre Nepal y China. La temperatura en su cima puede descender hasta -80 grados Celsius en invierno. Fue escalado por primera vez en 1953 por Edmund Hillary y Tenzing Norgay. El Everest crece aproximadamente 4 milímetros cada año debido al movimiento de las placas tectónicas." --top_k 8 --max_context_chars 10000
# python machiavelli_factcheck_cli.py --user_text "ChatGPT fue lanzado por OpenAI en noviembre de 2022 y revolucionó el campo de la inteligencia artificial. El modelo utiliza la arquitectura GPT-4 que tiene más de 200 billones de parámetros. Elon Musk es actualmente el CEO de OpenAI y ha invertido más de 50 mil millones de dólares en el desarrollo de la IA. La empresa fue fundada en 2015 con el objetivo de desarrollar IA general artificial segura." --top_k 5 --max_context_chars 8000
# python machiavelli_factcheck_cli.py --user_text "Apple fue fundada en 1976 por Steve Jobs, Steve Wozniak y Ronald Wayne. La empresa tiene su sede en Cupertino, California. En 2023, Apple se convirtió en la primera empresa en alcanzar una valoración de mercado de 4 billones de dólares. El iPhone representa aproximadamente el 80% de los ingresos totales de Apple. Tim Cook ha sido CEO de la empresa desde 2011." --top_k 7 --max_context_chars 15000
# python machiavelli_factcheck_cli.py --user_text "Sevilla tiene un PIB de aproximadamente 28.000 millones de euros, representando el 15% del PIB total de Andalucía. La renta per cápita en la ciudad alcanza los 32.000 euros anuales, situándose por encima de la media española. El sector servicios representa el 78% de la economía sevillana, mientras que la industria aporta un 18% y la agricultura un 4%. La tasa de desempleo en Sevilla se mantiene en torno al 12%, inferior a la media andaluza pero superior a la nacional." --top_k 6 --max_context_chars 12000
# python machiavelli_factcheck_cli.py --user_text "Sevilla recibe anualmente más de 4 millones de turistas, generando ingresos superiores a los 2.500 millones de euros. La ciudad cuenta con más de 180 hoteles y 25.000 plazas hoteleras. El turismo representa aproximadamente el 22% del PIB local. La Catedral de Sevilla y la Giralda atraen a más de 2 millones de visitantes al año. El sector turístico emplea directamente a unas 45.000 personas en la ciudad, siendo el principal motor económico junto con los servicios administrativos." --top_k 8 --max_context_chars 10000
# python machiavelli_factcheck_cli.py --user_text "El Puerto de Sevilla es el único puerto fluvial comercial de España y maneja anualmente más de 4 millones de toneladas de mercancías. La inversión en el AVE Sevilla-Madrid generó un impacto económico de más de 800 millones de euros en la región. El Aeropuerto de Sevilla-San Pablo registra un tráfico de 6,5 millones de pasajeros anuales. La ciudad cuenta con 4 líneas de metro que transportan 18 millones de usuarios al año. El sector logístico emplea a más de 15.000 personas en el área metropolitana." --top_k 7 --max_context_chars 11000
# python machiavelli_factcheck_cli.py --user_text "Sevilla alberga la sede de importantes empresas como Abengoa, que factura más de 1.500 millones de euros anuales. La factoría de Airbus en Sevilla produce el 60% de los componentes del A400M y emplea a más de 8.000 trabajadores directos. El Parque Tecnológico Cartuja concentra más de 350 empresas que generan 12.000 empleos. La industria aeroespacial aporta el 8% del PIB provincial. CajaSol, con sede en Sevilla, gestiona activos por valor de 45.000 millones de euros." --top_k 6 --max_context_chars 13000
# python machiavelli_factcheck_cli.py --user_text "Sevilla cuenta con más de 250 startups tecnológicas que han captado inversiones por valor de 180 millones de euros en los últimos cinco años. La Seville Tech City alberga a 80 empresas de base tecnológica. El sector de las TIC representa el 4,5% del PIB sevillano y emplea a más de 18.000 profesionales. La ciudad tiene 15 espacios de coworking y 8 aceleradoras de empresas. El hub de innovación Cámara de Comercio ha incubado más de 120 proyectos empresariales desde 2020." --top_k 6 --max_context_chars 10000

