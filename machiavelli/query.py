# -*- coding: utf-8 -*-
"""Enhanced RAG query system combining Milvus document retrieval with tourism data.

Provides a unified interface for querying both traditional document sources
(via Milvus vector search) and structured tourism statistics from Nexus/Excel
data, with LLM-powered answer generation and source relevance filtering.
"""

import os
import sys
import json
import argparse
from typing import List, Dict, Any, Tuple, Optional
from pathlib import Path
from tourism.extractor import UltimateTourismExtractor

# Reutilizamos todo tu stack existente
from rag.chatbot import RAGChatbot  # noqa: F401

# Importamos el sistema de turismo
try:
    from tourism.qa import TourismQA, TourismDataProcessor, get_excel_files_from_dir

    TOURISM_AVAILABLE = True
except ImportError as e:
    print(f"⚠️  Sistema de turismo no disponible: {e}")
    TOURISM_AVAILABLE = False


TOURISM_DATA_FILE="./data/tourism_data_ultimate.json"
TOURISM_EXCEL_DIR="./nexus/Excels_Limpios/"
TOURISM_CACHE_DIR="./cache"

class EnhancedRAGSystem:
    """Sistema RAG combinado que integra fuentes tradicionales + datos turísticos"""

    def __init__(self):
        """Initialize with a RAGChatbot instance; tourism system is lazy-loaded."""
        # Sistema RAG tradicional
        self.chatbot = RAGChatbot()

        # Sistema turístico (lazy initialization)
        self.tourism_qa = None
        self._tourism_initialized = False

    def _init_tourism_system(self):
        """Inicialización lazy del sistema turístico"""
        if self._tourism_initialized:
            return

        self._tourism_initialized = True

        if not TOURISM_AVAILABLE:
            print("⚠️  Sistema de turismo no disponible")
            return

        try:
            # Configuración desde variables de entorno
            tourism_data_file = os.getenv("TOURISM_DATA_FILE", TOURISM_DATA_FILE)
            tourism_excel_dir = os.getenv("TOURISM_EXCEL_DIR", "./nexus/Excels_Limpios/")
            tourism_cache_dir = os.getenv("TOURISM_CACHE_DIR", "./cache")

            # Cargar o procesar datos turísticos
            if tourism_data_file and Path(tourism_data_file).exists():
                print(f"📂 Cargando datos turísticos desde: {tourism_data_file}")
                with open(tourism_data_file, 'r', encoding='utf-8') as f:
                    tourism_data = json.load(f)
            else:
                # Procesar archivos Excel
                excel_files = get_excel_files_from_dir(tourism_excel_dir)
                if not excel_files:
                    print(f"⚠️  No se encontraron archivos Excel en {tourism_excel_dir}")
                    return

                print(f"🏖️ Procesando {len(excel_files)} archivos turísticos...")
                processor = TourismDataProcessor()
                tourism_data = processor.extract_all_excel_data(excel_files)

            # Inicializar sistema Q&A turístico
            self.tourism_qa = TourismQA(tourism_data, cache_dir=tourism_cache_dir)
            print("✅ Sistema turístico inicializado")

        except Exception as e:
            print(f"❌ Error inicializando sistema turístico: {e}")
            self.tourism_qa = None


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


def _preprocess_query_for_search(enhanced_system: EnhancedRAGSystem, original_question: str) -> str:
    """
    Preprocesa la pregunta del usuario para optimizar la búsqueda vectorial.
    Elimina referencias temporales y texto obvio para mejorar la recuperación.
    """
    prompt = (
        "Eres un asistente que optimiza consultas para búsqueda vectorial. "
        "Tu tarea es convertir preguntas largas en consultas de búsqueda concisas y efectivas.\n\n"
        "INSTRUCCIONES:\n"
        "- Elimina frases obvias como 'Dime las noticias más relevantes', 'Cuéntame sobre', etc.\n"
        "- Elimina referencias temporales específicas (fechas, meses, años)\n"
        "- Mantén solo los conceptos clave y temas principales\n"
        "- Responde ÚNICAMENTE con la consulta optimizada, sin explicaciones\n\n"
        "EJEMPLOS:\n"
        "Entrada: 'Dime las noticias más relevantes de Andalucía sobre incendios en Agosto 2025'\n"
        "Salida: 'Andalucía incendios'\n\n"
        "Entrada: '¿Cuáles fueron los principales eventos económicos de Sevilla el año pasado?'\n"
        "Salida: 'Sevilla eventos económicos'\n\n"
        f"Entrada: '{original_question}'\n"
        "Salida:"
    )

    try:
        response = enhanced_system.chatbot.llm.invoke(prompt)
        processed_query = getattr(response, "content", str(response)).strip()

        # Validación básica - si el resultado es muy corto o vacío, usar la consulta original
        if len(processed_query) < 5 or not processed_query:
            print(f"⚠️ Preprocesamiento LLM falló, usando consulta original")
            return original_question

        print(f"🔄 Query optimizada: '{original_question}' -> '{processed_query}'")
        return processed_query

    except Exception as e:
        print(f"⚠️ Error en preprocesamiento LLM: {e}, usando consulta original")
        return original_question


def _build_context_and_sources_enhanced(enhanced_system: EnhancedRAGSystem, question: str,
                                        top_k: int = 6, max_context_chars: int = 20000, mode='datos') -> Tuple[
    str, List[Dict[str, Any]], str]:
    """
    Versión mejorada que combina:
    1) RAG tradicional (Milvus + embeddings)
    2) Datos turísticos (tourism_qa)

    Returns:
        Tuple[str, List[Dict]]: (contexto_combinado, fuentes_estructuradas)
    """

    # ==========================================
    # 1) RAG TRADICIONAL (sistema existente)
    # ==========================================

    traditional_fuentes: List[Dict[str, Any]] = []
    # Construcción de contexto tradicional
    seen_chunks = set()
    traditional_ctx_parts: List[str] = []
    current_len = 0
    processed_question = ""


    # Fuentes tradicionales (evitar duplicados por documento)
    seen_docs = set()

    if mode == 'medios':
        # PREPROCESAMIENTO LLM DE LA CONSULTA
        processed_question = _preprocess_query_for_search(enhanced_system, question)

        # Embedding de la consulta procesada
        query_vector = enhanced_system.chatbot.embeddings.embed_query(processed_question)

        # Recuperación en Milvus
        similar = enhanced_system.chatbot.milvus_manager.search_similar(
            query_vector, top_k=top_k, dim=enhanced_system.chatbot.vector_dim
        )

        # Reservar espacio para datos turísticos (30% del límite)
        traditional_limit = int(max_context_chars * 0.7)

        for hit in similar:
            did = hit.get("document_id")
            cidx = hit.get("chunk_index")
            txt = (hit.get("text") or "").strip()
            if not txt:
                continue

            # Contexto (por chunk)
            key_chunk = (did, cidx)
            if key_chunk not in seen_chunks and current_len + len(txt) + 2 <= traditional_limit:
                seen_chunks.add(key_chunk)
                traditional_ctx_parts.append(f"[FUENTE DOCUMENTAL] {txt}")
                current_len += len(txt) + 2

            # Fuentes (una por documento)
            if did is not None and did not in seen_docs:
                meta = enhanced_system.chatbot.db_manager.get_document_by_id(int(did)) or {}
                autor = meta.get("autor") or "Autor desconocido"
                link = meta.get("url_original") or ""
                fecha = _format_date(meta.get("fecha"))
                seen_docs.add(did)

                traditional_fuentes.append({
                    "idx": str(len(traditional_fuentes) + 1),
                    "fuente": f"{autor} + {link}" if link else f"{autor}",
                    "publish_date": fecha,
                    "ref": txt[:200] + "..." if len(txt) > 200 else txt,
                    "tipo": "documental"
                })

    # ==========================================
    # 2) DATOS TURÍSTICOS
    # ==========================================

    tourism_ctx_parts: List[str] = []
    tourism_fuentes: List[Dict[str, Any]] = []

    # Inicializar sistema turístico si es necesario
    enhanced_system._init_tourism_system()

    if mode == 'datos':

        if enhanced_system.tourism_qa is not None:
            try:
                # Detectar si la pregunta puede tener componente turístico
                tourism_keywords = [
                    'turismo', 'turista', 'turístico', 'visitante', 'hotel', 'alojamiento',
                    'pernoctación', 'gasto', 'aeropuerto', 'crucero', 'andalucía', 'andaluces',
                    'sevilla', 'málaga', 'granada', 'córdoba', 'cádiz', 'almería', 'huelva', 'jaén',
                    'litoral', 'interior', 'cultural', 'extranjero', 'británico', 'alemán',
                    'ocupación', 'llegadas', 'estancia', 'ingresos'
                ]

                question_lower = question.lower()
                has_tourism_context = any(keyword in question_lower for keyword in tourism_keywords)

                if has_tourism_context or True:
                    print("🖖 Consultando datos turísticos...")

                    # Obtener contexto y fuentes del sistema turístico
                    tourism_context, tourism_sources = enhanced_system.tourism_qa._build_context_and_sources(question,
                                                                                                             top_k=16)

                    if tourism_context:
                        # Adaptar formato y agregar prefijo identificativo
                        tourism_lines = tourism_context.split('\n\n')
                        remaining_space = max_context_chars - current_len

                        for line in tourism_lines:
                            if line.strip() and current_len + len(line) + 30 <= max_context_chars:
                                formatted_line = f"[DATOS TURÍSTICOS NEXUS] {line.strip()}"
                                tourism_ctx_parts.append(formatted_line)
                                current_len += len(formatted_line) + 2

                    # Adaptar fuentes turísticas al formato estándar
                    for idx, source in enumerate(tourism_sources):
                        tourism_fuentes.append({
                            "idx": str(len(traditional_fuentes) + len(tourism_fuentes) + 1),
                            "fuente": source.get("fuente", "Nexus Andalucía"),
                            "publish_date": source.get("publish_date", "01/01/2025"),
                            "ref": source.get("ref", ""),
                            "tipo": "turístico",
                            "archivo_origen": source.get("archivo_origen", "")
                        })

                    print(f"✅ Agregados {len(tourism_ctx_parts)} fragmentos turísticos")

            except Exception as e:
                print(f"⚠️ Error consultando datos turísticos: {e}")

    # ==========================================
    # 3) COMBINACIÓN FINAL
    # ==========================================

    # Combinar contextos con separadores claros
    all_context_parts = []
    all_fuentes = []
    if traditional_ctx_parts and mode == 'medios':
        all_context_parts.extend(traditional_ctx_parts)
        all_fuentes += traditional_fuentes

    if tourism_ctx_parts and mode == 'datos':
        all_context_parts.extend(tourism_ctx_parts)
        all_fuentes += tourism_fuentes

    # Reindexar fuentes
    for idx, fuente in enumerate(all_fuentes):
        fuente["idx"] = str(idx + 1)

    combined_context = "\n\n".join(all_context_parts)

    print(
        f"📊 Contexto final: {len(combined_context)} chars, {len(all_fuentes)} fuentes ({len(traditional_fuentes)} doc + {len(tourism_fuentes)} turismo)")

    return combined_context, all_fuentes, processed_question


def _ask_llm_enhanced(chatbot: RAGChatbot, question: str, context: str, processed_question: str) -> str:
    """
    LLM mejorado que entiende múltiples tipos de fuentes
    """
    if not context.strip():
        return ""
    unified_question = processed_question if len(processed_question) else question
    prompt = (
        "Eres un asistente especializado que responde en español de forma clara y precisa. "
        "Tienes acceso a DOS tipos de fuentes de información:\n"
        "- [FUENTE DOCUMENTAL]: Documentos y artículos generales\n"
        "- [DATOS TURÍSTICOS NEXUS]: Estadísticas oficiales de turismo de Andalucía\n\n"
        "INSTRUCCIONES:\n"
        "- Usa EXCLUSIVAMENTE la información del contexto provisto\n"
        "- Si tienes datos turísticos específicos, priorízalos para preguntas sobre turismo\n"
        "- Combina fuentes cuando sea relevante\n"
        "- Si no hay información suficiente, dilo claramente\n"
        "- Sé específico con números, fechas y porcentajes\n\n"
        f"CONTEXTO DISPONIBLE:\n{context}\n\n"
        f"PREGUNTA: {unified_question}\n\n"
        "RESPUESTA:"
    )

    out = chatbot.llm.invoke(prompt)
    return getattr(out, "content", str(out)).strip()


def run_query_raw_for_fact_check(user_query: str, top_k: int = 6, max_context_chars: Optional[int] = None) -> Dict[str, Any]:
    """Retrieve raw context and sources for a fact-check query without LLM answer generation.

    Returns:
        Tuple of (context string, list of source dicts).
    """
    if max_context_chars is None:
        max_context_chars = int(os.getenv("RAG_MAX_CONTEXT_CHARS", "12000"))
    context, fuentes = "", []
    try:
        # Inicializar sistema combinado
        enhanced_system = EnhancedRAGSystem()

        # Recuperación combinada + armado de fuentes
        context, fuentes, processed_question = _build_context_and_sources_enhanced(
            enhanced_system, user_query, top_k=top_k, max_context_chars=max_context_chars
        )
    except Exception as e:
        pass

    return context, fuentes


def _filter_fuentes_by_relevance(chatbot: RAGChatbot, final_answer: str, fuentes: List[Dict[str, Any]]) -> List[
    Dict[str, Any]]:
    """
    Filtra las fuentes basándose en la relevancia de su contenido (ref) con la respuesta final.
    Usa LLM para determinar relevancia.

    Args:
        chatbot: Instancia del RAGChatbot con acceso al LLM
        final_answer: Respuesta final generada
        fuentes: Lista de fuentes con estructura {"idx", "fuente", "ref", ...}

    Returns:
        Lista filtrada de fuentes relevantes, o fuentes[:2] si el filtro resulta vacío
    """
    if not fuentes or not final_answer.strip():
        return fuentes[:2] if fuentes else []

    try:
        filtered_fuentes = []

        # Procesar cada fuente individualmente
        for fuente in fuentes:
            ref_content = fuente.get("ref", "").strip()
            if not ref_content:
                continue

            # Prompt para evaluar relevancia
            relevance_prompt = (
                "Eres un evaluador de relevancia. Tu tarea es determinar si un fragmento de texto "
                "está relacionado con una respuesta dada.\n\n"
                "INSTRUCCIONES:\n"
                "- Responde ÚNICAMENTE 'SÍ' si el fragmento contiene información que apoya, "
                "complementa o está directamente relacionada con la respuesta\n"
                "- Responde 'NO' si el fragmento no tiene relación clara con la respuesta\n"
                "- No des explicaciones, solo 'SÍ' o 'NO'\n\n"
                f"RESPUESTA FINAL:\n{final_answer}\n\n"
                f"FRAGMENTO A EVALUAR:\n{ref_content}\n\n"
                "¿ESTÁ RELACIONADO? (SÍ/NO):"
            )

            # Consultar LLM
            response = chatbot.llm.invoke(relevance_prompt)
            llm_decision = getattr(response, "content", str(response)).strip().upper()

            # Agregar si es relevante
            if "SÍ" in llm_decision or "SI" in llm_decision or "YES" in llm_decision:
                filtered_fuentes.append(fuente)

        # Fallback si no hay fuentes relevantes
        if not filtered_fuentes:
            print("⚠️ No se encontraron fuentes relevantes, usando fallback (primeras 2)")
            return fuentes[:2]

        print(f"✅ Filtradas {len(filtered_fuentes)}/{len(fuentes)} fuentes por relevancia")
        return filtered_fuentes

    except Exception as e:
        print(f"⚠️ Error filtrando fuentes: {e}, usando fallback")
        return fuentes[:2]

def run_query_enhanced(user_query: str, top_k: int = 6, max_context_chars: Optional[int] = None, mode='datos') -> Dict[str, Any]:
    """
    Orquestación completa del sistema combinado
    """
    # Permitir override desde ENV
    if max_context_chars is None:
        max_context_chars = int(os.getenv("RAG_MAX_CONTEXT_CHARS", "12000"))
    try:
        # Inicializar sistema combinado
        enhanced_system = EnhancedRAGSystem()

        # Recuperación combinada + armado de fuentes
        context, fuentes, processed_question = _build_context_and_sources_enhanced(
            enhanced_system, user_query, top_k=top_k, max_context_chars=max_context_chars, mode=mode
        )

        # Si no hay contexto
        if not context:
            return {
                "code": 0,
                "data": {
                    "interim_answer": "",
                    "final_answer": "",
                    "fuentes": []
                },
                "error": "No se encontró información suficiente para responder con las fuentes disponibles.",
                "is_done": True
            }

        # Pregunta al LLM con contexto combinado
        final_answer = _ask_llm_enhanced(enhanced_system.chatbot, user_query, context, processed_question)

        # NUEVO: Filtrar fuentes por relevancia usando LLM
        filtered_fuentes = _filter_fuentes_by_relevance(enhanced_system.chatbot, final_answer, fuentes)

        return {
            "code": 0,
            "data": {
                "interim_answer": "",
                "final_answer": final_answer,
                "fuentes": filtered_fuentes  # Usar fuentes filtradas
            },
            "error": "",
            "is_done": True
        }

    except Exception as e:
        return {
            "code": 2,
            "data": {
                "interim_answer": "",
                "final_answer": "",
                "fuentes": []
            },
            "error": f"{type(e).__name__}: {str(e)}",
            "is_done": True
        }

def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for the enhanced RAG query CLI."""
    parser = argparse.ArgumentParser(description="RAG combinado: fuentes tradicionales + datos turísticos")
    parser.add_argument("--user_query", "-q", required=True, help="Pregunta del usuario en español.")
    parser.add_argument("--mode", "-m", required=False, default='medios', help="['medios', 'datos']")
    parser.add_argument("--top_k", type=int, default=6, help="Cantidad de resultados a recuperar.")
    parser.add_argument("--max_context_chars", type=int, default=None, help="Límite de caracteres para el contexto.")
    return parser.parse_args()


def main():
    """CLI entry point: run an enhanced RAG query and print the JSON result."""
    args = parse_args()
    result = run_query_enhanced(args.user_query, top_k=args.top_k, max_context_chars=args.max_context_chars, mode=args.mode)

    # Salida JSON en stdout
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

# VARIABLES DE ENTORNO PARA CONFIGURAR EL SISTEMA TURÍSTICO:
# export TOURISM_DATA_FILE="./data/tourism_data.json"      # Archivo JSON con datos procesados
# export TOURISM_EXCEL_DIR="./nexus/Excels_Limpios/"     # Directorio con archivos Excel
# export TOURISM_CACHE_DIR="./cache"                      # Cache de embeddings turísticos

# EJEMPLOS DE USO:
# python machiavelli_query_enhanced.py --user_query "¿Cuántos turistas hubo en Sevilla en junio 2025?"
# python machiavelli_query_enhanced.py --user_query "¿Qué dice el informe sobre la economía andaluza?"
# python machiavelli_query_enhanced.py --user_query "Compara el turismo cultural con los datos económicos de la región"