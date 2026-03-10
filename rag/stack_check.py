#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Health-check utility for the RAG infrastructure stack.

Validates connectivity and status of vLLM, PostgreSQL, Milvus, and GPU/CUDA,
then prints a colour-coded summary and a full JSON report.
"""
# RAG / vLLM / DB / Milvus / GPU quick health check
# Usage: python rag_stack_check.py
# Optional env var: RAG_COLLECTION (default: document_vectors)

import os
import json
import time
import subprocess
from typing import Any, Dict, Tuple

def status(ok: bool) -> str:
    """Return a coloured OK/FAIL indicator string."""
    return "✅ OK" if ok else "❌ FAIL"

def safe_import(modname: str) -> Tuple[bool, Any, str]:
    """Attempt to import a module by name, returning (success, module, error_msg)."""
    try:
        m = __import__(modname)
        return True, m, ""
    except Exception as e:
        return False, None, f"{type(e).__name__}: {e}"

def print_kv(k: str, v: Any) -> None:
    """Print a left-aligned key-value pair for the health report."""
    print(f"{k:<28} {v}")

def _env(name: str, default: str = "") -> str:
    """Shorthand for os.getenv with an empty-string default."""
    return os.getenv(name, default)

def check_env() -> Dict[str, Any]:
    """Read and display relevant environment variables, masking secrets."""
    keys = [
        "VLLM_BASE_URL", "VLLM_MODEL", "VLLM_API_KEY", "OPENAI_API_KEY",
        "POSTGRES_HOST", "POSTGRES_PORT", "POSTGRES_DB", "POSTGRES_USER", "POSTGRES_PASSWORD",
        "MILVUS_HOST", "MILVUS_PORT",
        "RAG_COLLECTION"
    ]
    vals = {k: _env(k, "") for k in keys}
    if not vals.get("RAG_COLLECTION"):
        vals["RAG_COLLECTION"] = "document_vectors"
    disp = {k: ("***" if "KEY" in k or "PASSWORD" in k else v) for k, v in vals.items()}
    print("== ENVIRONMENT ==")
    for k, v in disp.items():
        print_kv(k, v or "(empty)")
    print()
    return vals

def check_gpu() -> Dict[str, Any]:
    """Check GPU availability via nvidia-smi and PyTorch CUDA."""
    print("== GPU ==")
    res = {"nvidia_smi": {}, "torch": {}}
    try:
        out = subprocess.check_output(["nvidia-smi", "--query-gpu=name,memory.total,driver_version", "--format=csv,noheader"], timeout=5)
        txt = out.decode("utf-8", "ignore").strip()
        lines = [x.strip() for x in txt.splitlines() if x.strip()]
        gpus = []
        for ln in lines:
            parts = [p.strip() for p in ln.split(",")]
            if len(parts) >= 3:
                gpus.append({"name": parts[0], "memory_total": parts[1], "driver": parts[2]})
        ok = len(gpus) > 0
        print_kv("nvidia-smi", status(ok))
        res["nvidia_smi"] = {"ok": ok, "gpus": gpus}
    except Exception as e:
        print_kv("nvidia-smi", f"❌ {type(e).__name__}: {e}")
        res["nvidia_smi"] = {"ok": False, "error": f"{type(e).__name__}: {e}"}

    ok_imp, torch, err = safe_import("torch")
    if not ok_imp:
        print_kv("torch import", f"❌ {err}")
        res["torch"] = {"ok": False, "error": err}
    else:
        try:
            cuda_avail = bool(torch.cuda.is_available())
            num = torch.cuda.device_count() if cuda_avail else 0
            name0 = torch.cuda.get_device_name(0) if cuda_avail and num > 0 else ""
            print_kv("torch.cuda", status(cuda_avail) + (f" ({num} x {name0})" if cuda_avail else ""))
            res["torch"] = {"ok": cuda_avail, "device_count": num, "name0": name0}
        except Exception as e:
            print_kv("torch.cuda", f"❌ {type(e).__name__}: {e}")
            res["torch"] = {"ok": False, "error": f"{type(e).__name__}: {e}"}
    print()
    return res

def _http_get(url: str, headers: Dict[str,str] = None, timeout: float = 5.0) -> Tuple[int, str]:
    """Perform a simple HTTP GET and return (status_code, body)."""
    try:
        import urllib.request
        req = urllib.request.Request(url, headers=headers or {})
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            code = resp.getcode()
            data = resp.read().decode("utf-8", "ignore")
            return code, data
    except Exception as e:
        return 0, f"{type(e).__name__}: {e}"

def _http_post_json(url: str, payload: Dict[str, Any], headers: Dict[str,str] = None, timeout: float = 8.0) -> Tuple[int, str]:
    """POST a JSON payload and return (status_code, body)."""
    try:
        import urllib.request, json as _json
        data = _json.dumps(payload).encode("utf-8")
        hdrs = {"Content-Type": "application/json"}
        if headers:
            hdrs.update(headers)
        req = urllib.request.Request(url, data=data, headers=hdrs, method="POST")
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            code = resp.getcode()
            txt = resp.read().decode("utf-8", "ignore")
            return code, txt
    except Exception as e:
        return 0, f"{type(e).__name__}: {e}"

def check_vllm(env: Dict[str, Any]) -> Dict[str, Any]:
    """Verify that the vLLM server is reachable and serves the expected model."""
    print("== vLLM ==")
    base = env.get("VLLM_BASE_URL") or ""
    model = env.get("VLLM_MODEL") or ""
    api_key = env.get("VLLM_API_KEY") or env.get("OPENAI_API_KEY") or ""
    res = {"base_url": base, "model": model, "models_ok": False, "chat_ok": False, "details": {}}

    if not base:
        print_kv("base url", "❌ (missing VLLM_BASE_URL)")
        return res

    auth = {"Authorization": f"Bearer {api_key}"} if api_key else {}

    # /models
    code, txt = _http_get(base.rstrip("/") + "/models", headers=auth, timeout=5.0)
    if code == 200:
        try:
            data = json.loads(txt)
            ids = [d.get("id") for d in data.get("data", []) if isinstance(d, dict)]
            ok = (model in ids) if model else bool(ids)
            print_kv("GET /models", status(ok) + (f" (found {model})" if ok and model else ""))
            res["models_ok"] = ok
            res["details"]["models"] = ids
        except Exception as e:
            print_kv("GET /models", f"❌ ParseError: {e}")
            res["details"]["models_error"] = f"ParseError: {e}"
    else:
        print_kv("GET /models", f"❌ {code} {txt[:80]}")
        res["details"]["models_error"] = f"{code} {txt}"

    # /chat/completions (fallback to first listed model if env VLLM_MODEL is empty)
    models_list = res["details"].get("models") or []
    chat_model = model if model else (models_list[0] if models_list else "")
    payload = {
        "model": chat_model,
        "messages": [{"role": "user", "content": "ping"}],
        "max_tokens": 4,
        "temperature": 0.0,
    }
    code2, txt2 = _http_post_json(base.rstrip("/") + "/chat/completions", payload, headers=auth, timeout=8.0)
    if code2 == 200 and '"choices"' in txt2:
        print_kv("POST /chat/completions", "✅ OK")
        res["chat_ok"] = True
    else:
        print_kv("POST /chat/completions", f"❌ {code2} {txt2[:120]}")
        res["details"]["chat_error"] = f"{code2} {txt2}"
    print()
    return res

def check_postgres(env: Dict[str, Any]) -> Dict[str, Any]:
    """Test PostgreSQL connectivity and check for expected tables."""
    print("== PostgreSQL ==")
    ok_imp, _, err = safe_import("psycopg2")
    if not ok_imp:
        print_kv("import psycopg2", f"❌ {err}")
        return {"ok": False, "error": err}

    import psycopg2
    from psycopg2.extras import RealDictCursor
    host = env.get("POSTGRES_HOST", "localhost")
    port = int(env.get("POSTGRES_PORT", "5432") or 5432)
    db = env.get("POSTGRES_DB", "rag_chatbot")
    user = env.get("POSTGRES_USER", "postgres")
    pwd = env.get("POSTGRES_PASSWORD", "")

    info = {"ok": False}
    try:
        conn = psycopg2.connect(host=host, port=port, dbname=db, user=user, password=pwd, connect_timeout=5)
        cur = conn.cursor(cursor_factory=RealDictCursor)
        cur.execute("SELECT version() AS v")
        ver = cur.fetchone()["v"]
        print_kv("connect", "✅ OK")
        print_kv("version", ver.split(" on ")[0])
        cur.execute("SELECT to_regclass('public.documents') AS t1, to_regclass('public.text_chunks') AS t2")
        row = cur.fetchone()
        t1 = bool(row["t1"])
        t2 = bool(row["t2"])
        print_kv("documents table", status(t1))
        print_kv("text_chunks table", status(t2))
        counts = {}
        if t1:
            cur.execute("SELECT COUNT(*) AS c FROM documents")
            counts["documents"] = cur.fetchone()["c"]
        if t2:
            cur.execute("SELECT COUNT(*) AS c FROM text_chunks")
            counts["text_chunks"] = cur.fetchone()["c"]
        if counts:
            print_kv("row counts", counts)
        cur.close()
        conn.close()
        info.update({"ok": True, "version": ver, "tables": {"documents": t1, "text_chunks": t2}, "counts": counts})
    except Exception as e:
        print_kv("connect", f"❌ {type(e).__name__}: {e}")
        info.update({"ok": False, "error": f"{type(e).__name__}: {e}"})
    print()
    return info

def check_milvus(env: Dict[str, Any]) -> Dict[str, Any]:
    """Test Milvus connectivity and inspect the target collection."""
    print("== Milvus ==")
    ok_imp, _, err = safe_import("pymilvus")
    if not ok_imp:
        print_kv("import pymilvus", f"❌ {err}")
        return {"ok": False, "error": err}

    from pymilvus import connections, utility, Collection
    host = env.get("MILVUS_HOST", "localhost")
    port = env.get("MILVUS_PORT", "19530")
    coll_name = env.get("RAG_COLLECTION") or "document_vectors"
    info = {"ok": False, "collection": coll_name}

    try:
        connections.connect(host=host, port=port, timeout=5)
        ver = utility.get_server_version()
        print_kv("connect", "✅ OK")
        print_kv("server version", ver)
        has = utility.has_collection(coll_name)
        print_kv(f"collection '{coll_name}'", status(has))
        details = {}
        if has:
            c = Collection(coll_name)
            c.load()
            n = c.num_entities
            vec_field = next((f for f in c.schema.fields if f.name == "vector"), None)
            dim = None
            if vec_field is not None:
                dim = vec_field.params.get("dim") or getattr(vec_field, "dim", None)
            details = {"num_entities": n, "dim": int(dim) if dim else None}
            print_kv("entities", n)
            if dim:
                print_kv("vector dim", dim)
        info.update({"ok": True, "server_version": ver, "has_collection": has, "details": details})
    except Exception as e:
        print_kv("connect", f"❌ {type(e).__name__}: {e}")
        info.update({"ok": False, "error": f"{type(e).__name__}: {e}"})
    print()
    return info

def main() -> int:
    """Run all health checks and print a JSON summary."""
    started = time.time()
    env = check_env()
    gpu = check_gpu()
    vllm = check_vllm(env)
    pg = check_postgres(env)
    mv = check_milvus(env)

    overall = all([vllm.get("chat_ok"), pg.get("ok"), mv.get("ok")])
    print("== SUMMARY ==")
    print_kv("vLLM", status(vllm.get("chat_ok", False)))
    print_kv("PostgreSQL", status(pg.get("ok", False)))
    print_kv("Milvus", status(mv.get("ok", False)))
    print_kv("GPU (torch)", status(gpu.get("torch", {}).get("ok", False)))
    print_kv("Elapsed", f"{time.time() - started:.2f}s")

    result = {
        "overall_ok": overall,
        "env": env,
        "gpu": gpu,
        "vLLM": vllm,
        "postgres": pg,
        "milvus": mv,
    }
    print("\\n== JSON ==")
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
