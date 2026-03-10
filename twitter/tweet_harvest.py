#!/usr/bin/env python3
"""
Tweet-Harvest — recolector simple de publicaciones de X/Twitter con snscrape

Características:
- Búsqueda por consulta (palabras clave, filtros X) o por usuario
- Filtros de idioma, fecha desde/hasta, máximo de items
- Salida en JSONL o CSV
- Reanudación incremental por consulta/usuario con archivo de estado (since_id)
- Deduplicación básica por id en la corrida actual

Requisitos:
    pip install snscrape pyyaml

Ejemplos:
    # Buscar en español, guardar como JSONL
    python tweet_harvest.py search \
        --q "inteligencia artificial AND argentina lang:es" \
        --since 2025-08-01 --until 2025-09-09 \
        --max 500 --out ./salida/tweets_ia_ar.jsonl

    # Descargar del usuario @ponce (últimos 300)
    python tweet_harvest.py user --username ponce --max 300 --out ./salida/ponce.csv

Notas importantes:
- snscrape accede a contenido público y puede dejar de funcionar si X cambia su sitio.
- Respeta Términos/ToS locales y de la plataforma; verifica requisitos legales antes de recolectar/usar datos.
- Si prefieres API oficial (X API v2) para mayor estabilidad/términos, este script se puede extender con un backend "api".
"""
from __future__ import annotations

import argparse
import csv
import datetime as dt
import hashlib
import json
from pathlib import Path
from typing import Dict, Iterable, Optional
import time

try:
    import snscrape.modules.twitter as sntwitter  # type: ignore
except Exception as e:  # pragma: no cover
    raise SystemExit("ERROR: Necesitas instalar snscrape: pip install snscrape") from e


ISO_DATE = "%Y-%m-%d"
DEFAULT_STATE = Path(".tweet_harvest_state.json")


# ---------------------------- Utilidades ---------------------------- #

def _hash_key(s: str) -> str:
    """Return a truncated SHA-1 hex digest used as a state-store key."""
    return hashlib.sha1(s.encode("utf-8")).hexdigest()[:16]


def _ensure_parent(path: Path) -> None:
    """Create parent directories for *path* if they do not exist."""
    path.parent.mkdir(parents=True, exist_ok=True)


def _dt_or_none(s: Optional[str]) -> Optional[str]:
    """Convert a value to an ISO-format datetime string, or return None."""
    if not s:
        return None
    if hasattr(s, "isoformat"):
        return s.isoformat()  # type: ignore
    try:
        return dt.datetime.fromisoformat(str(s)).isoformat()
    except Exception:
        return str(s)


# ---------------------------- Estado incremental ---------------------------- #

class StateStore:
    """Persistent JSON file that stores the last-seen tweet ID per query for incremental harvesting."""

    def __init__(self, path: Path):
        self.path = Path(path)
        if self.path.exists():
            try:
                self._data: Dict[str, Dict[str, str]] = json.loads(self.path.read_text(encoding="utf-8"))
            except Exception:
                self._data = {}
        else:
            self._data = {}

    def get_last_id(self, key: str) -> Optional[int]:
        """Retrieve the last-seen tweet ID for the given query key."""
        k = _hash_key(key)
        v = self._data.get(k, {}).get("since_id")
        return int(v) if v is not None else None

    def set_last_id(self, key: str, since_id: int) -> None:
        """Store the last-seen tweet ID for the given query key."""
        k = _hash_key(key)
        self._data.setdefault(k, {})["since_id"] = str(since_id)

    def save(self) -> None:
        """Persist the state dictionary to disk as JSON."""
        _ensure_parent(self.path)
        self.path.write_text(json.dumps(self._data, ensure_ascii=False, indent=2), encoding="utf-8")


# ---------------------------- Conversión ---------------------------- #

def tweet_to_dict(t) -> Dict:
    """Mapea Tweet de snscrape a un dict plano razonable."""
    d = {
        "id": str(getattr(t, "id", None)),
        "date": _dt_or_none(getattr(t, "date", None)),
        "url": getattr(t, "url", None),
        "content": getattr(t, "rawContent", None),
        "lang": getattr(t, "lang", None),
        "source": getattr(t, "sourceLabel", None),
        "user_id": str(getattr(getattr(t, "user", None), "id", None)) if getattr(t, "user", None) else None,
        "user_username": getattr(getattr(t, "user", None), "username", None) if getattr(t, "user", None) else None,
        "user_displayname": getattr(getattr(t, "user", None), "displayname", None) if getattr(t, "user", None) else None,
        "replyCount": getattr(t, "replyCount", None),
        "retweetCount": getattr(t, "retweetCount", None),
        "likeCount": getattr(t, "likeCount", None),
        "quoteCount": getattr(t, "quoteCount", None),
        "inReplyToTweetId": str(getattr(t, "inReplyToTweetId", None)) if getattr(t, "inReplyToTweetId", None) else None,
        "inReplyToUser": getattr(getattr(t, "inReplyToUser", None), "username", None) if getattr(t, "inReplyToUser", None) else None,
        "hashtags": list(getattr(t, "hashtags", []) or []),
        "cashtags": list(getattr(t, "cashtags", []) or []),
        "mentions": [getattr(u, "username", None) for u in (getattr(t, "mentionedUsers", None) or [])],
        "place": getattr(getattr(t, "place", None), "fullName", None) if getattr(t, "place", None) else None,
        "coordinates": getattr(t, "coordinates", None),
        "media": [getattr(m, "fullUrl", None) for m in (getattr(t, "media", None) or []) if hasattr(m, "fullUrl")],
        "is_retweet": getattr(t, "retweetedTweet", None) is not None,
        "quotedTweetId": str(getattr(getattr(t, "quotedTweet", None), "id", None)) if getattr(t, "quotedTweet", None) else None,
    }
    return d


# ---------------------------- Recolectores ---------------------------- #

def _build_search_query(base_q: str, since: Optional[str], until: Optional[str]) -> str:
    """Combine the base query with optional since/until date operators."""
    parts = [base_q.strip()]
    if since:
        parts.append(f"since:{since}")
    if until:
        parts.append(f"until:{until}")
    return " ".join(p for p in parts if p)


def harvest_search(
    base_query: str,
    since: Optional[str],
    until: Optional[str],
    max_items: int,
    since_id: Optional[int] = None,
    throttle_sec: float = 0.0,
) -> Iterable[Dict]:
    """Yield tweet dicts from a Twitter search query via snscrape."""
    query = _build_search_query(base_query, since, until)
    scraper = sntwitter.TwitterSearchScraper(query)
    count = 0
    for t in scraper.get_items():
        # Reanudación por since_id: cortamos cuando encontramos ids previos o excedemos el máximo
        if since_id is not None and getattr(t, "id", 0) <= since_id:
            break
        yield tweet_to_dict(t)
        count += 1
        if count >= max_items:
            break
        if throttle_sec > 0:
            time.sleep(throttle_sec)


def harvest_user(
    username: str,
    max_items: int,
    since_id: Optional[int] = None,
    throttle_sec: float = 0.0,
) -> Iterable[Dict]:
    """Yield tweet dicts from a specific user's timeline via snscrape."""
    scraper = sntwitter.TwitterUserScraper(username)
    count = 0
    for t in scraper.get_items():
        if since_id is not None and getattr(t, "id", 0) <= since_id:
            break
        yield tweet_to_dict(t)
        count += 1
        if count >= max_items:
            break
        if throttle_sec > 0:
            time.sleep(throttle_sec)


# ---------------------------- Salida ---------------------------- #

def write_jsonl(path: Path, rows: Iterable[Dict]) -> int:
    """Append rows as newline-delimited JSON to *path* and return the count."""
    _ensure_parent(path)
    n = 0
    with path.open("a", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
            n += 1
    return n


def write_csv(path: Path, rows: Iterable[Dict]) -> int:
    """Append rows as CSV to *path* (writing headers if the file is new) and return the count."""
    _ensure_parent(path)
    rows_iter = iter(rows)
    try:
        first = next(rows_iter)
    except StopIteration:
        return 0

    fieldnames = list(first.keys())
    n = 0
    with path.open("a", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if f.tell() == 0:  # archivo nuevo
            writer.writeheader()
        writer.writerow(first)
        n += 1
        for r in rows_iter:
            writer.writerow(r)
            n += 1
    return n


# ---------------------------- CLI ---------------------------- #

def parse_args() -> argparse.Namespace:
    """Parse CLI arguments for the search and user sub-commands."""
    p = argparse.ArgumentParser(
        description="Tweet-Harvest: recolector de publicaciones de X/Twitter con snscrape",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    sub = p.add_subparsers(dest="cmd", required=True)

    p_search = sub.add_parser("search", help="Buscar por consulta (X search)")
    p_search.add_argument("--q", required=True, help="Consulta: usa operadores de X (AND, OR, from:, to:, lang:es, etc.)")
    p_search.add_argument("--since", help="Fecha desde (YYYY-MM-DD)")
    p_search.add_argument("--until", help="Fecha hasta (YYYY-MM-DD, exclusivo)")
    p_search.add_argument("--max", type=int, default=500, help="Máximo de publicaciones a recolectar")
    p_search.add_argument("--out", required=True, help="Ruta de salida (extensión .jsonl o .csv)")
    p_search.add_argument("--state", default=str(DEFAULT_STATE), help="Archivo de estado para reanudar")
    p_search.add_argument("--no-resume", action="store_true", help="Ignorar estado previo y no reanudar")
    p_search.add_argument("--throttle", type=float, default=0.0, help="Pausa (seg) entre items para ser amable")

    p_user = sub.add_parser("user", help="Descargar publicaciones de un usuario")
    p_user.add_argument("--username", required=True, help="Nombre de usuario sin @")
    p_user.add_argument("--max", type=int, default=500, help="Máximo de publicaciones a recolectar")
    p_user.add_argument("--out", required=True, help="Ruta de salida (extensión .jsonl o .csv)")
    p_user.add_argument("--state", default=str(DEFAULT_STATE), help="Archivo de estado para reanudar")
    p_user.add_argument("--no-resume", action="store_true", help="Ignorar estado previo y no reanudar")
    p_user.add_argument("--throttle", type=float, default=0.0, help="Pausa (seg) entre items para ser amable")

    return p.parse_args()


def main() -> None:
    """Entry point: parse args, harvest tweets, write output, and update state."""
    args = parse_args()
    out_path = Path(args.out)
    writer = write_jsonl if out_path.suffix.lower() == ".jsonl" else write_csv

    state = StateStore(Path(args.state))

    if args.cmd == "search":
        key = f"search::{args.q}::{args.since or ''}::{args.until or ''}"
        last_id = None if args.no_resume else state.get_last_id(key)
        rows = harvest_search(
            base_query=args.q,
            since=args.since,
            until=args.until,
            max_items=args.max,
            since_id=last_id,
            throttle_sec=args.throttle,
        )
        # Para actualizar since_id necesitamos conocer el mayor id descargado
        max_seen = last_id or 0
        buffered = []
        for r in rows:
            try:
                rid = int(r.get("id") or 0)
                if rid > max_seen:
                    max_seen = rid
            except Exception:
                pass
            buffered.append(r)
        n = writer(out_path, buffered)
        if n > 0:
            state.set_last_id(key, max_seen)
            state.save()
        print(f"OK: guardados {n} items en {out_path}")
        if not args.no_resume:
            print(f"Estado actualizado en {state.path}")

    elif args.cmd == "user":
        key = f"user::{args.username}"
        last_id = None if args.no_resume else state.get_last_id(key)
        rows = harvest_user(
            username=args.username,
            max_items=args.max,
            since_id=last_id,
            throttle_sec=args.throttle,
        )
        max_seen = last_id or 0
        buffered = []
        for r in rows:
            try:
                rid = int(r.get("id") or 0)
                if rid > max_seen:
                    max_seen = rid
            except Exception:
                pass
            buffered.append(r)
        n = writer(out_path, buffered)
        if n > 0:
            state.set_last_id(key, max_seen)
            state.save()
        print(f"OK: guardados {n} items en {out_path}")
        if not args.no_resume:
            print(f"Estado actualizado en {state.path}")


if __name__ == "__main__":
    main()

