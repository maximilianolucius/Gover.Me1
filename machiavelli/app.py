"""Flask WebSocket API and main production server for the Machiavelli platform.

Provides endpoints for real-time audio transcription with fact-checking,
RAG-based query processing, political communication visualizations (radar charts,
media affinity, coverage), survey/polling management, and social monitoring.
"""

import warnings
warnings.filterwarnings("ignore", message="No artists with labels found")
warnings.filterwarnings("ignore", message="Tight layout not applied")
warnings.filterwarnings("ignore", message="datetime.datetime.utcnow()")

from dotenv import load_dotenv

# Load environment variables
load_dotenv('elysia/.elysia_env')
load_dotenv('.env')
load_dotenv()

import io
import json
import logging
import os, sys
import re
import threading
import subprocess
import time
import uuid
import wave
from dataclasses import dataclass
from datetime import datetime
from threading import Timer, Lock
from urllib.parse import urlparse
import eventlet
import numpy as np
from PIL import Image
import requests
from flask import Flask, request, Response, jsonify, send_file
from flask_socketio import SocketIO, emit

import matplotlib
matplotlib.use("Agg")
from matplotlib import pyplot as plt
from matplotlib.patches import FancyBboxPatch
from matplotlib.offsetbox import OffsetImage, AnnotationBbox

from ws_utils.rag_query_adapter import run_conversational_query




LOGO_PATHS = {
    "ABC Sevilla":            "./api_images/logos/abc_sevilla.png",
    "El País":                "./api_images/logos/el_pais.png",
    "eldiario.es Andalucía":  "./api_images/logos/eldiario_andalucia.png",
    "Diario SUR":             "./api_images/logos/diario_sur.png",
    "Diario de Sevilla":      "./api_images/logos/diario_de_sevilla.png",
    "El Correo de Andalucía": "./api_images/logos/el_correo_andalucia.png",
    "IDEAL":                  "./api_images/logos/ideal.png",
}

# Add to the constants section at the top of the file
MESSAGE_EVALUATIONS_CONST = [
    {
        "id": "msg_001",
        "text": "Se entiende, pero falta un golpe de efecto final.",
        "score": 63
    },
    {
        "id": "msg_002",
        "text": "Correcto, aunque no deja huella.",
        "score": 39
    },
    {
        "id": "msg_003",
        "text": "Funciona en contexto parlamentario, no tanto en medios.",
        "score": 17
    },
    {
        "id": "msg_004",
        "text": "Mensaje claro y directo, resuena con la audiencia objetivo.",
        "score": 82
    },
    {
        "id": "msg_005",
        "text": "Propuesta sólida pero con poca visibilidad en redes.",
        "score": 55
    },
]

@dataclass
class Config:
    """Application configuration loaded from environment variables."""

    ELEVENLABS_API_KEY: str = os.getenv("ELEVENLABS_API_KEY", "sk_e38cc2b727b1785beda86dd43c056ab5f9847cba571153ee")


config = Config()

# Import the actual RAG query functionality
try:
    from machiavelli.query import run_query_raw_for_fact_check

    RAG_AVAILABLE = True
except ImportError as e:
    print(f"⚠️ RAG functionality not available: {e}")
    RAG_AVAILABLE = False

# Import fact-checking functionality
try:
    from fact_check.machiavelli_factchecker_cli import classify_paragraph

    # Try to import RAGChatbot from multiple possible modules
    CHATBOT_CLASS = None
    import importlib
    for module_name in ["rag.chatbot", "machiavelli.query", "rag_chatbot_core"]:
        try:
            module = importlib.import_module(module_name)
            CHATBOT_CLASS = getattr(module, "RAGChatbot")
            break
        except (ImportError, AttributeError):
            continue

    FACTCHECK_AVAILABLE = CHATBOT_CLASS is not None
    print(f"📋 Fact-checking: {'✅ Available' if FACTCHECK_AVAILABLE else '❌ Not Available'}")
except ImportError as e:
    print(f"⚠️ Fact-checking functionality not available: {e}")
    FACTCHECK_AVAILABLE = False
    CHATBOT_CLASS = None

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Reduce engineio logging to reduce audio chunk spam
engineio_logger = logging.getLogger('engineio.server')
engineio_logger.setLevel(logging.WARNING)
socketio_logger = logging.getLogger('socketio.server')
socketio_logger.setLevel(logging.WARNING)

app = Flask(__name__)
app.config['SECRET_KEY'] = 'mock_secret_key'
# FIXED: Better SocketIO configuration with reduced logging
socketio = SocketIO(
    app,
    cors_allowed_origins="*",
    async_mode='eventlet',  # Explicitly set async mode
    logger=False,  # Disable SocketIO logger
    engineio_logger=False,  # Disable engineio logger
    ping_timeout=60,
    ping_interval=25
)

# In-memory storage
jobs = {}
scribe = {}
sessions = {}
audio_jobs = {}
encuestas = {}

# Audio transcription storage - UPDATED for ElevenLabs
active_transcribers = {}
chunk_counters = {}
session_job_ids = {}  # Map session_id -> job_id


# Data storage for political communication endpoints
data_lock = Lock()
file_lock = Lock()

radar_data = {}
coverage_data = {}
affinity_data = {}

last_reload = None
last_json_load_time = time.time()

IMAGES_DIR = './api_images'

SEVERITY = {
    "alta": {"label": "Alta", "color": "#e74c3c"},
    "media": {"label": "Media", "color": "#f1c40f"},
    "baja": {"label": "Baja", "color": "#2ecc71"},
}

TOPIC_LABELS = {
    "vivienda": "Vivienda",
    "educacion": "Educación",
    "sanidad": "Sanidad",
    "economia": "Economía",
    "seguridad": "Seguridad",
    "transporte": "Transporte",
}

AXES_BY_TOPIC = {
    "vivienda": [
        "Precio", "Disponibilidad", "Oferta pública",
        "Nuevas construcciones", "Costo a crédito"
    ],
    "economia": [
        "Inflación", "Empleo", "Crecimiento del PIB",
        "Salario real", "Inversión"
    ],
    "sanidad": [
        "Listas de espera", "Atención primaria",
        "Cobertura", "Personal sanitario",
        "Infraestructura hospitalaria"
    ],
    "seguridad": [
        "Delitos", "Respuesta policial",
        "Percepción de seguridad", "Recursos y equipamiento",
        "Eficacia judicial"
    ],
    "educacion": [
        "Resultados académicos", "Infraestructura",
        "Docentes", "Acceso y becas", "Digitalización"
    ],
    "transporte": [
        "Oferta y frecuencia", "Infraestructura vial",
        "Puntualidad y confiabilidad", "Accesibilidad",
        "Seguridad vial"
    ],
}

ALERTS_CONST = [
    {
        "id": "alquiler_costes",
        "severity": "alta",
        "topic": "vivienda",
        "title": "Narrativa crítica en medios sobre costes del alquiler",
    },
    {
        "id": "oposicion_edu_publica",
        "severity": "media",
        "topic": "educacion",
        "title": "Declaraciones del portavoz de la oposición sobre educación pública",
    },
    {
        "id": "cobertura_ayudas_alquiler",
        "severity": "baja",
        "topic": "vivienda",
        "title": "Cobertura insuficiente en medios sobre las ayudas al alquiler",
    },
]

MEDIA_URLS = [
    "https://www.abc.es/sevilla/",
    "https://www.diariosur.es/",
    "https://www.diariodesevilla.es",
    "https://www.elcorreoweb.es/andalucia/",
    "https://www.eldiario.es/andalucia/",
    "https://www.ideal.es",
    "https://elpais.com/?ed=es",
]

NAME_OVERRIDES = {
    "abc.es": "ABC Sevilla",
    "diariosur.es": "Diario SUR",
    "diariodesevilla.es": "Diario de Sevilla",
    "elcorreoweb.es": "El Correo de Andalucía",
    "eldiario.es": "eldiario.es Andalucía",
    "ideal.es": "IDEAL",
    "elpais.com": "El País",
}


def run_news_enricher():
    """Ejecuta el script news_enricher_run.py en el entorno virtual actual"""
    try:
        logger.info("🔄 Ejecutando news_enricher_run.py...")

        # Usamos el mismo Python del entorno virtual
        python_executable = sys.executable

        # Ejecutar el script
        result = subprocess.run(
            [python_executable, "news_enricher_run.py"],
            capture_output=True,
            text=True,
            cwd=os.getcwd(),  # Ejecutar en el mismo directorio
            timeout=3600  # Timeout de 1 hora por seguridad
        )

        if result.returncode == 0:
            logger.info("✅ news_enricher_run.py completado exitosamente")
            logger.info(f"Salida: {result.stdout}")

            # Recargar los datos después de la ejecución exitosa
            logger.info("🔄 Recargando datos JSON actualizados...")
            load_json_data()
        else:
            logger.error(f"❌ Error en news_enricher_run.py: {result.stderr}")

    except subprocess.TimeoutExpired:
        logger.error("⏰ news_enricher_run.py excedió el tiempo límite de 1 hora")
    except Exception as e:
        logger.error(f"❌ Error ejecutando news_enricher_run.py: {e}")


def news_enricher_scheduler():
    """Programa la ejecución de news_enricher_run.py cada 2 horas"""
    while True:
        try:
            # Ejecutar inmediatamente al inicio
            run_news_enricher()

            # Esperar 2 horas (7200 segundos) antes de la próxima ejecución
            logger.info("⏰ Programando próxima ejecución de news_enricher_run.py en 2 horas...")
            time.sleep(7200)

        except Exception as e:
            logger.error(f"❌ Error en el scheduler de news_enricher: {e}")
            # Esperar 10 minutos antes de reintentar si hay error
            time.sleep(600)


def start_news_enricher_scheduler():
    """Inicia el scheduler en un hilo separado"""
    scheduler_thread = threading.Thread(
        target=news_enricher_scheduler,
        daemon=True,  # Hilo daemon para que se cierre cuando la app se cierre
        name="NewsEnricherScheduler"
    )
    scheduler_thread.start()
    logger.info("🚀 Scheduler de news_enricher iniciado (ejecución cada 2 horas)")


def infer_media_name(url: str) -> str:
    """Derive a human-readable media outlet name from a URL."""
    host = urlparse(url).netloc.lower().split(":")[0]
    host = host[4:] if host.startswith("www.") else host
    if host in NAME_OVERRIDES:
        return NAME_OVERRIDES[host]
    core = host.split(".")[0]
    return core.capitalize()


MEDIA_SOURCES = [(infer_media_name(u), u) for u in MEDIA_URLS]


# ==== JSON data utilities for political communication endpoints ====
def load_json_data():
    """Load JSON files into memory and pre-generate static images."""
    global radar_data, coverage_data, affinity_data, last_reload, last_json_load_time

    try:
        with file_lock:
            if os.path.exists('./news_enricher/output/radar_scores.json'):
                with open('./news_enricher/output/radar_scores.json', 'r', encoding='utf-8') as f:
                    radar_data = json.load(f)
                print("✓ Cargado radar_scores.json")

            if os.path.exists('./news_enricher/output/media_coverage.json'):
                with open('./news_enricher/output/media_coverage.json', 'r', encoding='utf-8') as f:
                    coverage_data = json.load(f)
                print("✓ Cargado media_coverage.json")

            if os.path.exists('./news_enricher/output/media_affinity_scores.json'):
                with open('./news_enricher/output/media_affinity_scores.json', 'r', encoding='utf-8') as f:
                    affinity_data = json.load(f)
                print("✓ Cargado media_affinity_scores.json")

            last_reload = datetime.utcnow()
            last_json_load_time = time.time()
            print(f"✓ Datos recargados a las {last_reload.isoformat()}")

            os.makedirs(IMAGES_DIR, exist_ok=True)
            print(f"✓ Directorio de imágenes: {IMAGES_DIR}")

            print("\n🎨 Generando imágenes pre-calculadas...")
            generate_all_images()
            print("✓ Todas las imágenes generadas")

    except Exception as exc:
        print(f"✗ Error cargando datos: {exc}")


def generate_all_images():
    """Generate reusable image assets for radar, affinity, coverage, and alerts."""
    topics = ['vivienda', 'educacion', 'sanidad', 'economia', 'seguridad', 'transporte']
    windows = ['7d', '30d', '90d']
    parties = ['pp', 'vox', 'psoe', 'programa']

    print("  → Generando imágenes de radar...")
    for topic in topics:
        for window in windows:
            data = get_radar_scores(topic=topic, window=window, parties=parties)
            img_data = radar_png(data["axes"], data["series"])
            filename = f"{IMAGES_DIR}/radar_{topic}_{window}.png"
            with open(filename, 'wb') as file_out:
                file_out.write(img_data)
    print(f"    ✓ {len(topics) * len(windows)} imágenes de radar generadas")

    print("  → Generando imágenes de media affinity...")
    for window in windows:
        points = get_affinity_scores(window)
        img_data = scatter_png(points)
        filename = f"{IMAGES_DIR}/media_affinity_{window}.png"
        with open(filename, 'wb') as file_out:
            file_out.write(img_data)
    print(f"    ✓ {len(windows)} imágenes de media affinity generadas")

    print("  → Generando imágenes de media coverage...")
    all_topics = list(TOPIC_LABELS.keys())
    sorts = ['name', 'total', 'positive', 'negative']

    for window in windows:
        for sort in sorts:
            rows = get_coverage_data(all_topics, window)
            if sort == "total":
                rows.sort(key=lambda row: row["total"], reverse=True)
            elif sort in ("positive", "negative"):
                rows.sort(key=lambda row: row["counts"][sort], reverse=True)
            else:
                rows.sort(key=lambda row: row["label"])

            img_data = stacked_bars_png_coverage(rows)
            filename = f"{IMAGES_DIR}/media_coverage_{window}_{sort}.png"
            with open(filename, 'wb') as file_out:
                file_out.write(img_data)
    print(f"    ✓ {len(windows) * len(sorts)} imágenes de media coverage generadas")

    print("  → Generando imagen de alerts...")
    rows = ALERTS_CONST[:3]
    img_data = cards_png(rows)
    filename = f"{IMAGES_DIR}/alerts_default.png"
    with open(filename, 'wb') as file_out:
        file_out.write(img_data)
    print("    ✓ 1 imagen de alerts generada")


def check_and_reload_if_needed():
    """Reload JSON data every hour when endpoints are hit."""
    global last_json_load_time
    if time.time() - last_json_load_time > 3600:
        print("\n⟳ Recargando datos (1 hora transcurrida)...")
        load_json_data()


def normalize_window(window: str) -> str:
    """Normalize window parameter to 7d, 30d, or 90d."""
    if window.endswith("d") and window[:-1].isdigit():
        days = int(window[:-1])
        if days <= 7:
            return "7d"
        if days <= 30:
            return "30d"
        return "90d"
    return "30d"


def get_radar_scores(topic: str, window: str, parties: list[str]) -> dict:
    """Return radar chart scores from cached JSON."""
    win = normalize_window(window)

    with data_lock:
        if win not in radar_data or topic not in radar_data[win]:
            axes = AXES_BY_TOPIC.get(topic, [])
            return {
                "axes": axes,
                "series": {party: [50] * len(axes) for party in parties}
            }

        axes = AXES_BY_TOPIC.get(topic, [])
        topic_data = radar_data[win][topic]
        return {
            "axes": axes,
            "series": {party: topic_data.get(party, [50] * len(axes)) for party in parties}
        }


def get_coverage_data(topics: list[str], window: str) -> list:
    """Return media coverage counts for given topics and window."""
    win = normalize_window(window)

    with data_lock:
        if win not in coverage_data:
            return [{
                "key": topic,
                "label": TOPIC_LABELS.get(topic, topic),
                "counts": {"positive": 0, "neutral": 0, "negative": 0},
                "total": 0,
            } for topic in topics]

        all_data = coverage_data[win]
        result = []
        for item in all_data:
            if item["key"] in topics:
                result.append(item)
        return result


def get_affinity_scores(window: str) -> list:
    """Return media affinity scores for requested window."""
    win = normalize_window(window)

    with data_lock:
        if win not in affinity_data:
            return [{
                "name": name,
                "url": url,
                "affinity": 50,
                "reach": 50,
            } for name, url in MEDIA_SOURCES]

        data = affinity_data[win]
        points = []

        for name, url in MEDIA_SOURCES:
            key = name.replace(" ", "").replace(".", "").lower()
            found = False

            for json_key, values in data.items():
                if json_key.lower().replace(" ", "").replace(".", "") == key:
                    points.append({
                        "name": name,
                        "url": url,
                        "affinity": values.get("affinity", 50),
                        "reach": values.get("reach", 50),
                    })
                    found = True
                    break

            if not found:
                points.append({"name": name, "url": url, "affinity": 50, "reach": 50})

        return points


def radar_png(axes, series):
    """Render a polar radar chart as PNG bytes for the given axes and party series."""
    angles = np.linspace(0, 2 * np.pi, len(axes), endpoint=False)
    angles = np.concatenate([angles, angles[:1]])
    fig = plt.figure(figsize=(5, 5), dpi=180)
    ax = plt.subplot(111, polar=True)
    ax.set_theta_offset(np.pi / 2)
    ax.set_theta_direction(-1)
    ax.set_thetagrids(angles[:-1] * 180 / np.pi, axes)
    ax.set_ylim(0, 100)
    ax.grid(True, linewidth=0.5)

    palette = {"pp": "#1f77b4", "vox": "#2ca02c", "psoe": "#ff7f0e", "programa": "#111111"}
    for name, values in series.items():
        vals = np.array(values, float)
        vals = np.concatenate([vals, vals[:1]])
        lw = 2.5 if name.lower() == "programa" else 1.8
        ls = "--" if name.lower() == "programa" else "-"
        ax.plot(angles, vals, linewidth=lw, linestyle=ls, color=palette.get(name.lower()))
        if name.lower() != "programa":
            ax.fill(angles, vals, alpha=0.10, color=palette.get(name.lower()))

    ax.legend(loc="lower center", bbox_to_anchor=(0.5, -0.15), ncol=4, frameon=False)
    buf = io.BytesIO()
    plt.tight_layout()
    plt.savefig(buf, format="png")
    plt.close(fig)
    buf.seek(0)
    return buf.read()


def stacked_bars_png_coverage(rows):
    """Render a horizontal stacked-bar chart of media coverage as PNG bytes."""
    labels = [row["label"] for row in rows]
    pos = [row["counts"]["positive"] for row in rows]
    neu = [row["counts"]["neutral"] for row in rows]
    neg = [row["counts"]["negative"] for row in rows]

    fig = plt.figure(figsize=(7.5, 2.6 + 0.35 * len(rows)), dpi=180)
    ax = plt.subplot(111)

    y_positions = range(len(rows))
    ax.barh(y_positions, pos, label="Positivo", color="#2ecc71")
    ax.barh(y_positions, neu, left=pos, label="Neutro", color="#f1c40f")
    ax.barh(y_positions, neg, left=[p + n for p, n in zip(pos, neu)], label="Negativo", color="#e74c3c")

    for idx in y_positions:
        if pos[idx] > 0:
            ax.text(pos[idx] / 2, idx, str(pos[idx]), va="center", ha="center", fontsize=8, color="white")
        if neu[idx] > 0:
            ax.text(pos[idx] + neu[idx] / 2, idx, str(neu[idx]), va="center", ha="center", fontsize=8, color="black")
        if neg[idx] > 0:
            ax.text(pos[idx] + neu[idx] + neg[idx] / 2, idx, str(neg[idx]), va="center", ha="center", fontsize=8, color="white")

    ax.set_yticks(list(y_positions))
    ax.set_yticklabels(labels)
    ax.set_xlabel("nº de menciones")
    ax.set_xlim(left=0)
    ax.grid(axis="x", alpha=0.3, linewidth=0.6)
    ax.legend(loc="lower right", frameon=False, ncol=3)
    plt.tight_layout()

    buf = io.BytesIO()
    plt.savefig(buf, format="png")
    plt.close(fig)
    buf.seek(0)
    return buf.read()


def scatter_png(points, logos=LOGO_PATHS, logo_px=40, show_labels=False):
    """Render a media affinity scatter plot as a transparent PNG with optional logos."""
    fig = plt.figure(figsize=(5.5, 4.0), dpi=180)
    fig.patch.set_alpha(0)                 # <- fondo del figure transparente
    ax = plt.subplot(111)
    ax.set_facecolor('none')               # <- fondo de ejes transparente

    ax.set_xlim(0, 100);
    ax.set_ylim(0, 100)
    ax.set_xlabel("Afinidad argumental")
    ax.set_ylabel("Alcance")
    ax.grid(True, linewidth=0.4, alpha=0.5)

    # Si falta un logo, caemos al punto normal:
    def _draw_fallback(x, y):
        ax.scatter([x], [y], s=60, zorder=2)

    for p in points:
        x, y = p["affinity"], p["reach"]
        path = (logos or {}).get(p["name"])

        if path and os.path.exists(path):
            try:
                im = np.asarray(Image.open(path).convert("RGBA"))
                h, w = im.shape[:2]
                zoom = logo_px / float(max(h, w))   # escala para un tamaño ~logo_px
                ab = AnnotationBbox(
                    OffsetImage(im, zoom=zoom),
                    (x, y),
                    frameon=False,
                    box_alignment=(0.5, 0.5),
                    zorder=3,
                )
                ax.add_artist(ab)
            except Exception:
                _draw_fallback(x, y)
        else:
            _draw_fallback(x, y)

        if show_labels:
            ax.annotate(p["name"], (x, y), textcoords="offset points",
                        xytext=(5, 5), fontsize=7)

    buf = io.BytesIO()
    plt.tight_layout()
    plt.savefig(buf, format="png", transparent=True)  # <- PNG con alpha
    plt.close(fig)
    buf.seek(0)
    return buf.read()


def cards_png(alerts):
    """Render alert cards as a PNG image with severity badges and topic labels."""
    total = len(alerts)
    fig_w = 8.0
    card_w = (fig_w - 1.2) / max(total, 1)
    fig_h = 2.8
    fig = plt.figure(figsize=(fig_w, fig_h), dpi=180)
    ax = plt.subplot(111)
    ax.axis("off")

    left = 0.4
    gap = 0.2
    card_h = 1.7
    y_pos = 0.6

    for idx, alert in enumerate(alerts):
        x_pos = left + idx * (card_w + gap)
        box = FancyBboxPatch(
            (x_pos, y_pos), card_w, card_h,
            boxstyle="round,pad=0.02,rounding_size=0.06",
            linewidth=0.8, edgecolor="#e8e8f5", facecolor="#f6f6ff"
        )
        ax.add_patch(box)

        severity = SEVERITY[alert["severity"]]
        sev_box = FancyBboxPatch(
            (x_pos + 0.15, y_pos + card_h - 0.4), 0.5, 0.22,
            boxstyle="round,pad=0.02,rounding_size=0.05",
            facecolor=severity["color"], edgecolor="none"
        )
        ax.add_patch(sev_box)
        ax.text(x_pos + 0.4, y_pos + card_h - 0.29, severity["label"], ha="center", va="center", fontsize=7, color="white")

        topic_label = TOPIC_LABELS.get(alert["topic"], alert["topic"].capitalize())
        topic_box = FancyBboxPatch(
            (x_pos + 0.72, y_pos + card_h - 0.4), 0.8, 0.22,
            boxstyle="round,pad=0.02,rounding_size=0.05",
            facecolor="#6c6ea9", edgecolor="none", alpha=0.85
        )
        ax.add_patch(topic_box)
        ax.text(x_pos + 1.12, y_pos + card_h - 0.29, topic_label, ha="center", va="center", fontsize=7, color="white")

        ax.text(x_pos + 0.15, y_pos + card_h - 0.65, alert["title"], ha="left", va="top", fontsize=9, color="#2b2b2b", wrap=True)

    buf = io.BytesIO()
    plt.tight_layout()
    plt.savefig(buf, format="png", bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)
    return buf.read()

class ElevenLabsTranscriber:
    """ElevenLabs Scribe transcription handler with robust gap-free processing"""

    def __init__(self, session_id, api_key):
        """Initialize transcriber with session ID, API key, and default audio settings."""
        self.session_id = session_id
        self.api_key = api_key
        self.audio_chunks = []
        self.sample_rate = None
        self.is_recording = False
        self.total_frames = 0

        # Robust processing variables
        self.pseudo_realtime_timer = None
        self.pseudo_transcriptions = []
        self.fragment_duration = 4.0  # Duración de cada fragmento
        self.overlap_duration = 1.0  # Overlap entre fragmentos
        self.min_fragment_duration = 2.0  # Mínimo para evitar chunks pequeños
        self.last_processed_samples = 0  # Samples ya procesados

        # Buffer de transcripción acumulada
        self.buffer_transcribe = ""

        # ElevenLabs API endpoints
        self.base_url = "https://api.elevenlabs.io/v1/speech-to-text"
        self.headers = {
            "Accept": "application/json",
            "xi-api-key": self.api_key
        }

    def start_recording(self):
        """Start a new recording session with robust processing"""
        self.audio_chunks = []
        self.total_frames = 0
        self.is_recording = True
        self.pseudo_transcriptions = []
        self.last_processed_samples = 0
        self.buffer_transcribe = ""

        logger.info(f"🎤 ElevenLabs transcriber started for session {self.session_id}")
        self._schedule_next_fragment_processing()

    def _schedule_next_fragment_processing(self):
        """Schedule next fragment processing - more frequent checks"""
        if self.is_recording:
            # Check more frequently (every 2 seconds) for smoother processing
            self.pseudo_realtime_timer = Timer(2.0, self._process_available_fragments)
            self.pseudo_realtime_timer.start()

    def _process_available_fragments(self):
        """Process all available audio fragments that have accumulated since last check."""
        if not self.is_recording or len(self.audio_chunks) == 0:
            self._schedule_next_fragment_processing()
            return

        try:
            # Get current total audio
            combined_audio = np.concatenate(self.audio_chunks)
            total_samples = len(combined_audio)
            sample_rate = self.sample_rate or 44100

            total_duration = total_samples / sample_rate
            processed_duration = self.last_processed_samples / sample_rate

            logger.info(f"🔄 Fragment check: {total_duration:.1f}s total, {processed_duration:.1f}s processed")

            # Calculate next fragment boundaries
            while self._should_process_next_fragment(total_samples, sample_rate):
                self._process_next_fragment(combined_audio, sample_rate)

        except Exception as e:
            logger.error(f"❌ Error in fragment processing: {e}")

        # Schedule next check
        self._schedule_next_fragment_processing()

    def _should_process_next_fragment(self, total_samples, sample_rate):
        """Return True if enough new audio has accumulated for the next fragment."""
        if self.last_processed_samples == 0:
            # First fragment: process if we have at least fragment_duration
            available_duration = total_samples / sample_rate
            return available_duration >= self.fragment_duration

        # Subsequent fragments: process if we have enough new audio beyond overlap
        processed_duration = self.last_processed_samples / sample_rate
        total_duration = total_samples / sample_rate
        new_audio_duration = total_duration - processed_duration + self.overlap_duration

        # Process if we have enough new audio for a full fragment
        return new_audio_duration >= self.fragment_duration

    def _process_next_fragment(self, combined_audio, sample_rate):
        """Extract the next overlapping audio fragment, transcribe it, and update the buffer."""
        try:
            fragment_samples = int(sample_rate * self.fragment_duration)
            overlap_samples = int(sample_rate * self.overlap_duration)

            if self.last_processed_samples == 0:
                # First fragment: take first fragment_duration seconds
                start_sample = 0
                end_sample = min(fragment_samples, len(combined_audio))
                has_overlap = False
                logger.info(f"🆕 Processing FIRST fragment: 0.0s -> {end_sample / sample_rate:.1f}s")

            else:
                # Subsequent fragments: overlap + new audio
                overlap_start = max(0, self.last_processed_samples - overlap_samples)
                end_sample = min(self.last_processed_samples + fragment_samples, len(combined_audio))

                # Ensure minimum fragment duration
                actual_duration = (end_sample - overlap_start) / sample_rate
                if actual_duration < self.min_fragment_duration:
                    logger.info(f"⏭️ Skipping small fragment ({actual_duration:.1f}s < {self.min_fragment_duration}s)")
                    return

                start_sample = overlap_start
                has_overlap = True
                overlap_duration = (self.last_processed_samples - overlap_start) / sample_rate
                new_duration = (end_sample - self.last_processed_samples) / sample_rate
                logger.info(f"🔄 Processing fragment: {overlap_duration:.1f}s overlap + {new_duration:.1f}s new")

            # Extract fragment audio
            fragment_audio = combined_audio[start_sample:end_sample]
            duration = len(fragment_audio) / sample_rate

            # Send to ElevenLabs
            result = self._send_audio_to_elevenlabs(fragment_audio, is_partial=True)

            if result and result.get('text'):
                transcript = result['text'].strip()
                timestamp = time.strftime('%H:%M:%S')

                # Update buffer with intelligent merging
                self._update_buffer_transcribe(transcript, has_overlap)

                # Store transcription
                fragment_data = {
                    'timestamp': timestamp,
                    'duration': duration,
                    'text': transcript,
                    'start_sample': start_sample,
                    'end_sample': end_sample,
                    'has_overlap': has_overlap
                }
                self.pseudo_transcriptions.append(fragment_data)

                overlap_info = f" (+{self.overlap_duration}s overlap)" if has_overlap else ""
                logger.info(f"🎯 Fragment [{timestamp}] ({duration:.1f}s{overlap_info}): '{transcript[:50]}...'")

                # Send to client
                socketio.emit('transcription_update', {
                    'text': transcript,
                    'is_partial': True,
                    'timestamp': timestamp,
                    'duration': f"{duration:.1f}s",
                    'has_overlap': has_overlap,
                    'buffer_transcribe': self.buffer_transcribe
                }, room=self.session_id)

            # Update processed position (only advance by new audio, not overlap)
            if self.last_processed_samples == 0:
                self.last_processed_samples = end_sample
            else:
                # Advance by the new audio we processed (excluding overlap)
                new_samples = end_sample - self.last_processed_samples
                if new_samples > 0:
                    self.last_processed_samples += new_samples

            logger.info(
                f"📍 Advanced to sample {self.last_processed_samples} ({self.last_processed_samples / sample_rate:.1f}s)")

        except Exception as e:
            logger.error(f"❌ Error processing fragment: {e}")

    def set_sample_rate(self, sample_rate):
        """Set the sample rate for this session"""
        if self.sample_rate is None:
            self.sample_rate = int(sample_rate)
            logger.info(f"🔊 Sample rate set to {self.sample_rate}Hz for session {self.session_id}")

    def add_audio_chunk(self, audio_bytes, sample_rate=None):
        """Add audio chunk to buffer"""
        if self.is_recording:
            if sample_rate and self.sample_rate is None:
                self.set_sample_rate(sample_rate)

            if isinstance(audio_bytes, list):
                audio_bytes = bytes(audio_bytes)

            audio_array = np.frombuffer(audio_bytes, dtype=np.int16)
            self.audio_chunks.append(audio_array)
            self.total_frames += len(audio_array)

    def stop_recording_and_transcribe(self):
        """Stop recording and ensure complete coverage"""
        if not self.is_recording:
            return None

        # Cancel timer
        if self.pseudo_realtime_timer:
            self.pseudo_realtime_timer.cancel()
            logger.info("ℹ️ Fragment processing timer cancelled")

        if not self.audio_chunks:
            logger.warning("No audio chunks to transcribe")
            self.is_recording = False
            return None

        if self.sample_rate is None:
            self.sample_rate = 48000
            logger.warning(f"No sample rate detected, using default: {self.sample_rate}Hz")

        try:
            # Step 1: Process any remaining unprocessed fragments
            self._process_remaining_fragments()

            # Step 2: Always process final 4 seconds as requested
            self._process_final_fragment()

            self.is_recording = False

            return self.buffer_transcribe

        except Exception as e:
            logger.error(f"❌ Error in final transcription: {e}")
            return None



    def _process_remaining_fragments(self):
        """Process any remaining unprocessed audio"""
        try:
            combined_audio = np.concatenate(self.audio_chunks)
            total_samples = len(combined_audio)
            sample_rate = self.sample_rate

            # Process remaining fragments until we've covered most of the audio
            while self._should_process_next_fragment(total_samples, sample_rate):
                self._process_next_fragment(combined_audio, sample_rate)

            logger.info(
                f"🔄 Remaining fragments processed. Covered up to {self.last_processed_samples / sample_rate:.1f}s")

        except Exception as e:
            logger.error(f"❌ Error processing remaining fragments: {e}")

    def _process_final_fragment(self):
        """Process final 4 seconds as requested"""
        try:
            combined_audio = np.concatenate(self.audio_chunks)
            sample_rate = self.sample_rate
            total_duration = len(combined_audio) / sample_rate

            # Final fragment: last 4 seconds
            final_duration = 4.0
            final_samples = int(sample_rate * final_duration)

            if len(combined_audio) > final_samples:
                final_audio = combined_audio[-final_samples:]
                logger.info(f"📚 FINAL: Processing last {final_duration}s of {total_duration:.1f}s total")
            else:
                final_audio = combined_audio
                logger.info(f"📚 FINAL: Processing all audio ({total_duration:.1f}s - less than 4s)")

            # Send to ElevenLabs
            result = self._send_audio_to_elevenlabs(final_audio, is_partial=True)

            if result and result.get('text'):
                transcript = result['text'].strip()
                timestamp = time.strftime('%H:%M:%S')

                # Update buffer - always has overlap since it's the final fragment
                self._update_buffer_transcribe(transcript, has_overlap=True)

                # Store as final fragment
                final_data = {
                    'timestamp': timestamp,
                    'duration': len(final_audio) / sample_rate,
                    'text': transcript,
                    'is_final_fragment': True,
                    'has_overlap': True
                }
                self.pseudo_transcriptions.append(final_data)

                logger.info(f"🎯 FINAL [{timestamp}]: '{transcript[:50]}...' (últimos {final_duration}s)")

                # Send to client
                socketio.emit('transcription_update', {
                    'text': transcript,
                    'is_partial': True,
                    'timestamp': timestamp,
                    'duration': f"{len(final_audio) / sample_rate:.1f}s",
                    'has_overlap': True,
                    'is_final_fragment': True,
                    'buffer_transcribe': self.buffer_transcribe
                }, room=self.session_id)

        except Exception as e:
            logger.error(f"❌ Error in final fragment: {e}")

    def _update_buffer_transcribe(self, new_transcript, has_overlap):
        """Update buffer with intelligent merging"""
        if not has_overlap or not self.buffer_transcribe:
            self.buffer_transcribe = new_transcript
            logger.info(f"📝 BUFFER UPDATED (no overlap): '{self.buffer_transcribe[:50]}...'")
            return

        # Intelligent merging with overlap
        buffer_words = self.buffer_transcribe.split()
        new_words = new_transcript.split()

        if not buffer_words or not new_words:
            if new_transcript:
                self.buffer_transcribe = new_transcript
            return

        # Find word overlap
        max_overlap_words = min(len(buffer_words), len(new_words), 10)
        best_overlap = 0

        for i in range(1, max_overlap_words + 1):
            buffer_end = buffer_words[-i:]
            new_start = new_words[:i]
            if buffer_end == new_start:
                best_overlap = i

        if best_overlap > 0:
            # Merge by removing duplicate words
            new_words_to_add = new_words[best_overlap:]
            if new_words_to_add:
                self.buffer_transcribe += " " + " ".join(new_words_to_add)
            logger.info(f"🔗 MERGED: {best_overlap} overlapping words, added {len(new_words_to_add)} new")
        else:
            # Semantic overlap check
            last_word = buffer_words[-1].lower().strip('.,!?;:')
            overlap_found = False

            for i, word in enumerate(new_words[:3]):
                clean_word = word.lower().strip('.,!?;:')
                if clean_word == last_word:
                    new_words_to_add = new_words[i + 1:]
                    if new_words_to_add:
                        self.buffer_transcribe += " " + " ".join(new_words_to_add)
                    logger.info(f"🔗 SEMANTIC MERGE: Found '{clean_word}', added {len(new_words_to_add)} words")
                    overlap_found = True
                    break

            if not overlap_found:
                self.buffer_transcribe += " " + new_transcript
                logger.info("➕ NO OVERLAP: Added full transcript")

        logger.info(f"📝 BUFFER FINAL: '{self.buffer_transcribe[:100]}...'")

    def _send_audio_to_elevenlabs(self, audio_array, is_partial=False):
        """Send audio array to ElevenLabs"""
        try:
            # Create WAV file in memory
            wav_buffer = io.BytesIO()
            with wave.open(wav_buffer, 'wb') as wav_file:
                wav_file.setnchannels(1)
                wav_file.setsampwidth(2)
                wav_file.setframerate(self.sample_rate or 44100)
                wav_file.writeframes(audio_array.tobytes())

            wav_buffer.seek(0)
            return self._send_to_elevenlabs(wav_buffer, is_partial)

        except Exception as e:
            logger.error(f"❌ Error creating audio for ElevenLabs: {e}")
            return None

    def _send_to_elevenlabs(self, audio_buffer, is_partial=False):
        """Send audio to ElevenLabs API"""
        try:
            audio_buffer.seek(0)

            files = {
                "file": ("recording.wav", audio_buffer.read(), "audio/wav")
            }

            audio_buffer.seek(0)

            data = {
                "model_id": "scribe_v1",
                "language": "es",
                "timestamps_granularity": "word"
            }

            headers = {
                "Accept": "application/json",
                "xi-api-key": self.api_key
            }

            prefix = "FRAGMENT" if is_partial else "FINAL"
            response = requests.post(
                self.base_url,
                headers=headers,
                files=files,
                data=data,
                timeout=60
            )

            logger.info(f"📡 {prefix}: ElevenLabs response: {response.status_code}")

            if response.status_code == 200:
                result = response.json()
                text = result.get('text') or result.get('transcript')

                if text:
                    logger.info(f"🎯 {prefix}: '{text[:50]}...'")
                    return {'text': text, 'full_response': result}
                else:
                    logger.error(f"❌ {prefix}: No text in response")
                    return None
            else:
                logger.error(f"❌ {prefix}: API error {response.status_code}")
                return None

        except Exception as e:
            logger.error(f"❌ Error calling ElevenLabs: {e}")
            return None

    def get_buffer_transcribe(self):
        """Get the current accumulated buffer transcription"""
        return self.buffer_transcribe


def get_not_done_fact_check_result(user_text):
    """Generate mock fact-checking result"""
    return {
        "code": 0,
        "data": {
            "ideas_de_fuerza": {
                "data": [
                    {
                        "claim": f"Claim extracted from: {user_text[:50]}...",
                        "incoherencia_detectada": "En produccioón ...",
                        "resultado_verificado": "En produccioón ...",
                        "respuesta_sugerida": "En produccioón ...",
                        "referencias": []
                    }
                ],
                "score_line": []
            },
            "discurso": {
                "text": user_text,
                "color": [0] * len(user_text),
                "claim_mask": [0] * len(user_text)
            },
            "Titulares": {
                "Afirmaciones Criticas": [],
                "Afirmaciones Ambiguas": []
            },
            'user_text': user_text
        },
        "error": "",
        "is_done": False,
        'user_text': user_text
    }


def transcribe_audio_file_with_scribe(file_bytes, filename="audio", content_type="application/octet-stream", language="es"):
    """Transcribe a complete audio file using ElevenLabs Scribe API.

    Returns a dict like { 'text': str, 'full_response': {...} } or None on failure.
    """
    try:
        files = {
            "file": (filename, file_bytes, content_type)
        }
        data = {
            "model_id": "scribe_v1",
            "language": language,
            "timestamps_granularity": "word"
        }
        headers = {
            "Accept": "application/json",
            "xi-api-key": config.ELEVENLABS_API_KEY
        }
        resp = requests.post(
            "https://api.elevenlabs.io/v1/speech-to-text",
            headers=headers,
            files=files,
            data=data,
            timeout=120
        )
        logger.info(f"📡 FILE: Service response: {resp.status_code}")
        if resp.status_code != 200:
            logger.error(f"❌ FILE: Service error {resp.status_code}: {resp.text[:200]}")
            return None
        result = resp.json()
        text = result.get('text') or result.get('transcript')
        if not text:
            logger.error("❌ FILE: Service returned no text")
            return None
        return { 'text': text, 'full_response': result }
    except Exception as e:
        logger.error(f"❌ FILE: Service call failed: {e}")
        return None



def process_transcription(transcriber: ElevenLabsTranscriber, job_id):
    """Monitor live transcription and incrementally fact-check completed sentences.

    Runs in a background thread while the transcriber is recording, splitting
    the buffer into sentences and dispatching parallel fact-check tasks.
    """
    logger.info(f"📝 Processing fact-check for job {job_id}")

    # Sentence-level processing state
    sentence_cache = {}  # sentence_hash -> fact_check_result
    processed_sentences = set()
    current_sentences = []
    active_tasks = {}  # sentence_hash -> thread
    sentences, final_sentences = [], []
    final_sentences_results = {}

    curr_user_text = getattr(transcriber, "buffer_transcribe", "")
    session_id = jobs.get(job_id, {}).get('session_id')

    def correct_text_with_llm(text):
        """Use LLM to correct text and punctuation"""
        if not text.strip() or not FACTCHECK_AVAILABLE:
            return text

        try:
            chatbot = CHATBOT_CLASS()
            prompt = f"""Corrige la puntuación y errores menores en este texto transcrito, manteniendo el significado original:

Texto: "{text}"

Devuelve solo el texto corregido, sin explicaciones."""

            # Use the correct LLM interface
            response = chatbot.llm.invoke(prompt)
            corrected = getattr(response, "content", str(response)).strip()
            logger.info(f"📝 Text corrected: '{text[:50]}...' -> '{corrected[:50]}...'")
            return corrected
        except Exception as e:
            logger.warning(f"⚠️ Text correction failed: {e}")
            return text

    def split_into_sentences(text):
        """Split text into sentences"""
        import re
        # Simple sentence splitting
        sentences = re.split(r'[.!?]+\s*', text.strip())
        return [s.strip() for s in sentences if s.strip()]

    def get_sentence_hash(sentence):
        """Get hash for sentence caching"""
        import hashlib
        return hashlib.md5(sentence.encode()).hexdigest()[:12]

    def process_sentence_async(sentence, sentence_idx):
        """Process single sentence fact-checking"""
        try:
            if FACTCHECK_AVAILABLE:
                chatbot = CHATBOT_CLASS()
                result = classify_paragraph(
                    chatbot=chatbot,
                    user_text=sentence,
                    top_k=8,  # Reduced for single sentences
                    max_context_chars=8000,
                    debug=False
                )
                logger.info(f"✅ Sentence {sentence_idx} fact-checked: '{sentence[:30]}...'")
                return result
            else:
                return get_mock_fact_check_result(sentence)
        except Exception as e:
            import traceback
            tb = traceback.format_exc()
            logger.error(f"❌ Sentence {sentence_idx} fact-check failed: {e}")
            logger.error(f"Full traceback:\n{tb}")
            print(f"❌ SENTENCE {sentence_idx} TRACEBACK:")
            print(tb)
            return get_mock_fact_check_result(sentence)

    def update_partial_results(sentences_results, sentences, user_text: str):
        """Update job with partial results available"""
        # Generate final result
        try:
            # Use combined approach
            final_text = ' '.join(sentences) if len(sentences) else user_text
            combined_ideas = []
            combined_sources = {}
            combined_score_line = []
            combined_afirmaciones_criticas = []
            combined_afirmaciones_ambiguas = []
            discurso_combined_text, discurso_combined_color, discurso_combined_claim_mask = "", [], []

            for idx, sentence in enumerate(sentences):
                if not sentence in final_sentences_results:
                    # logger.info(f"⏳ CRITICAL ISSUE HERE!")
                    result = get_not_done_fact_check_result(sentence)
                else:
                    result = sentences_results[sentence]

                # for sentence, result in final_sentences_results.items():
                if result and result.get('data'):
                    data = result['data']

                    if 'ideas_de_fuerza' in data and 'data' in data['ideas_de_fuerza']:
                        combined_ideas.extend(data['ideas_de_fuerza']['data'])
                        if 'score_line' in data['ideas_de_fuerza']:
                            combined_score_line.extend(data['ideas_de_fuerza']['score_line'])

                    if 'fuentes' in data:
                        for source in data['fuentes']:
                            source_id = source.get('idx', source.get('id', len(combined_sources)))
                            combined_sources[str(source_id)] = source

                    if 'Titulares' in data:
                        if 'Afirmaciones Criticas' in data['Titulares']:
                            combined_afirmaciones_criticas.extend(data['Titulares']['Afirmaciones Criticas'])
                        if 'Afirmaciones Ambiguas' in data['Titulares']:
                            combined_afirmaciones_ambiguas.extend(data['Titulares']['Afirmaciones Ambiguas'])
                    if 'discurso' in data:
                        discurso_combined_text += data['discurso']['text']
                        discurso_combined_color.extend(data['discurso']['color'])
                        discurso_combined_claim_mask.extend(data['discurso']['claim_mask'])

            final_result = {
                "code": 0,
                "data": {
                    "ideas_de_fuerza": {
                        "data": combined_ideas,
                        "score_line": combined_score_line
                    },
                    "fuentes": list(combined_sources.values()),
                    "discurso": {
                        "text": discurso_combined_text,
                        "color": discurso_combined_color,
                        "claim_mask": discurso_combined_claim_mask
                    },
                    "Titulares": {
                        "Afirmaciones Criticas": combined_afirmaciones_criticas,
                        "Afirmaciones Ambiguas": combined_afirmaciones_ambiguas
                    }
                },
                "error": "",
                "is_done": True,
                'user_text': final_text
            }

            # Update final job status
            jobs[job_id]['status'] = 'partial'
            jobs[job_id]['completed_at'] = datetime.now().isoformat()
            jobs[job_id]['user_text'] = final_text
            jobs[job_id]['result'] = final_result

            # Send final result to client
            if session_id:
                socketio.emit('fact_check_complete', {
                    'job_id': job_id,
                    'result': final_result,
                    'message': 'Complete fact-check analysis finished'
                }, room=session_id)

            logger.info(f"✅ Final fact-check completed for job {job_id}: {len(sentences)} sentences")

        except Exception as e:
            logger.error(f"❌ Error in final processing: {e}")

            # Fallback error result
            jobs[job_id]['status'] = 'failed'
            jobs[job_id]['completed_at'] = datetime.now().isoformat()
            jobs[job_id]['result'] = {
                "code": 1,
                "data": {"interim_answer": "", "final_answer": "", "claims": [], "fuentes": []},
                "error": f"Error en procesamiento final: {str(e)}",
                "is_done": True
            }



    # Main processing loop
    curr_user_text = ""
    start_mark = time.time()
    while transcriber.is_recording and time.time() - start_mark < 80:
        latest = getattr(transcriber, "buffer_transcribe", "")

        if latest != curr_user_text and latest.strip():
            curr_user_text = latest

            # 1. Correct text with LLM
            corrected_text = correct_text_with_llm(curr_user_text)

            # 2. Split into sentences
            sentences = split_into_sentences(corrected_text)

            if len(sentences) > len(current_sentences):
                # New complete sentences detected (exclude last incomplete one)
                complete_sentences = sentences[:-1] if transcriber.is_recording else sentences

                # 3. Process new sentences in parallel using threading
                for i, sentence in enumerate(complete_sentences):
                    sentence_hash = get_sentence_hash(sentence)

                    if sentence_hash not in sentence_cache and sentence_hash not in active_tasks:
                        logger.info(f"🔄 Processing new sentence {i + 1}: '{sentence[:50]}...'")

                        # Start async processing
                        def process_wrapper(sent=sentence, s_hash=sentence_hash, idx=i):
                            result = process_sentence_async(sent, idx)
                            sentence_cache[s_hash] = result
                            # Remove from active tasks when done
                            if s_hash in active_tasks:
                                del active_tasks[s_hash]
                            return result

                        task = threading.Thread(target=process_wrapper, daemon=True)
                        task.start()
                        active_tasks[sentence_hash] = task

                current_sentences = complete_sentences

        if len(sentences):
            # Wait for some tasks to complete before updating partial results
            completed_sentences = []
            for sentence in sentences:
                sentence_hash = get_sentence_hash(sentence)

                # Check if result is available
                if sentence_hash in sentence_cache:
                    completed_sentences.append(sentence)

            # 6. Update partial results with completed sentences
            if completed_sentences:
                sentences_results = {}
                for sentence in completed_sentences:
                    sentence_hash = get_sentence_hash(sentence)
                    if sentence_hash in sentence_cache:
                        sentences_results[sentence] = sentence_cache[sentence_hash]

                if sentences_results:
                    update_partial_results(sentences_results, sentences, curr_user_text)
                else:
                    # Update final job status
                    jobs[job_id]['status'] = 'partial'
                    jobs[job_id]['completed_at'] = datetime.now().isoformat()
                    jobs[job_id]['user_text'] = latest
                    jobs[job_id]['result'] = get_not_done_fact_check_result(latest)
            else:
                # Update final job status
                jobs[job_id]['status'] = 'partial'
                jobs[job_id]['completed_at'] = datetime.now().isoformat()
                jobs[job_id]['user_text'] = latest
                jobs[job_id]['result'] = get_not_done_fact_check_result(latest)

        else:
            # Update final job status
            jobs[job_id]['status'] = 'partial'
            jobs[job_id]['completed_at'] = datetime.now().isoformat()
            jobs[job_id]['user_text'] = latest
            jobs[job_id]['result'] = get_not_done_fact_check_result(latest)

        time.sleep(0.5)

    # Final processing when recording stops
    time.sleep(2)
    logger.info(f"🏁 Final processing for job {job_id}")


    # Process final text including last sentence
    curr_user_text = getattr(transcriber, "buffer_transcribe", "")
    final_text = correct_text_with_llm(curr_user_text)
    final_sentences = split_into_sentences(final_text)

    # Process any remaining sentences (especially the last incomplete one)
    for i, sentence in enumerate(final_sentences):
        sentence_hash = get_sentence_hash(sentence)
        if sentence_hash not in sentence_cache and sentence_hash not in active_tasks:
            logger.info(f"🔄 Processing new sentence {i + 1}: '{sentence[:50]}...'")

            # Start async processing
            def process_wrapper(sent=sentence, s_hash=sentence_hash, idx=i):
                result = process_sentence_async(sent, idx)
                sentence_cache[s_hash] = result
                # # Remove from active tasks when done
                # if s_hash in active_tasks:
                #     del active_tasks[s_hash]

                return result

            task = threading.Thread(target=process_wrapper, daemon=True)
            task.start()
            active_tasks[sentence_hash] = task

    # Wait for any remaining active tasks to complete
    for sentence_hash, task in list(active_tasks.items()):
        logger.info(f"⏳ Waiting for remaining task: {sentence_hash}")
        task.join(timeout=30)  # Longer timeout for final processing

    # Create final combined result
    final_sentences_results = {}
    logger.info("=" * 64)
    for idx, sentence in enumerate(final_sentences):
        logger.info(f"{idx}: {sentence}")
        sentence_hash = get_sentence_hash(sentence)
        if sentence_hash in sentence_cache:
            final_sentences_results[sentence] = sentence_cache[sentence_hash]
        else:
            logger.info(f"⏳ ISSUE HERE!")
    logger.info("="*64)


    # Generate final result
    try:
        # Use combined approach
        combined_ideas = []
        combined_sources = {}
        combined_score_line = []
        combined_afirmaciones_criticas = []
        combined_afirmaciones_ambiguas = []
        discurso_combined_text, discurso_combined_color, discurso_combined_claim_mask = "", [], []

        for idx, sentence in enumerate(final_sentences):
            if not sentence in final_sentences_results:
                logger.info(f"⏳ CRITICAL ISSUE HERE!")
                result = get_not_done_fact_check_result(sentence)
            else:
                result = final_sentences_results[sentence]

            # for sentence, result in final_sentences_results.items():
            if result and result.get('data'):
                data = result['data']

                if 'ideas_de_fuerza' in data and 'data' in data['ideas_de_fuerza']:
                    combined_ideas.extend(data['ideas_de_fuerza']['data'])
                    if 'score_line' in data['ideas_de_fuerza']:
                        combined_score_line.extend(data['ideas_de_fuerza']['score_line'])

                if 'fuentes' in data:
                    for source in data['fuentes']:
                        source_id = source.get('idx', source.get('id', len(combined_sources)))
                        combined_sources[str(source_id)] = source

                if 'Titulares' in data:
                    if 'Afirmaciones Criticas' in data['Titulares']:
                        combined_afirmaciones_criticas.extend(data['Titulares']['Afirmaciones Criticas'])
                    if 'Afirmaciones Ambiguas' in data['Titulares']:
                        combined_afirmaciones_ambiguas.extend(data['Titulares']['Afirmaciones Ambiguas'])
                if 'discurso' in data:
                    discurso_combined_text += data['discurso']['text']
                    discurso_combined_color.extend(data['discurso']['color'])
                    discurso_combined_claim_mask.extend(data['discurso']['claim_mask'])

        # --- Limpio ideas_de_fuerza donde elementos de data tienen 'incoherencia_detectada': 'No se detecta incoherencia'
        combined_ideas_cleaned = [e for e in combined_ideas if e['incoherencia_detectada'] != 'No se detecta incoherencia']
        combined_score_line_cleaned = [c for c, e in zip(combined_ideas, combined_ideas) if e['incoherencia_detectada'] != 'No se detecta incoherencia']

        final_result = {
            "code": 0,
            "data": {
                "ideas_de_fuerza": {
                    "data": combined_ideas_cleaned,
                    "score_line": combined_score_line_cleaned
                },
                "fuentes": list(combined_sources.values()),
                "discurso": {
                    "text": discurso_combined_text,
                    "color": discurso_combined_color,
                    "claim_mask": discurso_combined_claim_mask
                },
                "Titulares": {
                    "Afirmaciones Criticas": combined_afirmaciones_criticas,
                    "Afirmaciones Ambiguas": combined_afirmaciones_ambiguas
                }
            },
            "error": "",
            "is_done": True,
            'user_text': final_text
        }

        # Update final job status
        jobs[job_id]['status'] = 'completed'
        jobs[job_id]['completed_at'] = datetime.now().isoformat()
        jobs[job_id]['user_text'] = final_text
        jobs[job_id]['result'] = final_result

        # Send final result to client
        if session_id:
            socketio.emit('fact_check_complete', {
                'job_id': job_id,
                'result': final_result,
                'message': 'Complete fact-check analysis finished'
            }, room=session_id)

        logger.info(f"✅ Final fact-check completed for job {job_id}: {len(final_sentences)} sentences")

    except Exception as e:
        logger.error(f"❌ Error in final processing: {e}")

        # Fallback error result
        jobs[job_id]['status'] = 'failed'
        jobs[job_id]['completed_at'] = datetime.now().isoformat()
        jobs[job_id]['result'] = {
            "code": 1,
            "data": {"interim_answer": "", "final_answer": "", "claims": [], "fuentes": []},
            "error": f"Error en procesamiento final: {str(e)}",
            "is_done": True
        }

        if session_id:
            socketio.emit('fact_check_error', {
                'job_id': job_id,
                'error': str(e),
                'message': 'Error during final fact-check analysis'
            }, room=session_id)


# ============== POLITICAL COMMUNICATION ENDPOINTS ==============
def build_political_comm_health_payload() -> dict:
    """Build a health-check JSON payload for the political communication subsystem."""
    check_and_reload_if_needed()
    return {
        "status": "ok",
        "service": "Political Communication API",
        "timestamp": datetime.utcnow().isoformat(timespec="seconds") + "Z",
        "last_data_reload": last_reload.isoformat(timespec="seconds") + "Z" if last_reload else None,
        "endpoints": {
            "radar": "/api/radar",
            "media_affinity": "/api/media-affinity",
            "media_coverage": "/api/media-coverage",
            "alerts": "/api/alerts",
            "health": "/health",
        }
    }


@app.get("/api/health")
def api_health():
    """Return political communication API health status."""
    return jsonify(build_political_comm_health_payload())


@app.get("/api/radar")
def radar_endpoint():
    """Return political radar data or PNG visualization."""
    try:
        check_and_reload_if_needed()

        topic = request.args.get("topic", "vivienda").lower()
        window = request.args.get("window", "7d").lower()
        parties = request.args.get("parties", "pp,vox,psoe,programa").lower().split(",")
        fmt = request.args.get("format", "json").lower()

        if fmt not in ['png', 'json']:
            raise ValueError("fmt parameter should be in ('json', 'png')")
        if window not in ['7d', '30d', '90d']:
            raise ValueError("window parameter should be in ('7d', '30d', '90d')")
        if topic not in TOPIC_LABELS:
            raise ValueError("topic parameter should be in ('vivienda', 'educacion', 'sanidad', 'economia', 'seguridad', 'transporte')")

        data = get_radar_scores(topic=topic, window=window, parties=parties)

        if fmt == "png":
            filename = f"{IMAGES_DIR}/radar_{topic}_{window}.png"
            if os.path.exists(filename):
                return send_file(filename, mimetype="image/png", as_attachment=False)
            return jsonify({"error": "Image not found"}), 404

        return jsonify({
            "meta": {
                "topic": topic,
                "window": window,
                "generated_at": datetime.utcnow().isoformat(timespec="seconds") + "Z",
                "units": "index_0_100",
            },
            "axes": data["axes"],
            "series": [{"name": key, "values": values} for key, values in data["series"].items()],
            "range": {"min": 0, "max": 100},
        })
    except Exception as exc:
        return jsonify({
            "error": str(exc),
            "type": type(exc).__name__
        }), 500


@app.get("/api/media-affinity")
def media_affinity_endpoint():
    """Return media affinity scatter data or PNG visualization."""
    try:
        check_and_reload_if_needed()

        window = request.args.get("window", "30d").lower()
        fmt = request.args.get("format", "json").lower()

        if fmt not in ['png', 'json']:
            raise ValueError("fmt parameter should be in ('json', 'png')")
        if window not in ['7d', '30d', '90d']:
            raise ValueError("window parameter should be in ('7d', '30d', '90d')")

        points = get_affinity_scores(window)

        if fmt == "png":
            filename = f"{IMAGES_DIR}/media_affinity_{window}.png"
            if os.path.exists(filename):
                return send_file(filename, mimetype="image/png", as_attachment=False)
            return jsonify({"error": "Image not found"}), 404

        return jsonify({
            "meta": {
                "window": window,
                "generated_at": datetime.utcnow().isoformat(timespec="seconds") + "Z",
                "units": {"affinity": "index_0_100", "reach": "index_0_100"},
            },
            "sources": MEDIA_SOURCES,
            "points": points,
            "range": {"affinity": [0, 100], "reach": [0, 100]},
        })
    except Exception as exc:
        return jsonify({
            "error": str(exc),
            "type": type(exc).__name__
        }), 500


@app.get("/api/media-affinity_socialnets")
def media_affinity_social_nets_endpoint():
    """Return media affinity scatter data or PNG visualization."""
    try:
        check_and_reload_if_needed()

        window = request.args.get("window", "30d").lower()
        fmt = request.args.get("format", "json").lower()

        if fmt not in ['png', 'json']:
            raise ValueError("fmt parameter should be in ('json', 'png')")
        if window not in ['7d', '30d', '90d']:
            raise ValueError("window parameter should be in ('7d', '30d', '90d')")

        points = get_affinity_scores(window)

        if fmt == "png":
            filename = f"{IMAGES_DIR}/media_affinity_{window}.png"
            if os.path.exists(filename):
                return send_file(filename, mimetype="image/png", as_attachment=False)
            return jsonify({"error": "Image not found"}), 404

        return jsonify({
            "meta": {
                "window": window,
                "generated_at": datetime.utcnow().isoformat(timespec="seconds") + "Z",
                "units": {"affinity": "index_0_100", "reach": "index_0_100"},
            },
            "sources": MEDIA_SOURCES,
            "points": points,
            "range": {"affinity": [0, 100], "reach": [0, 100]},
        })
    except Exception as exc:
        return jsonify({
            "error": str(exc),
            "type": type(exc).__name__
        }), 500


@app.get("/api/media-coverage")
def media_coverage_endpoint():
    """Return media coverage stacked bar data or PNG visualization."""
    try:
        check_and_reload_if_needed()

        topics = request.args.get("topics", ",".join(TOPIC_LABELS.keys())).lower().split(",")
        topics = [topic for topic in topics if topic in TOPIC_LABELS] or list(TOPIC_LABELS.keys())
        window = request.args.get("window", "30d").lower()
        sort = request.args.get("sort", "name").lower()
        fmt = request.args.get("format", "json").lower()

        if fmt not in ['png', 'json']:
            raise ValueError("fmt parameter should be in ('json', 'png')")
        if window not in ['7d', '30d', '90d']:
            raise ValueError("window parameter should be in ('7d', '30d', '90d')")

        rows = get_coverage_data(topics, window)

        if sort == "total":
            rows.sort(key=lambda row: row["total"], reverse=True)
        elif sort in ("positive", "negative"):
            rows.sort(key=lambda row: row["counts"][sort], reverse=True)
        else:
            rows.sort(key=lambda row: row["label"])

        if fmt == "png":
            if set(topics) == set(TOPIC_LABELS.keys()):
                filename = f"{IMAGES_DIR}/media_coverage_{window}_{sort}.png"
                if os.path.exists(filename):
                    return send_file(filename, mimetype="image/png", as_attachment=False)

            return send_file(
                io.BytesIO(stacked_bars_png_coverage(rows)),
                mimetype="image/png",
                as_attachment=False,
                download_name="media_coverage.png",
            )

        return jsonify({
            "meta": {
                "window": window,
                "generated_at": datetime.utcnow().isoformat(timespec="seconds") + "Z",
                "order": ["positive", "neutral", "negative"],
            },
            "topics": rows,
        })
    except Exception as exc:
        return jsonify({
            "error": str(exc),
            "type": type(exc).__name__
        }), 500


@app.get("/api/alerts")
def alerts_endpoint():
    """Return curated alert cards data or PNG visualization."""
    try:
        check_and_reload_if_needed()

        severities_q = request.args.get("severities", "").lower().split(",")
        severities = [severity for severity in severities_q if severity in SEVERITY]
        topics_q = request.args.get("topics", "").lower().split(",")
        topics = [topic for topic in topics_q if topic in TOPIC_LABELS]
        limit = int(request.args.get("limit", "3"))
        fmt = request.args.get("format", "json").lower()

        if fmt not in ['png', 'json']:
            raise ValueError("fmt parameter should be in ('json', 'png')")

        rows = []
        for alert in ALERTS_CONST:
            if severities and alert["severity"] not in severities:
                continue
            if topics and alert["topic"] not in topics:
                continue
            rows.append(alert)
            if len(rows) >= limit:
                break

        if fmt == "png":
            if not severities and not topics and limit == 3:
                filename = f"{IMAGES_DIR}/alerts_default.png"
                if os.path.exists(filename):
                    return send_file(filename, mimetype="image/png", as_attachment=False)

            return send_file(
                io.BytesIO(cards_png(rows)),
                mimetype="image/png",
                as_attachment=False,
                download_name="alerts.png",
            )

        return jsonify({
            "meta": {
                "generated_at": datetime.utcnow().isoformat(timespec="seconds") + "Z",
                "total": len(rows),
            },
            "alerts": [
                {
                    "id": alert["id"],
                    "severity": {"key": alert["severity"], **SEVERITY[alert["severity"]]},
                    "topic": {
                        "key": alert["topic"],
                        "label": TOPIC_LABELS.get(alert["topic"], alert["topic"].capitalize()),
                    },
                    "title": alert["title"],
                }
                for alert in rows
            ],
        })
    except Exception as exc:
        return jsonify({
            "error": str(exc),
            "type": type(exc).__name__
        }), 500


@app.errorhandler(404)
def handle_not_found(error):
    """Return the health payload as a fallback for unmatched routes."""
    logger.info(f"404 Not Found: {request.path}")
    return jsonify(build_political_comm_health_payload())


# ============== AUDIO TRANSCRIPTION ENDPOINTS ==============
# Add error handler for SocketIO
@socketio.on_error_default
def default_error_handler(e):
    """Log unhandled SocketIO errors."""
    logger.error(f"SocketIO Error: {e}")
    print(f"SocketIO Error: {e}")


# Add connection error handler
@app.errorhandler(500)
def handle_500(e):
    """Return a JSON error response for internal server errors."""
    logger.error(f"Internal Server Error: {e}")
    return jsonify({'error': 'Internal server error'}), 500


@socketio.on('connect')
def handle_connect():
    """Handle new WebSocket connection and initialize chunk counter."""
    logger.info(f"🔗 WebSocket CONNECTED: {request.sid}")
    chunk_counters[request.sid] = 0
    emit('status', {'message': 'Connected to server with Pseudo Real-Time'})


@socketio.on('disconnect')
def handle_disconnect():
    """Clean up transcriber and session state on WebSocket disconnect."""
    logger.info(f"❌ WebSocket DISCONNECTED: {request.sid}")
    # Clean up transcriber
    if request.sid in active_transcribers:
        transcriber = active_transcribers[request.sid]
        transcriber.is_recording = False  # Stop recording
        if transcriber.pseudo_realtime_timer:
            transcriber.pseudo_realtime_timer.cancel()
        del active_transcribers[request.sid]
        logger.info(f"🗑️ Cleaned up transcriber for {request.sid}")
    if request.sid in chunk_counters:
        del chunk_counters[request.sid]
    if request.sid in session_job_ids:
        del session_job_ids[request.sid]


@socketio.on('start_recording')
def handle_start_recording():
    """Initialize transcriber for new recording session"""
    session_id = request.sid
    logger.info(f"🎤 START RECORDING request from {session_id}")

    # Generate job_id immediately when recording starts
    job_id = str(uuid.uuid4())
    session_job_ids[session_id] = job_id

    # Create job entry
    jobs[job_id] = {
        'status': 'recording',
        'user_text': '',
        'session_id': session_id,
        'created_at': datetime.now().isoformat(),
        'completed_at': None,
        'result': None
    }

    # Create transcription placeholder
    scribe[job_id] = {
        'status': 'recording',
        'user_text': '',  # This will contain the buffer_transcribe
        'session_id': session_id,
        'created_at': datetime.now().isoformat(),
        'completed_at': None,
        'result': None,
        'buffer_transcribe': '',  # Store pseudo real-time accumulated transcription
        'final_transcription': '',  # Store final complete transcription
        'pseudo_transcriptions': []  # Store all partial transcriptions
    }

    # Stop existing transcriber if any
    if session_id in active_transcribers:
        old_transcriber = active_transcribers[session_id]
        old_transcriber.is_recording = False
        if old_transcriber.pseudo_realtime_timer:
            old_transcriber.pseudo_realtime_timer.cancel()
        del active_transcribers[session_id]
        logger.info(f"🔥 Stopped existing transcriber for {session_id}")

    # Reset chunk counter
    chunk_counters[session_id] = 0

    # Create new ElevenLabs transcriber
    logger.info(f"🔧 Creating new transcriber for {session_id} (job_id: {job_id})")
    transcriber = ElevenLabsTranscriber(session_id, config.ELEVENLABS_API_KEY)
    transcriber.start_recording()

    active_transcribers[session_id] = transcriber

    threading.Thread(
        target=process_transcription,
        args=(transcriber, job_id, ),
        daemon=True
    ).start()


    logger.info(f"✅ Transcriber initialized and started for {session_id}")
    emit('recording_started', {
        'status': 'Recording started',
        'job_id': job_id
    })


@socketio.on('stop_recording')
def handle_stop_recording():
    """Stop recording and finalize transcription"""
    session_id = request.sid
    logger.info(f"ℹ️ STOP RECORDING request from {session_id}")

    # Get existing job_id for this session
    job_id = session_job_ids.get(session_id)

    if session_id in active_transcribers:
        transcriber = active_transcribers[session_id]

        # Stop the transcriber and get results
        logger.info(f"🎯 Stopping transcriber for {session_id}")

        def process_final_transcription():
            try:
                _ = transcriber.stop_recording_and_transcribe()

                # Get the accumulated buffer (pseudo real-time result)
                buffer_text = transcriber.get_buffer_transcribe()

                # Determine which text to use - prioritize buffer if available
                user_text = buffer_text.strip() if buffer_text.strip() else ""

                # If no text from either source, use mock for testing
                if not user_text:
                    user_text = "Mock transcribed text for testing purposes"
                    logger.info(f"🔧 Using mock transcription for testing")

                if job_id:
                    logger.info(f"🎯 Transcription completed: '{user_text}' (job_id: {job_id})")

                    # Update job with transcribed text
                    jobs[job_id]['user_text'] = user_text
                    jobs[job_id]['status'] = 'transcribed'

                    socketio.emit('transcription_complete', {
                        'text': user_text,
                        'job_id': job_id,
                        'buffer_transcribe': buffer_text,
                        'final_transcription': user_text
                    }, room=session_id)

                    # Automatically start fact-checking with the transcribed text
                    logger.info(f"🔍 Starting automatic fact-check for: '{user_text}'")
                    start_automatic_fact_check(user_text, session_id, job_id)
                else:
                    # Fallback if no job_id (shouldn't happen)
                    new_job_id = str(uuid.uuid4())
                    logger.warning(f"No job_id found for session {session_id}, creating new one: {new_job_id}")
                    socketio.emit('transcription_complete', {
                        'text': user_text,
                        'job_id': new_job_id,
                        'buffer_transcribe': buffer_text,
                        'final_transcription': user_text
                    }, room=session_id)
                    start_automatic_fact_check(user_text, session_id, new_job_id)

            except Exception as e:
                logger.error(f"❌ Error in final transcription processing: {e}")
                if job_id:
                    socketio.emit('error', {
                        'message': f'Error processing final transcription: {str(e)}',
                        'job_id': job_id
                    }, room=session_id)

        # Process final transcription in background thread
        threading.Thread(target=process_final_transcription, daemon=True).start()

        # Clean up transcriber
        del active_transcribers[session_id]

        total_chunks = chunk_counters.get(session_id, 0)
        logger.info(f"✅ Session {session_id} ended. Total chunks processed: {total_chunks}")
    else:
        logger.warning(f"⚠️ No active transcriber found for {session_id}")

        # If no transcriber but we have a job_id, still trigger fact-checking with mock text
        if job_id:
            mock_text = "Mock transcribed text for testing purposes"
            jobs[job_id]['user_text'] = mock_text
            jobs[job_id]['status'] = 'transcribed'

            scribe[job_id].update({
                'status': 'transcribed',
                'user_text': mock_text,
                'completed_at': datetime.now().isoformat()
            })

            emit('transcription_complete', {
                'text': mock_text,
                'job_id': job_id
            })

            logger.info(f"🔍 Starting fact-check with mock text for job {job_id}")
            start_automatic_fact_check(mock_text, session_id, job_id)

    emit('recording_stopped', {
        'status': 'Recording stopped',
        'job_id': job_id
    })


@socketio.on('audio_chunk')
def handle_audio_chunk(data):
    """Process incoming audio chunk"""
    session_id = request.sid
    chunk_counters[session_id] = chunk_counters.get(session_id, 0) + 1
    chunk_num = chunk_counters[session_id]

    if session_id not in active_transcribers:
        logger.warning(f"No transcriber for {session_id}, auto-starting...")
        handle_start_recording()
        if session_id not in active_transcribers:
            emit('error', {'message': 'Failed to initialize transcriber'})
            return

    try:
        audio_bytes = data.get('audio_data')
        sample_rate = data.get('sample_rate')

        if audio_bytes:
            if isinstance(audio_bytes, list):
                audio_bytes = bytes(audio_bytes)

            # Log every 120 chunks to show activity (approximately every 4 seconds)
            if chunk_num % 120 == 0:
                logger.info(f"📦 Collecting chunk #{chunk_num} ({len(audio_bytes)} bytes @ {sample_rate}Hz)")

            transcriber = active_transcribers[session_id]
            transcriber.add_audio_chunk(audio_bytes, sample_rate)
        else:
            logger.error(f"No audio data in chunk #{chunk_num}")
            emit('error', {'message': 'No audio data received'})

    except Exception as e:
        logger.error(f"Error processing chunk #{chunk_num}: {e}")
        emit('error', {'message': f'Error processing audio chunk: {str(e)}'})


def start_automatic_fact_check(transcribed_text, session_id, job_id):
    """Automatically start fact-checking for transcribed text"""

    def process_fact_check_async(text, session_id, job_id):
        try:
            # Update job with processing status
            jobs[job_id] = {
                'status': 'processing',
                'user_text': text,
                'session_id': session_id,
                'created_at': datetime.now().isoformat(),
                'completed_at': None,
                'result': None
            }

            # Notify client that fact-checking started
            socketio.emit('fact_check_started', {
                'job_id': job_id,
                'text': text,
                'message': 'Starting fact-check analysis...'
            }, room=session_id)

            logger.info(f"🔍 Processing fact-check for job {job_id}")

            if FACTCHECK_AVAILABLE and text.strip():
                # Use real fact-checking system
                chatbot = CHATBOT_CLASS()
                result = classify_paragraph(
                    chatbot=chatbot,
                    user_text=text,
                    top_k=12,
                    max_context_chars=12000,
                    debug=False
                )
                logger.info(f"✅ Fact-check completed for job {job_id}")
            else:
                # Fallback to mock
                result = get_mock_fact_check_result(text)
                logger.info(f"🔍 Mock fact-check completed for job {job_id}")

            # ADD THIS: Check if result is None and provide fallback
            if result is None:
                logger.warning(f"⚠️ Fact-check returned None for job {job_id}, using mock")
                result = get_mock_fact_check_result(text)

            # Update job status
            jobs[job_id]['status'] = 'completed'
            jobs[job_id]['completed_at'] = datetime.now().isoformat()
            jobs[job_id]['result'] = result

            # Send result to client
            socketio.emit('fact_check_complete', {
                'job_id': job_id,
                'result': result,
                'message': 'Fact-check analysis completed'
            }, room=session_id)

            logger.info(f"📤 Fact-check result sent to client {session_id}")

        except Exception as e:
            logger.error(f"❌ Error in fact-check processing: {e}")

            # Handle errors
            jobs[job_id]['status'] = 'failed'
            jobs[job_id]['completed_at'] = datetime.now().isoformat()
            jobs[job_id]['result'] = {
                "code": 1,
                "data": {"interim_answer": "", "final_answer": "", "claims": [], "fuentes": []},
                "error": f"Error en fact-checking: {str(e)}",
                "is_done": True
            }

            socketio.emit('fact_check_error', {
                'job_id': job_id,
                'error': str(e),
                'message': 'Error during fact-check analysis'
            }, room=session_id)

    # Start background processing
    thread = threading.Thread(
        target=process_fact_check_async,
        args=(transcribed_text, session_id, job_id)
    )
    thread.daemon = True
    thread.start()


# ============== EXISTING FACT CHECKER ENDPOINTS ==============

@app.route('/api/v1/machiavelli/fact_checker', methods=['POST'])
def submit_fact_check():
    """Submit text for fact-checking and get a job_id"""
    data = request.get_json()
    user_text = data.get('user_text', '')
    top_k = data.get('top_k', 12)
    max_context_chars = data.get('max_context_chars', 12000)

    job_id = str(uuid.uuid4())
    jobs[job_id] = {
        'status': 'processing',
        'user_text': user_text,
        'created_at': datetime.now().isoformat(),
        'completed_at': None,
        'result': None
    }

    def process_fact_check(job_id, user_text, top_k, max_context_chars):
        """Background task for fact-checking"""
        try:
            if FACTCHECK_AVAILABLE and user_text.strip():
                # Use real fact-checking system
                chatbot = CHATBOT_CLASS()
                result = classify_paragraph(
                    chatbot=chatbot,
                    user_text=user_text,
                    top_k=top_k,
                    max_context_chars=max_context_chars,
                    debug=False
                )
            else:
                # Fallback to mock
                result = get_mock_fact_check_result(user_text)

            # Update job status
            jobs[job_id]['status'] = 'completed'
            jobs[job_id]['completed_at'] = datetime.now().isoformat()
            jobs[job_id]['result'] = result

        except Exception as e:
            # Handle errors
            jobs[job_id]['status'] = 'failed'
            jobs[job_id]['completed_at'] = datetime.now().isoformat()
            jobs[job_id]['result'] = {
                "code": 1,
                "data": {"interim_answer": "", "final_answer": "", "claims": [], "fuentes": []},
                "error": f"Error en fact-checking: {str(e)}",
                "is_done": True
            }

    # Start background processing
    thread = threading.Thread(
        target=process_fact_check,
        args=(job_id, user_text, top_k, max_context_chars)
    )
    thread.daemon = True
    thread.start()

    return jsonify({
        'job_id': job_id,
        'status': 'submitted',
        'message': 'Fact-checking job submitted successfully'
    })


@app.route('/api/v1/machiavelli/fact_checker/<job_id>/status', methods=['GET'])
def get_fact_check_status(job_id):
    """Get fact-checking job status without results"""
    if job_id not in jobs:
        return jsonify({'error': 'Job not found'}), 404

    job = jobs[job_id]
    return jsonify({
        'job_id': job_id,
        'status': job['status'],
        'user_text': job['user_text'],
        'created_at': job['created_at'],
        'completed_at': job.get('completed_at'),
        'has_result': job['result'] is not None
    })


@app.route('/api/v1/machiavelli/fact_checker/<job_id>/result', methods=['GET'])
def get_fact_check_result_json(job_id):
    """Get fact-checking results as JSON"""
    if job_id not in jobs:
        return jsonify({'error': 'Job not found'}), 404

    job = jobs[job_id]
    if job['status'] == 'processing':
        return jsonify({
            'job_id': job_id,
            'status': 'processing',
            'message': 'Job still processing'
        }), 202

    if job['status'] == 'failed':
        return jsonify({
            'job_id': job_id,
            'status': 'failed',
            'error': job['result'].get('error', 'Unknown error')
        }), 500

    return jsonify({
        'job_id': job_id,
        'status': 'completed',
        'result': job['result'],
        'completed_at': job['completed_at']
    })


@app.route('/api/v1/machiavelli/fact_checker/<job_id>', methods=['GET'])
def get_fact_check_result(job_id):
    """Get fact-checking results via Server-Sent Events"""
    if job_id not in jobs:
        return jsonify({'error': 'Job not found'}), 404

    job = jobs[job_id]
    if job['status'] != 'completed':
        return jsonify({'error': 'Job still processing'}), 202

    def generate_sse_response():
        time.sleep(1)
        data_line = f"data: {json.dumps(job['result'])}\n\n"
        yield data_line

    return Response(
        generate_sse_response(),
        mimetype='text/event-stream',
        headers={
            'Cache-Control': 'no-cache',
            'Connection': 'keep-alive',
            'Access-Control-Allow-Origin': '*'
        }
    )

@app.route('/api/v1/machiavelli/fact_checker/jobs', methods=['GET'])
def list_fact_check_jobs():
    """List all fact-check jobs sorted by creation time (newest first)."""

    print('Maxi')

    try:

        print('Maxi')
        print(jobs)

        sorted_jobs = sorted(
            jobs.items(),
            key=lambda item: item[1].get('created_at', ''),
            reverse=True,
        )
        data = []
        for job_id, info in sorted_jobs:
            data.append(
                {
                    'job_id': job_id,
                    'status': info.get('status'),
                    'created_at': info.get('created_at'),
                    'completed_at': info.get('completed_at'),
                    'has_result': info.get('result') is not None,
                    'user_text_preview': (info.get('user_text') or '')[:140],
                }
            )
        return jsonify({'jobs': data, 'count': len(data)}), 200
    except Exception as exc:  # pragma: no cover - safety net
        logger.error(f"Failed to list jobs: {exc}")
        return jsonify({'error': 'Failed to list jobs'}), 500


# ============== NEW AUDIO FACT CHECKER (HTTP file upload) ==============

@app.route('/api/v1/machiavelli/audio_fact_checker', methods=['POST'])
def audio_fact_checker():
    """Accept an audio file, transcribe with internal service, and submit a fact-check job.

    Accepts multipart/form-data with keys 'user_audio', 'audio', or 'file'.
    Also accepts raw body bytes (binary) as a fallback.
    Returns JSON with a job_id and the transcribed text on success.
    """
    uploaded = None
    filename = "audio"
    mimetype = "application/octet-stream"

    # Try common keys
    for key in ('user_audio', 'audio', 'file'):
        if key in request.files:
            uploaded = request.files[key]
            filename = uploaded.filename or filename
            mimetype = uploaded.mimetype or mimetype
            break

    file_bytes = None
    if uploaded is not None:
        file_bytes = uploaded.read()
    elif request.data:
        file_bytes = request.data
        # Keep defaults for filename/mimetype
    else:
        return jsonify({'error': 'No audio data provided'}), 400

    # Transcribe using ElevenLabs
    transcribed = transcribe_audio_file_with_scribe(file_bytes, filename=filename, content_type=mimetype, language='es')
    if not transcribed or not transcribed.get('text'):
        return jsonify({'error': 'Transcription failed'}), 400

    user_text = transcribed['text'].strip()

    if not user_text:
        return jsonify({'error': 'Empty transcription'}), 400

    # Optional params for fact-check
    top_k = int(request.form.get('top_k', request.args.get('top_k', 12))) if isinstance(request.form.get('top_k', request.args.get('top_k', '12')), str) else 12
    max_context_chars = int(request.form.get('max_context_chars', request.args.get('max_context_chars', 12000))) if isinstance(request.form.get('max_context_chars', request.args.get('max_context_chars', '12000')), str) else 12000

    # Create job entry
    job_id = str(uuid.uuid4())
    jobs[job_id] = {
        'status': 'processing',
        'user_text': user_text,
        'created_at': datetime.now().isoformat(),
        'completed_at': None,
        'result': None
    }

    # Background processing
    def process_fact_check(job_id, user_text, top_k, max_context_chars):
        try:
            if FACTCHECK_AVAILABLE and user_text.strip():
                chatbot = CHATBOT_CLASS()
                result = classify_paragraph(
                    chatbot=chatbot,
                    user_text=user_text,
                    top_k=int(top_k),
                    max_context_chars=int(max_context_chars),
                    debug=False
                )
            else:
                result = get_mock_fact_check_result(user_text)

            jobs[job_id]['status'] = 'completed'
            jobs[job_id]['completed_at'] = datetime.now().isoformat()
            jobs[job_id]['result'] = result
        except Exception as e:
            jobs[job_id]['status'] = 'failed'
            jobs[job_id]['completed_at'] = datetime.now().isoformat()
            jobs[job_id]['result'] = {
                "code": 1,
                "data": {"interim_answer": "", "final_answer": "", "claims": [], "fuentes": []},
                "error": f"Error en fact-checking: {str(e)}",
                "is_done": True
            }

    thread = threading.Thread(
        target=process_fact_check,
        args=(job_id, user_text, top_k, max_context_chars)
    )
    thread.daemon = True
    thread.start()

    return jsonify({
        'job_id': job_id,
        'status': 'submitted',
        'transcription_text': user_text,
        'message': 'Audio transcribed and fact-checking job submitted'
    })


# ============== POLLING/SURVEY ENDPOINTS ==============

@app.route('/api/v1/validacion-intervencion', methods=['POST'])
def create_survey_intervention():
    """Create and validate new survey intervention"""
    data = request.get_json()

    encuesta_id = str(uuid.uuid4())
    encuestas[encuesta_id] = {
        'titulo': data.get('titulo', ''),
        'fecha': data.get('fecha', ''),
        'base_datos': data.get('base_datos', ''),
        'preguntas': data.get('preguntas', []),
        'numero_encuestados': 0,
        'created_at': datetime.now().isoformat()
    }

    return jsonify({
        'id': encuesta_id,
        'status': 'created',
        'numero_encuestados': 0,
        'fecha_creacion': datetime.now().isoformat(),
        'url_encuesta': f'https://encuesta.com/{encuesta_id}'
    })


@app.route('/api/v1/encuesta/<encuesta_id>/resultados', methods=['GET'])
def get_survey_results(encuesta_id):
    """Get survey results with optional filtering"""
    if encuesta_id not in encuestas:
        return jsonify({'error': 'Survey not found'}), 404

    # Mock filters (not actually used in mock)
    filtro_genero = request.args.get('filtro_genero', 'todos')
    filtro_edad = request.args.get('filtro_edad', 'cualquiera')

    return jsonify({
        'encuesta_id': encuesta_id,
        'titulo': 'Validacion de la intervencion (vivienda)',
        'numero_encuestados': 654,
        'fecha': '04/08/2025',
        'resultados': {
            'pregunta_1': {
                'texto': 'Valoracion de la intervencion',
                'respuestas': {
                    'sin_cambios': {'count': 412, 'porcentaje': 63.1},
                    'requiere_reformulacion': {'count': 106, 'porcentaje': 16.2},
                    'indiferente': {'count': 91, 'porcentaje': 13.9},
                    'prefiere_no_decirlo': {'count': 45, 'porcentaje': 6.8}
                }
            },
            'pregunta_2': {
                'texto': 'Refleja adecuadamente el argumentario del partido?',
                'nps_score': 41.3,
                'media_esperada': 35,
                'distribucion': {
                    'a_favor': {'porcentaje': 77.5, 'count': 507},
                    'parcialmente': {'porcentaje': 18.2, 'count': 119},
                    'no': {'porcentaje': 4.3, 'count': 28}
                }
            }
        }
    })


@app.route('/api/v1/encuesta/<encuesta_id>/insights', methods=['GET'])
def get_survey_insights(encuesta_id):
    """Get AI-generated insights about survey results"""
    if encuesta_id not in encuestas:
        return jsonify({'error': 'Survey not found'}), 404

    return jsonify({
        'insights_positivos': [
            {'texto': 'Se entiende, pero falta un golpe de efecto final.', 'score': 51, 'categoria': 'neutral'},
            {'texto': 'Correcta, aunque no deja huella.', 'score': 39, 'categoria': 'neutral'},
            {'texto': 'Funciona en contexto parlamentario, no tanto en medios.', 'score': 17, 'categoria': 'neutral'}
        ],
        'frases_debiles': [
            {'texto': 'No genera emocion ni orgullo de replica.', 'score': -31, 'categoria': 'negativo'},
            {'texto': 'Demasiado defensiva para ser creible.', 'score': -19, 'categoria': 'negativo'},
            {'texto': 'Parece hecha para no molestar, no para convencer.', 'score': -8, 'categoria': 'negativo'}
        ],
        'insights_gente_alta': [
            {'texto': 'Este discurso se puede defender en cualquier territorio.', 'count': 220,
             'sentimiento': 'positivo'},
            {'texto': 'Contundente sin sonar agresivo. Perfecto para el clima actual.', 'count': 196,
             'sentimiento': 'positivo'},
            {'texto': 'Es una intervencion que ordena el marco y genera autoridad.', 'count': 188,
             'sentimiento': 'positivo'}
        ]
    })


@app.route('/api/v1/chatbot/consulta', methods=['POST'])
def chatbot_query():
    """Chatbot queries about survey results"""
    data = request.get_json()
    encuesta_id = data.get('encuesta_id')
    pregunta = data.get('pregunta', '')

    if encuesta_id and encuesta_id not in encuestas:
        return jsonify({'error': 'Survey not found'}), 404

    return jsonify({
        'respuesta': 'Basandome en los resultados de la encuesta sobre la intervencion de vivienda, observo que hay una aceptacion mayoritaria (77.5% a favor) pero con un NPS de 41.3 que indica espacio de mejora. Las criticas se centran en la falta de impacto emocional...',
        'fuentes': ['pregunta_1_resultados', 'nps_score', 'insights_negativos'],
        'confianza': 0.85
    })


@app.route('/api/v1/encuesta/<encuesta_id>/segmentacion', methods=['GET'])
def get_survey_segmentation(encuesta_id):
    """Get demographic segmentation analysis"""
    if encuesta_id not in encuestas:
        return jsonify({'error': 'Survey not found'}), 404

    segmento = request.args.get('segmento', 'genero')

    segmentation_data = {
        'genero': {
            'masculino': {'count': 324, 'porcentaje_total': 49.5, 'nps_score': 43.1,
                          'pregunta_1_dominante': 'sin_cambios'},
            'femenino': {'count': 330, 'porcentaje_total': 50.5, 'nps_score': 39.7,
                         'pregunta_1_dominante': 'sin_cambios'}
        },
        'edad': {
            '18-25': {'count': 98, 'porcentaje_total': 15.0, 'nps_score': 38.2,
                      'pregunta_1_dominante': 'requiere_reformulacion'},
            '26-35': {'count': 156, 'porcentaje_total': 23.9, 'nps_score': 42.1, 'pregunta_1_dominante': 'sin_cambios'},
            '36-50': {'count': 245, 'porcentaje_total': 37.5, 'nps_score': 43.8, 'pregunta_1_dominante': 'sin_cambios'},
            '50+': {'count': 155, 'porcentaje_total': 23.6, 'nps_score': 40.5, 'pregunta_1_dominante': 'sin_cambios'}
        }
    }

    return jsonify({
        'segmento': segmento,
        'resultados': segmentation_data.get(segmento, segmentation_data['genero'])
    })


# ============== SURVEY SYNTHESIS ENDPOINTS ==============

@app.route('/api/v1/sintesis_encuesta', methods=['GET'])
def get_survey_synthesis():
    """Get survey synthesis results from analysis file"""
    try:
        # Read the analysis file
        with open('data/analisis_encuesta.json', 'r', encoding='utf-8') as f:
            synthesis_data = json.load(f)

        return jsonify(synthesis_data)

    except FileNotFoundError:
        return jsonify({'error': 'Analysis file not found'}), 404
    except json.JSONDecodeError:
        return jsonify({'error': 'Invalid JSON format in analysis file'}), 500
    except Exception as e:
        return jsonify({'error': f'Error reading analysis file: {str(e)}'}), 500


# ============== EXISTING RAG QUERY ENDPOINTS ==============

@app.route('/api/v1/machiavelli/query', methods=['POST'])
def machiavelli_query():
    """Query processing with streaming response using actual RAG system"""
    data = request.get_json()
    user_text = data.get('user_text', '')
    mode = data.get('mode', 'datos')
    top_k = data.get('top_k', 16)
    session_id = data.get('session_id')
    provider = data.get('provider')

    # Validate mode
    valid_modes = ['preguntas_frecuentes', 'datos', 'medios', 'debates', 'encuestas']
    if mode not in valid_modes:
        return jsonify({'error': f'Invalid mode. Must be one of: {valid_modes}'}), 400

    def generate_query_stream(mode):
        try:
            yield f"data: {json.dumps(get_search_response())}\n\n"
            time.sleep(0.5)

            yield f"data: {json.dumps(get_analyzing_response())}\n\n"
            time.sleep(0.5)

            if RAG_AVAILABLE and user_text.strip():
                try:
                    final_response, resolved_session = run_conversational_query(
                        user_text=user_text,
                        mode=mode,
                        top_k=top_k,
                        session_id=session_id,
                        provider=provider,
                    )
                    if resolved_session:
                        final_response.setdefault("data", {})["session_id"] = resolved_session
                except Exception as e:
                    final_response = {
                        "code": 1,
                        "data": {
                            "interim_answer": "",
                            "final_answer": f"Error procesando consulta: {str(e)}",
                            "fuentes": [],
                            "session_id": session_id
                        },
                        "error": f"RAG system error: {str(e)}",
                        "is_done": True
                    }
            else:
                if not user_text.strip():
                    final_response = {
                        "code": 1,
                        "data": {
                            "interim_answer": "",
                            "final_answer": "Error: Consulta vacía",
                            "fuentes": [],
                            "session_id": session_id
                        },
                        "error": "Empty query provided",
                        "is_done": True
                    }
                else:
                    final_response = get_final_query_response(user_text, mode)
                    final_response.setdefault("data", {})["session_id"] = session_id

            final_response = _normalize_final_answer_sources(final_response)
            yield f"data: {json.dumps(final_response)}\n\n"

        except (BrokenPipeError, ConnectionError):
            logger.info("Streaming aborted: client disconnected during query response")
        except GeneratorExit:
            logger.info("Streaming generator closed by client")
        except Exception as e:
            logger.exception("Unexpected error while streaming query response: %s", e)

    return Response(
        generate_query_stream(mode),
        mimetype='text/event-stream',
        headers={
            'Cache-Control': 'no-cache',
            'Connection': 'keep-alive',
            'Access-Control-Allow-Origin': '*'
        }
    )


# ============== HELPER FUNCTIONS ==============

def _extract_files_from_final_answer(text: str):
    """Split a final answer into clean text and a list of referenced filenames."""
    marker = "\n\nFuentes:"
    idx = text.find(marker)
    if idx == -1:
        return text, []

    tail = text[idx + len(marker):]
    filenames = re.findall(r'([\w\-.]+\.(?:xlsx|pdf))', tail, flags=re.IGNORECASE)
    if not filenames:
        return text, []

    cleaned = text[:idx].rstrip()
    print(cleaned, filenames)
    return cleaned, filenames


def _normalize_final_answer_sources(response: dict) -> dict:
    """Extract inline source filenames from final_answer and move them to fuentes."""
    if not isinstance(response, dict):
        return response

    data = response.get("data")
    if not isinstance(data, dict):
        return response

    final_answer = data.get("final_answer")
    if not isinstance(final_answer, str):
        return response

    cleaned_answer, filenames = _extract_files_from_final_answer(final_answer)
    if not filenames:
        return response

    data["final_answer"] = cleaned_answer
    fuentes = data.get("fuentes")
    if not isinstance(fuentes, list):
        fuentes = []
    for name in filenames:
        fuentes.append({"titulo": f"NEXUS: {name}", "url": ""})

    data["fuentes"] = fuentes
    return response


def get_mock_fact_check_result(user_text):
    """Generate mock fact-checking result"""
    return {
        "code": 0,
        "data": {
            "ideas_de_fuerza": {
                "data": [
                    {
                        "claim": f"Claim extracted from: {user_text[:50]}...",
                        "incoherencia_detectada": "Se detectó una posible inconsistencia en los datos mencionados",
                        "resultado_verificado": "Verificación completada con fuentes oficiales",
                        "respuesta_sugerida": "Los datos más actualizados indican información diferente",
                        "referencias": [
                            {
                                "fuente": "Fuente Oficial",
                                "id": 1,
                                "title": "Documento de verificación",
                                "url": "https://fuente-oficial.com/documento"
                            }
                        ]
                    }
                ],
                "score_line": []
            },
            "discurso": {
                "text": user_text,
                "color": [-0.2, -0.2, -0.2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                "claim_mask": [1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
            },
            "Titulares": {
                "Afirmaciones Criticas": [
                    {
                        "Titulo": "Afirmación detectada",
                        "respuesta_sugerida": "Respuesta sugerida basada en verificación"
                    }
                ],
                "Afirmaciones Ambiguas": []
            }
        },
        "error": "",
        "is_done": True
    }


def get_search_response():
    """Generate initial search response"""
    return {
        "code": 0,
        "data": {
            "interim_answer": "Buscando información en fuentes...",
            "final_answer": "",
            "fuentes": []
        },
        "error": "",
        "is_done": False
    }


def get_analyzing_response():
    """Generate analyzing response"""
    return {
        "code": 0,
        "data": {
            "interim_answer": "... analizando respuestas de las fuentes.",
            "final_answer": "",
            "fuentes": []
        },
        "error": "",
        "is_done": False
    }


def get_final_query_response(user_text, mode):
    """Generate final query response based on mode (fallback when RAG not available)"""
    mode_responses = {
        'preguntas_frecuentes': "Según las fuentes analizadas, esta es una consulta frecuente sobre {topic}...",
        'datos': "Los datos oficiales muestran que {topic} presenta las siguientes cifras...",
        'medios': "La cobertura mediática sobre {topic} indica diversas perspectivas...",
        'debates': "El debate público sobre {topic} incluye diferentes posturas...",
        'encuestas': "Las encuestas recientes sobre {topic} reflejan la opinión ciudadana..."
    }

    # Determine topic from user_text
    topic = "ayudas económicas para familias" if "ayuda" in user_text.lower() else "el tema consultado"

    base_response = mode_responses.get(mode, mode_responses['datos']).format(topic=topic)

    final_answer = f"""{base_response}

El País - 2024-08-10 https://elpais.com/ayudas-madrid

Las medidas incluyen bonos de 1.200 euros para familias con ingresos inferiores a 35.000 euros anuales..."""

    return {
        "code": 0,
        "data": {
            "interim_answer": "",
            "final_answer": final_answer,
            "fuentes": [
                {
                    "idx": "1",
                    "fuente": "El Pais",
                    "publish_date": "2024-08-10",
                    "ref": "https://elpais.com/ayudas-madrid"
                },
                {
                    "idx": "2",
                    "fuente": "BOCM - Boletín Oficial",
                    "publish_date": "2024-08-05",
                    "ref": "http://goverme.hiprax.com:5000/static/sources/bocm_2024_08_05.pdf"
                }
            ]
        },
        "error": "",
        "is_done": True
    }


# Add this to your api_mock.py file

# Add this to your api_mock.py file

# Add to the constants section at the top of the file
SOCIAL_POSTS_CONST = [
    {
        "id": "post_001",
        "author": {
            "name": "Iñaki López",
            "handle": "@InakiLopez_",
            "verified": True,
            "avatar": "https://example.com/avatar1.jpg"
        },
        "text": "Vivienda, ese tema donde los datos siempre se interpretan. ¿Fue convincente la respuesta del PP? Juzguen ustedes.",
        "platform": "twitter",
        "timestamp": "2025-10-15T10:30:00Z",
        "engagement": {
            "views": 12400,
            "likes": 340,
            "retweets": 87
        }
    },
    {
        "id": "post_002",
        "author": {
            "name": "Josep María Francés",
            "handle": "@jmfrances",
            "verified": True,
            "avatar": "https://example.com/avatar2.jpg"
        },
        "text": "El PSOE ha convertido el debate sobre vivienda en un sketch ideológico. Lo que proponen no se sostiene ni en Excel ni en la calle.",
        "platform": "twitter",
        "timestamp": "2025-10-15T11:15:00Z",
        "engagement": {
            "views": 8900,
            "likes": 210,
            "retweets": 45
        }
    },
    {
        "id": "post_003",
        "author": {
            "name": "Lucía Méndez Prada",
            "handle": "@LuciaMendezPM",
            "verified": True,
            "avatar": "https://example.com/avatar3.jpg"
        },
        "text": "El PSOE volvió a la carga con un discurso cargado de ideología. El PP defiende su modelo con datos, pero sin punch.",
        "platform": "twitter",
        "timestamp": "2025-10-15T12:45:00Z",
        "engagement": {
            "views": 15600,
            "likes": 520,
            "retweets": 120
        }
    },
    {
        "id": "post_004",
        "author": {
            "name": "Edu Medina",
            "handle": "@EduMedina",
            "verified": True,
            "avatar": "https://example.com/avatar4.jpg"
        },
        "text": "Ritmo, tono y narrativa claros. El portavoz del PP mantuvo el control con una necesidad de elevar la voz.",
        "platform": "twitter",
        "timestamp": "2025-10-15T13:20:00Z",
        "engagement": {
            "views": 6700,
            "likes": 180,
            "retweets": 32
        }
    },
    {
        "id": "post_005",
        "author": {
            "name": "Podemos",
            "handle": "@PODEMOS",
            "verified": True,
            "avatar": "https://example.com/avatar5.jpg"
        },
        "text": "Unos hablan de viviendas con cláusulas y rentabilidades. Otros, de personas sin hogar.",
        "platform": "twitter",
        "timestamp": "2025-10-15T14:05:00Z",
        "engagement": {
            "views": 22100,
            "likes": 890,
            "retweets": 245
        }
    },
]


# New endpoint
@app.get("/api/social-monitoring")
def social_monitoring_endpoint():
    """
    GET /api/social-monitoring
      Query params:
        - topics: comma-separated filter (vivienda,educacion,...). Default: all
        - limit: integer. Default: 10
        - since: ISO8601 datetime. Default: last 24h

    Returns recent social media posts from political figures about monitored topics.
    """
    topics_q = request.args.get("topics", "").lower().split(",")
    topics = [t for t in topics_q if t in TOPIC_LABELS] if topics_q != [""] else []

    limit = int(request.args.get("limit", "10"))
    since = request.args.get("since", "")

    # Filter posts (simplified - in real implementation would filter by topics)
    filtered = SOCIAL_POSTS_CONST[:limit]

    # Sort by timestamp descending (most recent first)
    filtered.sort(key=lambda x: x["timestamp"], reverse=True)

    # JSON response
    payload = {
        "meta": {
            "generated_at": datetime.utcnow().isoformat(timespec="seconds") + "Z",
            "total": len(filtered),
            "platforms": ["twitter"]
        },
        "posts": filtered
    }
    return jsonify(payload)

# New endpoint
@app.get("/api/message-evaluation")
def message_evaluation_endpoint():
    """
    GET /api/message-evaluation
      Query params:
        - min_score: integer (0-100). Default: 0
        - limit: integer. Default: 5

    Returns evaluations of political messages/discourse with scores.
    """
    min_score = int(request.args.get("min_score", "0"))
    limit = int(request.args.get("limit", "5"))

    # Filter evaluations
    filtered = [msg for msg in MESSAGE_EVALUATIONS_CONST if msg["score"] >= min_score]

    # Sort by score descending
    filtered.sort(key=lambda x: x["score"], reverse=True)

    # Apply limit
    filtered = filtered[:limit]

    # JSON response
    payload = {
        "meta": {
            "generated_at": datetime.utcnow().isoformat(timespec="seconds") + "Z",
            "total": len(filtered),
            "score_range": [0, 100]
        },
        "evaluations": filtered
    }
    return jsonify(payload)
# ============== CORS & OPTIONS ==============

@app.route('/api/v1/validacion-intervencion', methods=['OPTIONS'])
@app.route('/api/v1/encuesta/<encuesta_id>/resultados', methods=['OPTIONS'])
@app.route('/api/v1/encuesta/<encuesta_id>/insights', methods=['OPTIONS'])
@app.route('/api/v1/encuesta/<encuesta_id>/segmentacion', methods=['OPTIONS'])
@app.route('/api/v1/chatbot/consulta', methods=['OPTIONS'])
@app.route('/api/v1/machiavelli/fact_checker', methods=['OPTIONS'])
@app.route('/api/v1/machiavelli/fact_checker/<job_id>', methods=['OPTIONS'])
@app.route('/api/v1/machiavelli/fact_checker/<job_id>/status', methods=['OPTIONS'])
@app.route('/api/v1/machiavelli/fact_checker/<job_id>/result', methods=['OPTIONS'])
@app.route('/api/v1/machiavelli/fact_checker/jobs', methods=['OPTIONS'])
@app.route('/api/v1/machiavelli/audio_fact_checker', methods=['OPTIONS'])
@app.route('/api/v1/machiavelli/query', methods=['OPTIONS'])
@app.route('/api/message-evaluation', methods=['OPTIONS'])
@app.route('/api/media-affinity_socialnets', methods=['OPTIONS'])
@app.route('/api/social-monitoring', methods=['OPTIONS'])
def handle_options(**kwargs):
    """Respond to CORS preflight OPTIONS requests for all API routes."""
    return Response(
        status=200,
        headers={
            'Access-Control-Allow-Origin': '*',
            'Access-Control-Allow-Headers': 'Content-Type',
            'Access-Control-Allow-Methods': 'GET, POST, DELETE, OPTIONS'
        }
    )


@app.route('/health', methods=['GET'])
def health_check():
    """Return overall server health status including subsystem availability."""
    payload = {
        "status": "OK",
        "message": "Machiavelli Audio Fact-Check API is running",
        "rag_available": RAG_AVAILABLE,
        "factcheck_available": FACTCHECK_AVAILABLE,
        "active_jobs": len(jobs),
        "active_sessions": len(sessions),
        "active_transcribers": len(active_transcribers)
    }
    payload["political_communication"] = build_political_comm_health_payload()
    return jsonify(payload)




# Iniciar el scheduler de news_enricher
start_news_enricher_scheduler()

if __name__ == '__main__':
    print("🎤 Machiavelli Audio Fact-Check API started!")
    print("🚀 Loading political communication datasets...")
    load_json_data()
    print("⏰ Automatic reload every hour (lazy loading in endpoints)")
    print(f"📊 RAG System: {'✅ Available' if RAG_AVAILABLE else '❌ Not Available'}")
    print(f"🔍 Fact-checking: {'✅ Available' if FACTCHECK_AVAILABLE else '❌ Not Available'}")
    print("\nWebSocket Audio Endpoints:")
    print("  WS   ws://localhost:5000/socket.io/ (start_recording, audio_chunk, stop_recording)")
    print("\nHTTP Fact-Check Endpoints:")
    print("  POST http://localhost:5000/api/v1/machiavelli/fact_checker")
    print("  GET  http://localhost:5000/api/v1/machiavelli/fact_checker/<job_id>")
    print("  POST http://localhost:5000/api/v1/machiavelli/query")
    print("\nHealth: GET http://localhost:5000/health")

    print(f'GOOGLE_API_KEY: {os.getenv("GOOGLE_API_KEY", "#N/A")}')

    socketio.run(
        app,
        debug=True,
        host='0.0.0.0',
        port=5000,
        use_reloader=False  # Important: disable reloader in debug mode
    )
