# Gover.Me

**Plataforma de analisis politico y gobernanza para Andalucia, Espana.**

Gover.Me agrega, analiza y verifica discurso politico y datos gubernamentales de la Junta de Andalucia y medios regionales. Combina inferencia local de LLMs, busqueda vectorial, scraping web y visualizacion para ofrecer un toolkit analitico orientado a la transparencia politica y la participacion ciudadana.

---

## Funcionalidades

- **Fact-checking de discurso politico** -- descompone textos en afirmaciones, las contrasta con fuentes RAG y las clasifica con LLM. Genera anotaciones visuales con matplotlib y PIL.
- **Chatbot RAG** -- consulta documentos gubernamentales almacenados en Milvus (base vectorial) y PostgreSQL, con respuestas en streaming.
- **Analisis de datos turisticos** -- ingesta datasets Excel de [Nexus Andalucia](https://nexus.andalucia.org/es/data/informes/ultimos-datos-turisticos/) y responde preguntas en lenguaje natural via indexacion FAISS.
- **Scraping de noticias** -- recolecta articulos de multiples medios regionales y del [portal de datos abiertos de la Junta](https://www.juntadeandalucia.es/datosabiertos/portal.html).
- **Transcripcion de audio** -- procesamiento en tiempo real via WebSocket.
- **Busqueda web profunda** -- busqueda adaptativa con optimizacion de calidad usando Thompson Sampling sobre DuckDuckGo.
- **Recoleccion de Twitter/X** -- cosecha tweets con snscrape para analisis de discurso en redes sociales.

### Fuentes de datos

| Fuente | URL |
|--------|-----|
| Datos Abiertos Junta de Andalucia | https://www.juntadeandalucia.es/datosabiertos/portal.html |
| datos.gob.es (Andalucia) | https://datos.gob.es/es/iniciativas/datos-abiertos-de-la-junta-de-andalucia |
| IECA (Estadisticas) | https://www.juntadeandalucia.es/organismos/ieca.html |
| Nexus Andalucia (Turismo) | https://nexus.andalucia.org/es/data/informes/ultimos-datos-turisticos/ |
| ABC Sevilla | Prensa regional |
| Diario Sur | Prensa regional |
| Diario de Sevilla | Prensa regional |

---

## Stack tecnologico

| Capa | Tecnologia |
|------|------------|
| Lenguaje | Python 3.12 |
| Framework web | Flask + Flask-SocketIO (eventlet) |
| Orquestacion LLM | LangChain + vLLM (inferencia local, desplegado como servicio systemd) |
| Base de datos vectorial | Milvus (standalone, via Docker) |
| Base de datos relacional | PostgreSQL (metadatos de documentos) |
| Indexacion local | FAISS (datos turisticos) |
| Embeddings | SentenceTransformers |
| Busqueda web | DuckDuckGo Search API |
| Redes sociales | snscrape (Twitter/X) |
| Scraping web | BeautifulSoup + requests |
| Visualizacion | matplotlib + PIL |

---

## Estructura del proyecto

```
Gover.Me/
├── machiavelli/                # Servidor API principal
│   ├── app.py                  # Servidor Flask WebSocket (produccion)
│   ├── factcheck.py            # Motor de fact-checking
│   ├── query.py                # Sistema RAG mejorado
│   └── mock_app.py             # API mock para testing
├── rag/                        # Pipeline RAG
│   ├── chatbot.py              # Chatbot RAG core (Milvus + PostgreSQL)
│   ├── stack_check.py          # Health check de infraestructura
│   └── document_tools/         # Pipeline de ingesta de documentos
│       ├── scraper_dato.py     # Scraper Junta de Andalucia
│       ├── scraper_diarios.py  # Scraper de noticias (ABC, Diario Sur, etc.)
│       ├── uploader.py         # Carga a base vectorial
│       └── remove_duplicates.py
├── deepsearcher/               # Motor de busqueda web adaptativo
│   ├── deepsearch.py           # Busqueda con Thompson Sampling
│   ├── adaptive_deepsearch.py  # Busqueda con optimizacion de calidad
│   ├── interactive_search.py   # CLI interactivo
│   ├── run.py                  # Launcher simple
│   └── debug_search.py         # Depuracion de busquedas
├── tourism/                    # Pipeline de datos turisticos
│   ├── qa.py                   # QA turistico con FAISS
│   └── extractor.py            # Extractor de datos Excel
├── twitter/                    # Redes sociales
│   └── tweet_harvest.py        # Recolector de tweets
├── tasks/                      # Automatizacion
│   └── scheduler.py            # Scheduler tipo cron
├── tests/                      # Suite de tests
├── data/                       # Archivos JSON de datos
├── audio/                      # Archivos de audio
├── nexus/                      # Fuentes Excel de turismo
├── docs/                       # Documentacion y specs de API
└── rag_document_data/          # Archivo de documentos scrapeados
```

---

## Instalacion

### Prerrequisitos

- Python 3.12+
- Docker y Docker Compose (para Milvus)
- Una instancia vLLM corriendo (local o remota)
- Base de datos PostgreSQL

### Variables de entorno

Crear un archivo `.env` en la raiz del proyecto:

```env
# Endpoint vLLM (API compatible con OpenAI)
OPENAI_API_KEY=<tu-api-key-vllm>
OPENAI_API_BASE=<tu-endpoint-vllm>

# Milvus
MILVUS_HOST=<host-milvus>
MILVUS_PORT=19530

# PostgreSQL
POSTGRES_HOST=<host-postgres>
POSTGRES_PORT=5432
POSTGRES_DB=<nombre-base-datos>
POSTGRES_USER=<usuario>
POSTGRES_PASSWORD=<password>
```

### Milvus (Base vectorial)

Despliegue standalone recomendado:

```bash
# Descargar script de Milvus standalone
curl -sfL https://raw.githubusercontent.com/milvus-io/milvus/master/scripts/standalone_embed.sh \
  -o standalone_embed.sh

# Iniciar Milvus
bash standalone_embed.sh start
```

### vLLM (Inferencia local de LLM)

vLLM corre como servicio systemd:

```bash
sudo systemctl daemon-reload
sudo systemctl restart vllm.service
sudo systemctl status vllm.service

# Ver logs en tiempo real
sudo journalctl -u vllm.service -f
```

### Dependencias Python

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

---

## Ejecucion

### Servidor API principal

```bash
python -m machiavelli.app
```

Inicia una aplicacion Flask-SocketIO con eventlet, exponiendo endpoints REST y eventos WebSocket.

### Health check de infraestructura

```bash
python -m rag.stack_check
```

Verifica conectividad con Milvus, PostgreSQL y vLLM.

### Scheduler de tareas

```bash
python -m tasks.scheduler
```

Ejecuta tareas automatizadas de scraping e ingesta en un schedule periodico.

### Busqueda profunda (CLI)

```bash
python -m deepsearcher.run
```

Modo interactivo con metricas de calidad:

```bash
python -m deepsearcher.interactive_search
```

### Tests

```bash
pytest tests/
```

---

## Endpoints principales

Todos los endpoints son servidos por `machiavelli/app.py`.

| Metodo | Ruta | Descripcion |
|--------|------|-------------|
| `POST` | `/api/fact-checker` | Verifica un texto politico. Retorna resultados anotados con referencias a fuentes. |
| `POST` | `/api/query` | Consulta RAG con respuesta en streaming. Busca en documentos gubernamentales y retorna respuestas fundamentadas. |

### Eventos WebSocket

El servidor expone eventos WebSocket en tiempo real (via Flask-SocketIO) para:

- **Transcripcion de audio** -- enviar chunks de audio y recibir transcripciones en tiempo real.
- **Streaming de consultas** -- recibir respuestas RAG token a token a medida que se generan.

---

## Despliegue

### Backup

```bash
tar -czf /mnt/BackUps/Gover.Me_backup_$(date +%Y-%m-%d_%H-%M-%S).tar.gz \
  -C /home/maxim/PycharmProjects \
  --exclude='.venv' --exclude='venv' --exclude='volumes' \
  Gover.Me
```

### Sincronizacion remota

```bash
rsync -avz \
  --exclude='rag_document_data' \
  --exclude='volumes/' \
  --exclude='venv/' \
  --exclude='.venv/' \
  -e ssh \
  /home/maxim/PycharmProjects/Gover.Me/ \
  user@remote-host:/path/to/Gover.Me/
```

---

## Licencia

Privado / propietario. Todos los derechos reservados.
