"""RAG chatbot with a Rich CLI, backed by Milvus vector search, PostgreSQL
metadata storage, and a local or vLLM-served language model for generation.
"""

# requirements.txt
"""
# Versiones compatibles y actualizadas
langchain==0.3.27
langchain-community==0.3.27
langchain-openai==0.0.6
langchain-core>=0.3.72
langsmith>=0.1.17
pymilvus==2.3.4
psycopg2-binary==2.9.9
python-dotenv==1.0.0
openai==1.6.1
tiktoken>=0.11.0
rich==13.7.0
click==8.1.7
colorama==0.4.6
"""

import os
import json
import sys
import argparse
import time
import hashlib
from typing import List, Dict, Any, Optional
from dotenv import load_dotenv
import psycopg2
from psycopg2.extras import RealDictCursor
from langchain.text_splitter import RecursiveCharacterTextSplitter
from pymilvus import FieldSchema, CollectionSchema, DataType, Collection, utility, exceptions, connections
from sentence_transformers import SentenceTransformer

EMBED_DIM_MAP = {
    "text-embedding-3-small": 1536,
    "text-embedding-3-large": 3072,
    "intfloat/e5-small-v2": 384,
    "intfloat/e5-base-v2": 768,
    "intfloat/e5-large-v2": 1024,
    "BAAI/bge-base-en-v1.5": 768,
    "BAAI/bge-large-en-v1.5": 1024,
    "BAAI/bge-m3": 1024,
}

# Imports actualizados de LangChain - versiones modernas
try:
    from langchain_openai import OpenAIEmbeddings, OpenAI
    from langchain_openai import ChatOpenAI
    from langchain_community.vectorstores import Milvus

    LANGCHAIN_NEW = True
except ImportError:
    try:
        # Fallback a imports antiguos
        from langchain.embeddings import OpenAIEmbeddings
        from langchain.llms import OpenAI
        from langchain.vectorstores import Milvus

        LANGCHAIN_NEW = False
    except ImportError as e:
        print(f"❌ Error crítico con LangChain: {e}")
        print("🔧 Ejecuta: pip install --upgrade langchain langchain-openai langchain-community")
        sys.exit(1)

from langchain.chains import RetrievalQA
from langchain.schema import Document

# Bibliotecas para CLI
try:
    from rich.console import Console
    from rich.panel import Panel
    from rich.table import Table
    from rich.progress import Progress, SpinnerColumn, TextColumn
    from rich.prompt import Prompt, Confirm
    from rich.text import Text
    from rich.layout import Layout
    from rich.live import Live
    from rich.markdown import Markdown
    import click
    from colorama import init, Fore, Back, Style

    # Inicializar colorama y rich
    init(autoreset=True)
    console = Console()
    CLI_AVAILABLE = True

except ImportError as e:
    print(f"⚠️ Dependencias CLI no instaladas: {e}")
    print("📦 Instala con: pip install rich click colorama")
    CLI_AVAILABLE = False


    # Mock classes para que el código no falle
    class Console:
        def print(self, *args, **kwargs): print(*args)

        def status(self, msg): return self

        def __enter__(self): return self

        def __exit__(self, *args): pass


    console = Console()

# Cargar variables de entorno
load_dotenv()


# class LocalEmbeddings:
#     def __init__(self, model_name: str = "all-MiniLM-L6-v2", device: str | None = None):
#         # Force CPU if device not specified
#         if device is None:
#             device = 'cpu'
#
#         self.model = SentenceTransformer(model_name, device=device)
#         self.dimension = self.model.get_sentence_embedding_dimension()
#
#     def embed_query(self, text: str) -> List[float]:
#         # Normaliza si usas COSINE en Milvus
#         vec = self.model.encode(text, normalize_embeddings=True)
#         return vec.tolist()
#
#     def embed_documents(self, texts: List[str]) -> List[List[float]]:
#         vecs = self.model.encode(texts, normalize_embeddings=True, batch_size=64)
#         return [v.tolist() for v in vecs]

#
# class LocalEmbeddings:
#     def __init__(self, model_name: str = "all-MiniLM-L6-v2", device=None):
#         import torch
#         import os
#         import shutil
#
#         # Clear cached models that might have GPU metadata
#         cache_dir = os.path.expanduser("~/.cache/torch/sentence_transformers/")
#         model_cache = os.path.join(cache_dir, model_name.replace("/", "_"))
#         if os.path.exists(model_cache):
#             print(f"Clearing cached model: {model_cache}")
#             shutil.rmtree(model_cache, ignore_errors=True)
#
#         # Force CPU environment
#         os.environ["CUDA_VISIBLE_DEVICES"] = ""
#         torch.cuda.is_available = lambda: False
#
#         try:
#             self.model = SentenceTransformer(model_name, device='cpu')
#         except Exception as e:
#             print(f"Standard loading failed: {e}")
#             # Force download fresh copy
#             self.model = SentenceTransformer(model_name, cache_folder=None, device='cpu')
#
#         self.dimension = self.model.get_sentence_embedding_dimension()
#         print(f"Loaded {model_name} on CPU with dimension: {self.dimension}")
#
#     def embed_query(self, text: str):
#         vec = self.model.encode(text, normalize_embeddings=True)
#         return vec.tolist()
#
#     def embed_documents(self, texts):
#         vecs = self.model.encode(texts, normalize_embeddings=True, batch_size=64)
#         return [v.tolist() for v in vecs]
#

import threading
import torch
import os
import shutil
from sentence_transformers import SentenceTransformer


class LocalEmbeddings:
    """Thread-safe singleton wrapper around SentenceTransformer for CPU-only embedding."""

    _instances = {}
    _lock = threading.Lock()

    def __new__(cls, model_name: str = "all-MiniLM-L6-v2", device=None):
        with cls._lock:
            if model_name not in cls._instances:
                instance = super().__new__(cls)
                cls._instances[model_name] = instance
                instance._initialized = False
            return cls._instances[model_name]

    def __init__(self, model_name: str = "all-MiniLM-L6-v2", device=None):
        if self._initialized:
            return

        # Force CPU environment
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
        torch.cuda.is_available = lambda: False

        # Clear cache to avoid meta tensor issues
        cache_dir = os.path.expanduser("~/.cache/torch/sentence_transformers/")
        model_cache = os.path.join(cache_dir, model_name.replace("/", "_"))
        if os.path.exists(model_cache):
            print(f"Clearing cached model: {model_cache}")
            shutil.rmtree(model_cache, ignore_errors=True)

        try:
            # Load with explicit device and avoid meta tensors
            self.model = SentenceTransformer(
                model_name,
                device='cpu',
                cache_folder=None,  # Don't use cache
                trust_remote_code=False
            )
        except Exception as e:
            print(f"Standard loading failed: {e}")
            # Force fresh download
            self.model = SentenceTransformer(
                model_name,
                cache_folder=None,
                device='cpu'
            )

        self.dimension = self.model.get_sentence_embedding_dimension()
        self._initialized = True
        print(f"Loaded {model_name} on CPU with dimension: {self.dimension}")

    def embed_query(self, text: str):
        """Return the normalized embedding vector for a single text string."""
        vec = self.model.encode(text, normalize_embeddings=True)
        return vec.tolist()

    def embed_documents(self, texts):
        """Return normalized embedding vectors for a batch of text strings."""
        vecs = self.model.encode(texts, normalize_embeddings=True, batch_size=64)
        return [v.tolist() for v in vecs]





class MilvusManager:
    """Gestor de Milvus para vectores - inicialización lazy"""

    def __init__(self, collection_name: str = "document_vectors"):
        self.collection_name = collection_name
        self.collection = None
        self._dim = None
        self._connected = False

        # No conectar automáticamente, solo almacenar configuración
        self.host = os.getenv("MILVUS_HOST", "localhost")
        self.port = os.getenv("MILVUS_PORT", "19530")

    def connect(self):
        """Conectar a Milvus solo cuando sea necesario"""
        if self._connected:
            return

        try:
            connections.connect(
                host=self.host,
                port=self.port,
                timeout=10
            )
            self._connected = True
            print("✅ Conectado a Milvus")
        except Exception as e:
            print(f"❌ Error conectando a Milvus: {e}")
            print("💡 Asegúrate de que Milvus esté ejecutándose:")
            print("   ./standalone_embed.sh start")
            raise

    def ensure_collection(self, dim: int):
        """Crear/verificar colección solo cuando sea necesario"""
        if not self._connected:
            self.connect()

        name = self.collection_name

        if utility.has_collection(name):
            # Si existe, validar la dimensión
            self.collection = Collection(name)
            vec_field = next(f for f in self.collection.schema.fields if f.name == "vector")
            existing_dim = vec_field.params.get("dim") or getattr(vec_field, "dim", None)
            existing_dim = int(existing_dim) if existing_dim is not None else None

            if existing_dim != int(dim):
                raise RuntimeError(
                    f"La colección '{name}' ya existe con dim={existing_dim}, "
                    f"pero necesitas dim={dim}. "
                    f"Elimina la colección o usa otro nombre."
                )
        else:
            # Crear nueva colección
            fields = [
                FieldSchema(name="id", dtype=DataType.VARCHAR, max_length=100, is_primary=True),
                FieldSchema(name="vector", dtype=DataType.FLOAT_VECTOR, dim=int(dim)),
                FieldSchema(name="document_id", dtype=DataType.INT64),
                FieldSchema(name="chunk_index", dtype=DataType.INT64),
                FieldSchema(name="text", dtype=DataType.VARCHAR, max_length=10000),
            ]
            schema = CollectionSchema(fields, "Colección de vectores de documentos")
            self.collection = Collection(name=name, schema=schema)

            # Crear índice
            index_params = {
                "metric_type": "COSINE",
                "index_type": "HNSW",
                "params": {"M": 16, "efConstruction": 100},
            }
            self.collection.create_index("vector", index_params)
            print(f"✅ Colección creada: '{name}' (dim={dim})")

        # Cargar en memoria
        try:
            self.collection.load()
            print(f"✅ Colección cargada en memoria: '{name}' (dim={dim})")
        except Exception as e:
            print(f"⚠️ Advertencia al cargar colección '{name}': {e}")

    def insert_vectors(self, vectors_data: List[Dict], dim: int):
        """Insertar vectores en Milvus"""
        if not vectors_data:
            return

        # Asegurar que la colección existe
        self.ensure_collection(dim)

        # Preparar datos
        ids = [item["id"] for item in vectors_data]
        vectors = [item["vector"] for item in vectors_data]
        document_ids = [item["document_id"] for item in vectors_data]
        chunk_indices = [item["chunk_index"] for item in vectors_data]
        texts = [item["text"] for item in vectors_data]

        # Insertar
        data = [ids, vectors, document_ids, chunk_indices, texts]
        self.collection.insert(data)
        self.collection.flush()

    def search_similar(self, query_vector: List[float], top_k: int = 5, dim: int = None) -> List[Dict]:
        """Buscar vectores similares"""
        if dim:
            self.ensure_collection(dim)
        elif not self.collection:
            raise RuntimeError("Colección no inicializada. Procesa algunos documentos primero.")

        search_params = {"metric_type": "COSINE", "params": {"ef": 64}}

        results = self.collection.search(
            data=[query_vector],
            anns_field="vector",
            param=search_params,
            limit=top_k,
            output_fields=["document_id", "chunk_index", "text"]
        )

        similar_docs = []
        for hits in results:
            for hit in hits:
                similar_docs.append({
                    "id": hit.id,
                    "score": hit.score,
                    "document_id": hit.entity.get("document_id"),
                    "chunk_index": hit.entity.get("chunk_index"),
                    "text": hit.entity.get("text")
                })

        return similar_docs


class DatabaseManager:
    """Gestor de base de datos PostgreSQL para metadatos"""

    def __init__(self):
        self.connection = None
        self.connect()

    def connect(self):
        """Conectar a PostgreSQL"""
        try:
            self.connection = psycopg2.connect(
                host=os.getenv("POSTGRES_HOST", "localhost"),
                database=os.getenv("POSTGRES_DB", "rag_chatbot"),
                user=os.getenv("POSTGRES_USER", "postgres"),
                password=os.getenv("POSTGRES_PASSWORD", "password"),
                port=os.getenv("POSTGRES_PORT", "5432")
            )
            self.create_tables()
        except Exception as e:
            print(f"Error conectando a PostgreSQL: {e}")

    def create_tables(self):
        """Crear tablas necesarias"""
        cursor = self.connection.cursor()

        # Tabla para documentos
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS documents (
                id SERIAL PRIMARY KEY,
                titulo VARCHAR(500),
                autor VARCHAR(200),
                fecha TIMESTAMP,
                url_original TEXT,
                formato_detectado VARCHAR(100),
                contenido_completo TEXT,
                vector_ids TEXT[],
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)

        # Asegurar columnas adicionales (migraciones simples)
        cursor.execute("ALTER TABLE documents ADD COLUMN IF NOT EXISTS content_hash VARCHAR(32)")
        cursor.execute("ALTER TABLE documents ADD COLUMN IF NOT EXISTS updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP")

        # Tabla para chunks de texto
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS text_chunks (
                id SERIAL PRIMARY KEY,
                document_id INTEGER REFERENCES documents(id),
                chunk_text TEXT,
                chunk_index INTEGER,
                vector_id VARCHAR(100),
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)

        self.connection.commit()
        cursor.close()

    def document_exists(self, doc_data: Dict) -> Optional[int]:
        """Verificar si el documento ya existe basándose en URL o hash del contenido"""
        cursor = self.connection.cursor()

        # Primera verificación: por URL si existe
        if doc_data.get('url_original'):
            cursor.execute(
                "SELECT id FROM documents WHERE url_original = %s",
                (doc_data['url_original'],)
            )
            result = cursor.fetchone()
            if result:
                cursor.close()
                return result[0]

        # Segunda verificación: por hash del contenido
        content = doc_data.get('contenido_completo', '')
        content_hash = hashlib.md5(content.encode('utf-8')).hexdigest()

        cursor.execute(
            "SELECT id FROM documents WHERE content_hash = %s",
            (content_hash,)
        )
        result = cursor.fetchone()
        cursor.close()

        return result[0] if result else None

    def delete_document_vectors(self, doc_id: int):
        """Eliminar vectores de un documento de Milvus"""
        cursor = self.connection.cursor()
        cursor.execute(
            "SELECT vector_id FROM text_chunks WHERE document_id = %s",
            (doc_id,)
        )
        vector_ids = [row[0] for row in cursor.fetchall()]
        cursor.close()

        if vector_ids:
            # Aquí eliminaríamos de Milvus si la API lo permitiera
            # Por ahora, solo limpiamos PostgreSQL
            cursor = self.connection.cursor()
            cursor.execute(
                "DELETE FROM text_chunks WHERE document_id = %s",
                (doc_id,)
            )
            self.connection.commit()
            cursor.close()

    def update_document(self, doc_id: int, doc_data: Dict) -> int:
        """Actualizar documento existente"""
        cursor = self.connection.cursor()

        # Calcular hash del contenido
        content = doc_data.get('contenido_completo', '')
        content_hash = hashlib.md5(content.encode('utf-8')).hexdigest()

        cursor.execute("""
            UPDATE documents 
            SET titulo = %s, autor = %s, fecha = %s, url_original = %s, 
                formato_detectado = %s, contenido_completo = %s, content_hash = %s,
                updated_at = CURRENT_TIMESTAMP
            WHERE id = %s
        """, (
            doc_data.get('titulo'),
            doc_data.get('autor'),
            doc_data.get('fecha'),
            doc_data.get('url_original'),
            doc_data.get('formato_detectado'),
            content,
            content_hash,
            doc_id
        ))

        self.connection.commit()
        cursor.close()
        return doc_id

    def insert_document(self, doc_data: Dict) -> int:
        """Insertar documento en PostgreSQL"""
        cursor = self.connection.cursor()
        content = doc_data.get('contenido_completo', '')
        content_hash = hashlib.md5(content.encode('utf-8')).hexdigest()
        cursor.execute(
            """
            INSERT INTO documents (titulo, autor, fecha, url_original, formato_detectado, contenido_completo, content_hash)
            VALUES (%s, %s, %s, %s, %s, %s, %s)
            RETURNING id
            """,
            (
                doc_data.get('titulo'),
                doc_data.get('autor'),
                doc_data.get('fecha'),
                doc_data.get('url_original'),
                doc_data.get('formato_detectado'),
                content,
                content_hash,
            )
        )
        new_id = cursor.fetchone()[0]
        self.connection.commit()
        cursor.close()
        return new_id

    def insert_chunk(self, document_id: int, chunk_text: str, chunk_index: int, vector_id: str):
        """Insertar chunk de texto"""
        cursor = self.connection.cursor()

        cursor.execute("""
            INSERT INTO text_chunks (document_id, chunk_text, chunk_index, vector_id)
            VALUES (%s, %s, %s, %s)
        """, (document_id, chunk_text, chunk_index, vector_id))

        self.connection.commit()
        cursor.close()

    def get_document_stats(self) -> Dict[str, int]:
        """Obtener estadísticas de documentos"""
        cursor = self.connection.cursor()

        # Contar documentos
        cursor.execute("SELECT COUNT(*) FROM documents")
        doc_count = cursor.fetchone()[0]

        # Contar chunks
        cursor.execute("SELECT COUNT(*) FROM text_chunks")
        chunk_count = cursor.fetchone()[0]

        # Documentos por formato
        cursor.execute("""
            SELECT formato_detectado, COUNT(*) 
            FROM documents 
            WHERE formato_detectado IS NOT NULL
            GROUP BY formato_detectado
        """)
        formats = dict(cursor.fetchall())

        cursor.close()
        return {
            "total_documents": doc_count,
            "total_chunks": chunk_count,
            "formats": formats
        }

    def list_documents(self, limit: int = 20) -> List[Dict]:
        """Listar documentos en la base de datos"""
        cursor = self.connection.cursor(cursor_factory=RealDictCursor)
        cursor.execute("""
            SELECT id, titulo, autor, fecha, formato_detectado, 
                   LENGTH(contenido_completo) as content_length,
                   created_at, updated_at
            FROM documents 
            ORDER BY created_at DESC 
            LIMIT %s
        """, (limit,))

        results = cursor.fetchall()
        cursor.close()
        return [dict(row) for row in results]

    def delete_document(self, doc_id: int) -> bool:
        """Eliminar documento y sus chunks"""
        try:
            cursor = self.connection.cursor()

            # Eliminar chunks (los vectores de Milvus se quedan por limitaciones de la API)
            cursor.execute("DELETE FROM text_chunks WHERE document_id = %s", (doc_id,))

            # Eliminar documento
            cursor.execute("DELETE FROM documents WHERE id = %s", (doc_id,))

            rows_affected = cursor.rowcount
            self.connection.commit()
            cursor.close()

            return rows_affected > 0
        except Exception as e:
            print(f"Error eliminando documento: {e}")
            return False

    def clear_all_documents(self) -> bool:
        """Limpiar todas las tablas (¡CUIDADO!)"""
        try:
            cursor = self.connection.cursor()
            cursor.execute("TRUNCATE text_chunks, documents RESTART IDENTITY CASCADE")
            self.connection.commit()
            cursor.close()
            return True
        except Exception as e:
            print(f"Error limpiando base de datos: {e}")
            return False

    def get_document_by_id(self, doc_id: int) -> Optional[Dict]:
        """Obtener documento por ID"""
        cursor = self.connection.cursor(cursor_factory=RealDictCursor)
        cursor.execute("SELECT * FROM documents WHERE id = %s", (doc_id,))
        result = cursor.fetchone()
        cursor.close()
        return dict(result) if result else None


class RAGChatbot:
    """Chatbot principal con RAG - inicialización lazy"""

    def __init__(self):
        # Inicializar solo componentes básicos
        self.db_manager = DatabaseManager()
        self.milvus_manager = MilvusManager()

        # Configurar embeddings
        self.embedding_model = os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2")
        self.embeddings = LocalEmbeddings("all-MiniLM-L6-v2")
        self.embedding_dim = self.embeddings.dimension

        # Configurar LLM
        base_url = os.getenv("OPENAI_BASE_URL") or os.getenv("VLLM_BASE_URL")
        api_key = os.getenv("OPENAI_API_KEY") or os.getenv("VLLM_API_KEY") or "EMPTY"
        chat_model = os.getenv("VLLM_MODEL", "llama-3-8b-instruct")

        self.llm = ChatOpenAI(
            model=chat_model,
            base_url=base_url,
            api_key=api_key,
            temperature=0.2
        )

        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len
        )

        # Resolver dimensión
        forced_dim = os.getenv("RAG_VECTOR_DIM")
        if forced_dim:
            self.vector_dim = int(forced_dim)
        else:
            dim_from_table = EMBED_DIM_MAP.get(self.embedding_model)
            if dim_from_table:
                self.vector_dim = dim_from_table
            else:
                self.vector_dim = self.embedding_dim

        print("✅ RAG Chatbot inicializado (sin cargar Milvus)")
        print(f"📏 Dimensión de vectores: {self.vector_dim}")

    def process_document(self, doc_data: Dict, force_update: bool = False):
        """Procesar y almacenar documento con verificación de duplicados"""

        # Verificar si el documento ya existe
        existing_doc_id = self.db_manager.document_exists(doc_data)

        if existing_doc_id and not force_update:
            print(f"⚠️ Documento ya existe (ID: {existing_doc_id}): {doc_data.get('titulo', 'Sin título')}")
            print("   Use force_update=True para actualizar o 'reload' en CLI")
            return existing_doc_id

        if existing_doc_id and force_update:
            print(f"🔄 Actualizando documento existente: {doc_data.get('titulo', 'Sin título')}")
            # Eliminar vectores antiguos
            self.db_manager.delete_document_vectors(existing_doc_id)
            # Actualizar metadatos
            doc_id = self.db_manager.update_document(existing_doc_id, doc_data)
        else:
            print(f"📄 Procesando nuevo documento: {doc_data.get('titulo', 'Sin título')}")
            # Insertar nuevo documento
            doc_id = self.db_manager.insert_document(doc_data)

        # Dividir texto en chunks
        content = doc_data.get('contenido_completo', '')
        if not content.strip():
            print("⚠️ Documento sin contenido, saltando procesamiento de vectores")
            return doc_id

        chunks = self.text_splitter.split_text(content)

        if not chunks:
            print("⚠️ No se pudieron crear chunks del documento")
            return doc_id

        # Generar embeddings y almacenar
        vectors_data = []
        print(f"🔄 Generando {len(chunks)} embeddings...")

        for i, chunk in enumerate(chunks):
            if not chunk.strip():  # Saltar chunks vacíos
                continue

            try:
                vector = self.embeddings.embed_query(chunk)
                vector_id = f"{doc_id}_{i}"

                # Almacenar chunk en PostgreSQL
                self.db_manager.insert_chunk(doc_id, chunk, i, vector_id)

                # Preparar para Milvus
                vectors_data.append({
                    "id": vector_id,
                    "vector": vector,
                    "document_id": doc_id,
                    "chunk_index": i,
                    "text": chunk[:9000]  # Milvus tiene límite de caracteres
                })
            except Exception as e:
                print(f"⚠️ Error procesando chunk {i}: {e}")
                continue

        if vectors_data:
            # Insertar vectores en Milvus (lazy loading)
            try:
                self.milvus_manager.insert_vectors(vectors_data, self.vector_dim)
                print(f"✅ Documento procesado: {len(vectors_data)} vectores almacenados")
            except Exception as e:
                print(f"❌ Error almacenando vectores en Milvus: {e}")
        else:
            print("⚠️ No se generaron vectores válidos")

        return doc_id

    def query(self, question: str, top_k: int = 5) -> str:
        """Realizar consulta con RAG: embed → retrieve → generar respuesta."""
        try:
            if not question or not question.strip():
                return "❌ La pregunta está vacía."

            # 1) Embedding de la consulta
            try:
                query_vector = self.embeddings.embed_query(question)
            except Exception as e:
                return f"❌ Error generando el embedding de la consulta: {e}"

            # 2) Recuperación en Milvus (con lazy loading)
            try:
                similar = self.milvus_manager.search_similar(query_vector, top_k=top_k, dim=self.vector_dim)
            except Exception as e:
                return f"❌ Error buscando en Milvus: {e}"

            if not similar:
                return "No encontré información relevante para responder tu pregunta."

            # 3) Construir contexto
            max_chars = int(os.getenv("RAG_MAX_CONTEXT_CHARS", "12000"))
            seen = set()
            ctx_parts = []
            current_len = 0

            for hit in similar:
                did = hit.get("document_id")
                cidx = hit.get("chunk_index")
                txt = (hit.get("text") or "").strip()
                key = (did, cidx)

                if not txt or key in seen:
                    continue
                if current_len + len(txt) + 2 > max_chars:
                    break

                seen.add(key)
                ctx_parts.append(txt)
                current_len += len(txt) + 2

            if not ctx_parts:
                return "No encontré fragmentos de contexto suficientes para responder."

            context = "\n\n".join(ctx_parts)

            # 4) Prompt para el LLM
            prompt = (
                "Eres un asistente que responde en español de forma clara y concisa, "
                "basándote exclusivamente en el CONTEXTO provisto. "
                "Si la respuesta no está en el contexto, indica que no tienes suficiente información.\n\n"
                f"CONTEXTO:\n{context}\n\n"
                f"PREGUNTA: {question}\n\n"
                "RESPUESTA:"
            )

            # 5) Llamar al LLM
            try:
                response = self.llm.invoke(prompt)
                return getattr(response, "content", str(response)).strip()

            except Exception as e:
                return f"❌ Error generando la respuesta del modelo: {e}"

        except Exception as e:
            return f"❌ Error en la consulta: {e}"

    def add_documents_from_json(self, json_files: List[str], force_update: bool = False):
        """Agregar documentos desde archivos JSON"""
        loaded = 0
        updated = 0
        errors = 0

        for json_file in json_files:
            try:
                if not os.path.exists(json_file):
                    print(f"⚠️ Archivo no encontrado: {json_file}")
                    errors += 1
                    continue

                with open(json_file, 'r', encoding='utf-8') as f:
                    doc_data = json.load(f)

                # Verificar si existe
                existing_id = self.db_manager.document_exists(doc_data)

                self.process_document(doc_data, force_update=force_update)

                if existing_id and force_update:
                    updated += 1
                elif not existing_id:
                    loaded += 1

            except Exception as e:
                print(f"❌ Error procesando {json_file}: {e}")
                errors += 1

        print(f"📊 Resumen: {loaded} nuevos, {updated} actualizados, {errors} errores")


# Interfaz CLI con Rich
class CLIChatbot:
    """Interfaz de línea de comandos para el chatbot RAG"""

    def __init__(self):
        if not CLI_AVAILABLE:
            raise ImportError("CLI dependencies not available. Install with: pip install rich click colorama")

        self.console = Console()
        self.chatbot = None
        self.conversation_history = []

    def print_banner(self):
        """Mostrar banner de bienvenida"""
        banner = """
╔══════════════════════════════════════════════════════════════════╗
║                    🤖 RAG CHATBOT CLI                        ║
║              Powered by LangChain, Milvus & PostgreSQL      ║
╚══════════════════════════════════════════════════════════════════╝
        """
        self.console.print(banner, style="bold blue")

        # Información del sistema
        info_panel = Panel.fit(
            "[bold green]✅ Sistema iniciado correctamente[/bold green]\n"
            "💾 Base de datos: PostgreSQL + Milvus (lazy loading)\n"
            "🧠 Modelo: Local/vLLM\n"
            "🔍 Búsqueda: Semántica vectorial\n\n"
            "[dim]Escribe 'help' para ver comandos disponibles[/dim]",
            title="[bold]Estado del Sistema[/bold]",
            border_style="green"
        )
        self.console.print(info_panel)
        self.console.print()

    def init_chatbot(self):
        """Inicializar el chatbot con indicador de progreso"""
        with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                console=self.console
        ) as progress:
            task1 = progress.add_task("🔌 Conectando a PostgreSQL...", total=None)
            time.sleep(0.5)
            progress.update(task1, description="✅ PostgreSQL conectado")

            task2 = progress.add_task("⚙️ Configurando Milvus (lazy)...", total=None)
            time.sleep(0.5)
            progress.update(task2, description="✅ Milvus configurado")

            task3 = progress.add_task("🧠 Inicializando modelo local...", total=None)
            time.sleep(0.5)
            progress.update(task3, description="✅ Modelo configurado")

            task4 = progress.add_task("⚙️ Configurando sistema RAG...", total=None)
            self.chatbot = RAGChatbot()
            progress.update(task4, description="✅ Sistema RAG listo")

    def show_help(self):
        """Mostrar comandos disponibles"""
        help_table = Table(title="📚 Comandos Disponibles")
        help_table.add_column("Comando", style="cyan", no_wrap=True)
        help_table.add_column("Descripción", style="white")
        help_table.add_column("Ejemplo", style="dim")

        commands = [
            ("help", "Mostrar esta ayuda", "help"),
            ("load <archivo>", "Cargar documento JSON", "load documento.json"),
            ("reload <archivo>", "Recargar documento (actualizar si existe)", "reload documento.json"),
            ("load-dir <directorio>", "Cargar todos los JSON de un directorio", "load-dir ./docs/"),
            ("reload-dir <directorio>", "Recargar directorio (actualizar duplicados)", "reload-dir ./docs/"),
            ("list", "Listar documentos cargados", "list"),
            ("list <N>", "Listar N documentos más recientes", "list 10"),
            ("delete <ID>", "Eliminar documento por ID", "delete 5"),
            ("clear-all", "⚠️ PELIGRO: Eliminar todos los documentos", "clear-all"),
            ("stats", "Mostrar estadísticas del sistema", "stats"),
            ("history", "Ver historial de conversación", "history"),
            ("clear", "Limpiar historial de conversación", "clear"),
            ("search <consulta>", "Búsqueda directa (sin conversación)", "search procesión"),
            ("export <archivo>", "Exportar conversación", "export chat.txt"),
            ("quit, exit, q", "Salir del programa", "quit"),
        ]

        for cmd, desc, example in commands:
            help_table.add_row(cmd, desc, example)

        self.console.print(help_table)
        self.console.print()

    def load_document(self, filepath: str, force_update: bool = False):
        """Cargar documento JSON"""
        try:
            if not os.path.exists(filepath):
                self.console.print(f"❌ [red]Archivo no encontrado: {filepath}[/red]")
                return

            with self.console.status(f"📥 {'Recargando' if force_update else 'Cargando'} {filepath}..."):
                with open(filepath, 'r', encoding='utf-8') as f:
                    doc_data = json.load(f)
                doc_id = self.chatbot.process_document(doc_data, force_update=force_update)

            action = "recargado" if force_update else "cargado"
            self.console.print(
                f"✅ [green]Documento {action} exitosamente (ID: {doc_id}): {doc_data.get('titulo', 'Sin título')}[/green]")

        except json.JSONDecodeError:
            self.console.print(f"❌ [red]Error: {filepath} no es un JSON válido[/red]")
        except Exception as e:
            self.console.print(f"❌ [red]Error cargando documento: {str(e)}[/red]")

    def load_directory(self, dirpath: str, force_update: bool = False):
        """Cargar todos los archivos JSON de un directorio"""
        try:
            if not os.path.exists(dirpath):
                self.console.print(f"❌ [red]Directorio no encontrado: {dirpath}[/red]")
                return

            json_files = [f for f in os.listdir(dirpath) if f.endswith('.json')]

            if not json_files:
                self.console.print(f"⚠️ [yellow]No se encontraron archivos JSON en {dirpath}[/yellow]")
                return

            action = "Recargando" if force_update else "Cargando"
            with Progress(console=self.console) as progress:
                task = progress.add_task(f"📂 {action} documentos...", total=len(json_files))

                loaded = 0
                updated = 0
                errors = 0

                for filename in json_files:
                    filepath = os.path.join(dirpath, filename)
                    try:
                        with open(filepath, 'r', encoding='utf-8') as f:
                            doc_data = json.load(f)

                        # Verificar si existe antes de procesar
                        existing_id = self.chatbot.db_manager.document_exists(doc_data)

                        self.chatbot.process_document(doc_data, force_update=force_update)

                        if existing_id and force_update:
                            updated += 1
                            progress.update(task, advance=1, description=f"🔄 {filename}")
                        elif not existing_id:
                            loaded += 1
                            progress.update(task, advance=1, description=f"✅ {filename}")
                        else:
                            progress.update(task, advance=1, description=f"⭐️ {filename} (ya existe)")

                    except Exception as e:
                        errors += 1
                        progress.update(task, advance=1, description=f"❌ {filename}")
                        self.console.print(f"[red]Error en {filename}: {str(e)}[/red]")

            # Resumen
            summary = f"📊 Resumen: "
            if loaded > 0:
                summary += f"{loaded} nuevos, "
            if updated > 0:
                summary += f"{updated} actualizados, "
            if errors > 0:
                summary += f"{errors} errores"
            else:
                summary = summary.rstrip(", ")

            self.console.print(f"✅ [green]{summary}[/green]")

        except Exception as e:
            self.console.print(f"❌ [red]Error: {str(e)}[/red]")

    def show_stats(self):
        """Mostrar estadísticas del sistema"""
        if not self.chatbot:
            self.console.print("❌ [red]Chatbot no inicializado[/red]")
            return

        try:
            stats = self.chatbot.db_manager.get_document_stats()

            stats_table = Table(title="📊 Estadísticas del Sistema")
            stats_table.add_column("Métrica", style="cyan")
            stats_table.add_column("Valor", style="green")

            stats_table.add_row("Documentos cargados", str(stats["total_documents"]))
            stats_table.add_row("Chunks de texto", str(stats["total_chunks"]))
            stats_table.add_row("Consultas en sesión", str(len(self.conversation_history)))

            # Mostrar estadísticas por formato
            if stats["formats"]:
                formats_str = ", ".join([f"{fmt}: {count}" for fmt, count in stats["formats"].items()])
                stats_table.add_row("Formatos", formats_str)

            stats_table.add_row("Estado PostgreSQL", "✅ Conectado")
            stats_table.add_row("Estado Milvus", "⚡ Lazy loading")

            self.console.print(stats_table)

        except Exception as e:
            self.console.print(f"❌ [red]Error obteniendo estadísticas: {str(e)}[/red]")

    def list_documents(self, limit: int = 10):
        """Listar documentos cargados"""
        if not self.chatbot:
            self.console.print("❌ [red]Chatbot no inicializado[/red]")
            return

        try:
            docs = self.chatbot.db_manager.list_documents(limit)

            if not docs:
                self.console.print("📄 [dim]No hay documentos cargados[/dim]")
                return

            docs_table = Table(title=f"📚 Últimos {len(docs)} Documentos")
            docs_table.add_column("ID", style="cyan", width=4)
            docs_table.add_column("Título", style="white", width=40)
            docs_table.add_column("Autor", style="yellow", width=15)
            docs_table.add_column("Formato", style="green", width=12)
            docs_table.add_column("Tamaño", style="blue", width=8)
            docs_table.add_column("Fecha", style="dim", width=12)

            for doc in docs:
                # Truncar título si es muy largo
                titulo = doc.get('titulo', 'Sin título')
                if len(titulo) > 37:
                    titulo = titulo[:34] + "..."

                # Formatear tamaño
                content_length = doc.get('content_length', 0)
                if content_length > 1024:
                    size_str = f"{content_length // 1024}KB"
                else:
                    size_str = f"{content_length}B"

                # Formatear fecha
                created_at = doc.get('created_at')
                if created_at:
                    date_str = created_at.strftime('%Y-%m-%d') if hasattr(created_at, 'strftime') else str(created_at)[
                        :10]
                else:
                    date_str = "N/A"

                docs_table.add_row(
                    str(doc.get('id', '')),
                    titulo,
                    doc.get('autor', 'N/A') or 'N/A',
                    doc.get('formato_detectado', 'N/A') or 'N/A',
                    size_str,
                    date_str
                )

            self.console.print(docs_table)
            self.console.print(f"[dim]Mostrando {len(docs)} de {limit} documentos solicitados[/dim]")

        except Exception as e:
            self.console.print(f"❌ [red]Error listando documentos: {str(e)}[/red]")

    def delete_document(self, doc_id: str):
        """Eliminar documento por ID"""
        if not self.chatbot:
            self.console.print("❌ [red]Chatbot no inicializado[/red]")
            return

        try:
            doc_id_int = int(doc_id)
        except ValueError:
            self.console.print(f"❌ [red]ID inválido: {doc_id}[/red]")
            return

        # Confirmar eliminación
        if not Confirm.ask(f"¿Estás seguro de eliminar el documento ID {doc_id_int}?"):
            self.console.print("❌ Eliminación cancelada")
            return

        try:
            success = self.chatbot.db_manager.delete_document(doc_id_int)
            if success:
                self.console.print(f"✅ [green]Documento {doc_id_int} eliminado[/green]")
            else:
                self.console.print(f"❌ [red]No se encontró documento con ID {doc_id_int}[/red]")
        except Exception as e:
            self.console.print(f"❌ [red]Error eliminando documento: {str(e)}[/red]")

    def clear_all_documents(self):
        """Limpiar todos los documentos"""
        if not self.chatbot:
            self.console.print("❌ [red]Chatbot no inicializado[/red]")
            return

        # Doble confirmación para operación peligrosa
        self.console.print("⚠️ [red bold]PELIGRO: Esto eliminará TODOS los documentos y vectores[/red bold]")

        if not Confirm.ask("¿Estás seguro de que quieres eliminar TODO?"):
            self.console.print("❌ Operación cancelada")
            return

        if not Confirm.ask("¿REALMENTE seguro? Esta acción no se puede deshacer"):
            self.console.print("❌ Operación cancelada")
            return

        try:
            with self.console.status("🗑️ Eliminando todos los documentos..."):
                success = self.chatbot.db_manager.clear_all_documents()

            if success:
                self.console.print("✅ [green]Todos los documentos eliminados[/green]")
                self.console.print("⚠️ [yellow]Los vectores en Milvus permanecen (limitación técnica)[/yellow]")
            else:
                self.console.print("❌ [red]Error eliminando documentos[/red]")
        except Exception as e:
            self.console.print(f"❌ [red]Error: {str(e)}[/red]")

    def show_history(self):
        """Mostrar historial de conversación"""
        if not self.conversation_history:
            self.console.print("📄 [dim]No hay historial de conversación[/dim]")
            return

        self.console.print("[bold]📄 Historial de Conversación:[/bold]")
        self.console.print()

        for i, entry in enumerate(self.conversation_history, 1):
            if entry['role'] == 'user':
                self.console.print(f"[bold blue]{i}. 👤 Usuario:[/bold blue] {entry['content']}")
            else:
                self.console.print(f"[bold green]🤖 Asistente:[/bold green]")
                # Usar Markdown para mejor formato de la respuesta
                self.console.print(Markdown(entry['content']))
            self.console.print()

    def clear_history(self):
        """Limpiar historial"""
        if Confirm.ask("¿Estás seguro de que quieres limpiar el historial?"):
            self.conversation_history.clear()
            self.console.print("✅ [green]Historial limpiado[/green]")

    def export_conversation(self, filepath: str):
        """Exportar conversación a archivo"""
        try:
            with open(filepath, 'w', encoding='utf-8') as f:
                f.write("# Conversación RAG Chatbot\n\n")
                for entry in self.conversation_history:
                    if entry['role'] == 'user':
                        f.write(f"**Usuario:** {entry['content']}\n\n")
                    else:
                        f.write(f"**Asistente:** {entry['content']}\n\n")
                        f.write("---\n\n")

            self.console.print(f"✅ [green]Conversación exportada a {filepath}[/green]")
        except Exception as e:
            self.console.print(f"❌ [red]Error exportando: {str(e)}[/red]")

    def process_query(self, query: str) -> str:
        """Procesar consulta y devolver respuesta"""
        if not self.chatbot:
            return "❌ Error: Chatbot no inicializado"

        with self.console.status("🤔 Analizando tu pregunta..."):
            response = self.chatbot.query(query)

        return response

    def chat_loop(self):
        """Loop principal de chat interactivo"""
        self.console.print("[bold]💬 Modo Conversación Iniciado[/bold]")
        self.console.print("[dim]Escribe tu pregunta o comando. 'quit' para salir.[/dim]")
        self.console.print()

        while True:
            try:
                # Prompt personalizado
                user_input = Prompt.ask("[bold blue]Tú[/bold blue]").strip()

                if not user_input:
                    continue

                # Comandos especiales
                if user_input.lower() in ['quit', 'exit', 'q']:
                    break
                elif user_input.lower() == 'help':
                    self.show_help()
                    continue
                elif user_input.lower() == 'stats':
                    self.show_stats()
                    continue
                elif user_input.lower() == 'history':
                    self.show_history()
                    continue
                elif user_input.lower() == 'clear':
                    self.clear_history()
                    continue
                elif user_input.lower() == 'list':
                    self.list_documents()
                    continue
                elif user_input.startswith('list '):
                    try:
                        limit = int(user_input[5:].strip())
                        self.list_documents(limit)
                    except ValueError:
                        self.console.print("❌ [red]Número inválido para 'list'[/red]")
                    continue
                elif user_input.startswith('delete '):
                    doc_id = user_input[7:].strip()
                    self.delete_document(doc_id)
                    continue
                elif user_input.lower() == 'clear-all':
                    self.clear_all_documents()
                    continue
                elif user_input.startswith('load '):
                    filepath = user_input[5:].strip()
                    self.load_document(filepath, force_update=False)
                    continue
                elif user_input.startswith('reload '):
                    filepath = user_input[7:].strip()
                    self.load_document(filepath, force_update=True)
                    continue
                elif user_input.startswith('load-dir '):
                    dirpath = user_input[9:].strip()
                    self.load_directory(dirpath, force_update=False)
                    continue
                elif user_input.startswith('reload-dir '):
                    dirpath = user_input[11:].strip()
                    self.load_directory(dirpath, force_update=True)
                    continue
                elif user_input.startswith('export '):
                    filepath = user_input[7:].strip()
                    self.export_conversation(filepath)
                    continue
                elif user_input.startswith('search '):
                    query = user_input[7:].strip()
                    response = self.process_query(query)
                    self.console.print(f"[bold green]🔍 Resultado:[/bold green]")
                    self.console.print(Markdown(response))
                    self.console.print()
                    continue

                # Consulta normal
                self.conversation_history.append({'role': 'user', 'content': user_input})

                response = self.process_query(user_input)
                self.conversation_history.append({'role': 'assistant', 'content': response})

                # Mostrar respuesta
                self.console.print(f"[bold green]🤖 Asistente:[/bold green]")
                self.console.print(Markdown(response))
                self.console.print()

            except KeyboardInterrupt:
                self.console.print("\n[yellow]Presiona Ctrl+C de nuevo o escribe 'quit' para salir[/yellow]")
                continue
            except EOFError:
                break
            except Exception as e:
                self.console.print(f"❌ [red]Error: {str(e)}[/red]")

        self.console.print("[bold yellow]👋 ¡Hasta luego![/bold yellow]")


# Función principal CLI
if CLI_AVAILABLE:
    @click.command()
    @click.option('--load-docs', '-l', help='Cargar documentos desde directorio al inicio')
    @click.option('--reload-docs', '-r', help='Recargar documentos (actualizar duplicados)')
    @click.option('--single-query', '-q', help='Realizar una sola consulta y salir')
    @click.option('--batch-file', '-b', help='Procesar consultas desde archivo')
    @click.option('--export-on-exit', '-e', help='Archivo donde exportar conversación al salir')
    @click.option('--force-update', '-f', is_flag=True, help='Forzar actualización de documentos duplicados')
    def main_cli(load_docs, reload_docs, single_query, batch_file, export_on_exit, force_update):
        """
        🤖 RAG Chatbot - Interfaz de línea de comandos
        """
        main_function(load_docs, reload_docs, single_query, batch_file, export_on_exit, force_update)
else:
    def main_cli():
        print("⚠️ Modo CLI completo no disponible (faltan dependencias)")
        print("📦 Instala con: pip install rich click colorama")
        print("🔄 Iniciando modo básico...")
        main_function(None, None, None, None, None, False)


def main_function(load_docs, reload_docs, single_query, batch_file, export_on_exit, force_update):
    """Función principal que funciona con o sin CLI avanzada"""

    if CLI_AVAILABLE:
        cli = CLIChatbot()

        try:
            # Banner de bienvenida
            cli.print_banner()

            # Inicializar sistema
            cli.init_chatbot()

            # Cargar documentos si se especifica
            if load_docs:
                cli.console.print(f"📂 [bold]Cargando documentos desde {load_docs}[/bold]")
                cli.load_directory(load_docs, force_update=force_update)
                cli.console.print()

            # Recargar documentos (equivalente a load_docs con force_update=True)
            if reload_docs:
                cli.console.print(f"🔄 [bold]Recargando documentos desde {reload_docs}[/bold]")
                cli.load_directory(reload_docs, force_update=True)
                cli.console.print()

            # Modo consulta única
            if single_query:
                cli.console.print(f"[bold blue]Consulta:[/bold blue] {single_query}")
                response = cli.process_query(single_query)
                cli.console.print(f"[bold green]Respuesta:[/bold green]")
                cli.console.print(Markdown(response))
                return

            # Modo batch (archivo de consultas)
            if batch_file:
                if not os.path.exists(batch_file):
                    cli.console.print(f"❌ [red]Archivo no encontrado: {batch_file}[/red]")
                    return

                with open(batch_file, 'r', encoding='utf-8') as f:
                    queries = [line.strip() for line in f if line.strip()]

                cli.console.print(f"📄 [bold]Procesando {len(queries)} consultas desde {batch_file}[/bold]")
                cli.console.print()

                for i, query in enumerate(queries, 1):
                    cli.console.print(f"[bold blue]{i}. Consulta:[/bold blue] {query}")
                    response = cli.process_query(query)
                    cli.console.print(f"[bold green]Respuesta:[/bold green]")
                    cli.console.print(Markdown(response))
                    cli.console.print("─" * 50)
                    cli.console.print()
                return

            # Modo interactivo (por defecto)
            cli.chat_loop()

            # Exportar conversación si se especifica
            if export_on_exit and cli.conversation_history:
                cli.export_conversation(export_on_exit)

        except KeyboardInterrupt:
            cli.console.print("\n[yellow]Programa interrumpido por el usuario[/yellow]")
        except Exception as e:
            cli.console.print(f"❌ [red]Error fatal: {str(e)}[/red]")
            sys.exit(1)

    else:
        # Modo básico sin CLI avanzada
        print("🤖 RAG CHATBOT - MODO BÁSICO")
        print("=" * 40)

        try:
            chatbot = RAGChatbot()
            print("✅ Sistema iniciado")

            # Cargar documentos si se especifica
            if load_docs:
                print(f"📂 Cargando documentos desde {load_docs}")
                if os.path.exists(load_docs):
                    json_files = [os.path.join(load_docs, f) for f in os.listdir(load_docs) if f.endswith('.json')]
                    chatbot.add_documents_from_json(json_files, force_update=force_update)
                else:
                    print(f"❌ Directorio no encontrado: {load_docs}")

            # Consulta única
            if single_query:
                print(f"❓ Consulta: {single_query}")
                response = chatbot.query(single_query)
                print(f"🤖 Respuesta: {response}")
                return

            # Chat básico
            print("\n💬 Modo chat básico (escribe 'quit' para salir)")
            while True:
                try:
                    user_input = input("\nTú: ").strip()
                    if user_input.lower() in ['quit', 'exit', 'q']:
                        break
                    if user_input:
                        response = chatbot.query(user_input)
                        print(f"🤖: {response}")
                except KeyboardInterrupt:
                    break
                except Exception as e:
                    print(f"❌ Error: {e}")

            print("👋 ¡Hasta luego!")

        except Exception as e:
            print(f"❌ Error fatal: {e}")
            sys.exit(1)


# Ejemplo de uso desde línea de comandos
if __name__ == "__main__":
    if CLI_AVAILABLE:
        main_cli()
    else:
        main_cli()