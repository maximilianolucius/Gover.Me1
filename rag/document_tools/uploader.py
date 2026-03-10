#!/usr/bin/env python3
"""Upload JSON documents to PostgreSQL and Milvus for RAG retrieval.

Recursively scans a directory for JSON files, chunks the text content,
generates CPU-only sentence-transformer embeddings, and stores them in
PostgreSQL (metadata + chunks) and Milvus (vector index).
"""

import os
import json
import hashlib
import argparse
from typing import List, Dict, Optional
from pathlib import Path
from dotenv import load_dotenv
import psycopg2
from langchain.text_splitter import RecursiveCharacterTextSplitter
from pymilvus import FieldSchema, CollectionSchema, DataType, Collection, utility, connections
from sentence_transformers import SentenceTransformer

# Cargar variables de entorno
load_dotenv()


class LocalEmbeddings:
    """Embeddings locales usando sentence-transformers - SOLO CPU"""

    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        print(f"🔄 Cargando modelo {model_name} en CPU...")
        self.model = SentenceTransformer(model_name, device='cpu')
        self.dimension = self.model.get_sentence_embedding_dimension()
        print(f"✅ Modelo cargado en CPU (dim={self.dimension})")

    def embed_query(self, text: str) -> List[float]:
        """Return the normalized embedding for a single text."""
        vec = self.model.encode(text, normalize_embeddings=True)
        return vec.tolist()

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Return normalized embeddings for a batch of texts."""
        vecs = self.model.encode(texts, normalize_embeddings=True, batch_size=32)  # Batch más pequeño para CPU
        return [v.tolist() for v in vecs]


class DatabaseManager:
    """Gestor simplificado de PostgreSQL"""

    def __init__(self):
        self.connection = psycopg2.connect(
            host=os.getenv("POSTGRES_HOST", "localhost"),
            database=os.getenv("POSTGRES_DB", "rag_chatbot"),
            user=os.getenv("POSTGRES_USER", "postgres"),
            password=os.getenv("POSTGRES_PASSWORD", "password"),
            port=os.getenv("POSTGRES_PORT", "5432")
        )
        self._create_tables()

    def _create_tables(self):
        """Crear tablas necesarias"""
        cursor = self.connection.cursor()

        cursor.execute("""
            CREATE TABLE IF NOT EXISTS documents (
                id SERIAL PRIMARY KEY,
                titulo VARCHAR(500),
                autor VARCHAR(200),
                fecha TIMESTAMP,
                url_original TEXT,
                formato_detectado VARCHAR(100),
                contenido_completo TEXT,
                content_hash VARCHAR(32),
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)

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
        """Verificar si documento existe por hash de contenido"""
        cursor = self.connection.cursor()
        content = doc_data.get('contenido_completo', '')
        content_hash = hashlib.md5(content.encode('utf-8')).hexdigest()

        cursor.execute("SELECT id FROM documents WHERE content_hash = %s", (content_hash,))
        result = cursor.fetchone()
        cursor.close()
        return result[0] if result else None

    def insert_document(self, doc_data: Dict) -> int:
        """Insertar nuevo documento"""
        cursor = self.connection.cursor()
        content = doc_data.get('contenido_completo', '')
        content_hash = hashlib.md5(content.encode('utf-8')).hexdigest()

        cursor.execute("""
            INSERT INTO documents (titulo, autor, fecha, url_original, formato_detectado, contenido_completo, content_hash)
            VALUES (%s, %s, %s, %s, %s, %s, %s)
            RETURNING id
        """, (
            doc_data.get('titulo'),
            doc_data.get('autor'),
            doc_data.get('fecha'),
            doc_data.get('url_original'),
            doc_data.get('formato_detectado'),
            content,
            content_hash
        ))

        doc_id = cursor.fetchone()[0]
        self.connection.commit()
        cursor.close()
        return doc_id

    def insert_chunk(self, document_id: int, chunk_text: str, chunk_index: int, vector_id: str):
        """Insertar chunk de texto"""
        cursor = self.connection.cursor()
        cursor.execute("""
            INSERT INTO text_chunks (document_id, chunk_text, chunk_index, vector_id)
            VALUES (%s, %s, %s, %s)
        """, (document_id, chunk_text, chunk_index, vector_id))
        self.connection.commit()
        cursor.close()


class MilvusManager:
    """Gestor simplificado de Milvus"""

    def __init__(self, collection_name: str = "document_vectors", dim: int = 384):
        self.collection_name = collection_name
        self._connect()
        self._setup_collection(dim)

    def _connect(self):
        """Conectar a Milvus"""
        connections.connect(
            host=os.getenv("MILVUS_HOST", "localhost"),
            port=os.getenv("MILVUS_PORT", "19530")
        )

    def _setup_collection(self, dim: int):
        """Configurar colección de Milvus"""
        if utility.has_collection(self.collection_name):
            self.collection = Collection(self.collection_name)
        else:
            fields = [
                FieldSchema(name="id", dtype=DataType.VARCHAR, max_length=100, is_primary=True),
                FieldSchema(name="vector", dtype=DataType.FLOAT_VECTOR, dim=dim),
                FieldSchema(name="document_id", dtype=DataType.INT64),
                FieldSchema(name="chunk_index", dtype=DataType.INT64),
                FieldSchema(name="text", dtype=DataType.VARCHAR, max_length=10000),
            ]
            schema = CollectionSchema(fields, "Vectores de documentos RAG")
            self.collection = Collection(name=self.collection_name, schema=schema)

            # Crear índice
            index_params = {
                "metric_type": "COSINE",
                "index_type": "HNSW",
                "params": {"M": 16, "efConstruction": 100},
            }
            self.collection.create_index("vector", index_params)

        self.collection.load()

    def insert_vectors(self, vectors_data: List[Dict]):
        """Insertar vectores en Milvus"""
        if not vectors_data:
            return

        ids = [item["id"] for item in vectors_data]
        vectors = [item["vector"] for item in vectors_data]
        document_ids = [item["document_id"] for item in vectors_data]
        chunk_indices = [item["chunk_index"] for item in vectors_data]
        texts = [item["text"] for item in vectors_data]

        data = [ids, vectors, document_ids, chunk_indices, texts]
        self.collection.insert(data)
        self.collection.flush()


class DocumentUploader:
    """Clase principal para subir documentos"""

    def __init__(self):
        print("🔌 Conectando a bases de datos...")
        self.db_manager = DatabaseManager()
        self.milvus_manager = MilvusManager()
        self.embeddings = LocalEmbeddings()  # Forzará uso de CPU
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len
        )
        print("✅ Sistema inicializado (CPU mode)")

    def find_json_files(self, directory: str) -> List[str]:
        """Buscar recursivamente archivos JSON en directorio"""
        json_files = []
        for root, dirs, files in os.walk(directory):
            for file in files:
                if file.endswith('.json'):
                    json_files.append(os.path.join(root, file))
        return json_files

    def process_document(self, doc_data: Dict) -> Optional[int]:
        """Procesar un documento y generar vectores"""
        # Verificar si ya existe
        if self.db_manager.document_exists(doc_data):
            return None  # Ya existe, saltar

        # Insertar documento
        doc_id = self.db_manager.insert_document(doc_data)

        # Procesar contenido
        content = doc_data.get('contenido_completo', '').strip()
        if not content:
            return doc_id  # Sin contenido para vectorizar

        # Dividir en chunks
        chunks = self.text_splitter.split_text(content)
        if not chunks:
            return doc_id

        # Generar vectores (en CPU)
        vectors_data = []
        print(f"   🔄 Generando {len(chunks)} embeddings...", end="")

        for i, chunk in enumerate(chunks):
            if not chunk.strip():
                continue

            vector = self.embeddings.embed_query(chunk)
            vector_id = f"{doc_id}_{i}"

            # Guardar chunk en PostgreSQL
            self.db_manager.insert_chunk(doc_id, chunk, i, vector_id)

            # Preparar para Milvus
            vectors_data.append({
                "id": vector_id,
                "vector": vector,
                "document_id": doc_id,
                "chunk_index": i,
                "text": chunk[:9000]  # Límite de Milvus
            })

        # Insertar vectores en Milvus
        if vectors_data:
            self.milvus_manager.insert_vectors(vectors_data)
            print(f" {len(vectors_data)} vectores")

        return doc_id

    def upload_from_directory(self, directory: str):
        """Subir todos los documentos JSON de un directorio recursivamente"""
        if not os.path.exists(directory):
            print(f"❌ Directorio no encontrado: {directory}")
            return

        # Encontrar archivos JSON
        json_files = self.find_json_files(directory)
        if not json_files:
            print(f"⚠️ No se encontraron archivos JSON en: {directory}")
            return

        print(f"📄 Encontrados {len(json_files)} archivos JSON")

        # Procesar archivos
        loaded = 0
        skipped = 0
        errors = 0

        for i, file_path in enumerate(json_files, 1):
            try:
                print(f"📄 [{i}/{len(json_files)}] {os.path.basename(file_path)}", end="")

                with open(file_path, 'r', encoding='utf-8') as f:
                    doc_data = json.load(f)

                doc_id = self.process_document(doc_data)

                if doc_id is None:
                    print(" ⭐ Ya existe")
                    skipped += 1
                else:
                    print(f" ✅ (ID: {doc_id})")
                    loaded += 1

            except Exception as e:
                print(f" ❌ Error: {str(e)}")
                errors += 1

        # Resumen
        print(f"\n📊 Resumen:")
        print(f"   ✅ Cargados: {loaded}")
        print(f"   ⭐ Ya existían: {skipped}")
        print(f"   ❌ Errores: {errors}")


def main():
    """Función principal"""
    parser = argparse.ArgumentParser(
        description="RAG Document Uploader - Subir documentos JSON recursivamente (CPU only)")
    parser.add_argument("directory", help="Directorio con documentos JSON")

    args = parser.parse_args()

    try:
        uploader = DocumentUploader()
        uploader.upload_from_directory(args.directory)
        print("🎉 Proceso completado")

    except KeyboardInterrupt:
        print("\n⚠️ Proceso interrumpido por el usuario")
    except Exception as e:
        print(f"❌ Error fatal: {e}")


if __name__ == "__main__":
    main()