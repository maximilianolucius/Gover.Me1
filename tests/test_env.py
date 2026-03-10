"""Environment setup verification for vLLM embeddings and Milvus connectivity.

Tests that required environment variables are set, the vLLM embedding endpoint
is reachable, and the Milvus collection can be queried.
"""

import os
from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings

# Cargar variables de entorno del archivo .env (debe estar en la misma carpeta o especificar ruta)
load_dotenv()

def create_vllm_embeddings():
    """Create an OpenAIEmbeddings instance pointing at the local vLLM server."""
    api_key = os.getenv("VLLM_API_KEY")
    base_url = os.getenv("VLLM_BASE_URL")

    if not api_key or not base_url:
        raise ValueError("Debes definir VLLM_API_KEY y VLLM_BASE_URL en tu entorno o archivo .env")

    embedding_model = "text-embedding-3-small"  # O el que uses para vLLM

    embeddings = OpenAIEmbeddings(
        model=embedding_model,
        api_key=api_key,
        base_url=base_url  # Endpoint vLLM
    )

    return embeddings

if __name__ == "__main__":
    embeddings = create_vllm_embeddings()
    sample_text = "Prueba de embeddings usando vLLM con variables de entorno cargadas"

    print("Generando embedding...")
    vector = embeddings.embed_query(sample_text)
    print(f"Vector de embedding (dim={len(vector)}):")
    print(vector)

from pymilvus import connections, Collection

connections.connect("default", host="127.0.0.1", port="19530")
coll = Collection("document_vectors")         # tu nombre
print(coll.num_entities)                      # → int con el total
