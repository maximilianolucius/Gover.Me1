"""Tourism question-answering pipeline for Andalusia data.

Builds a FAISS vector index over tourism records extracted from Excel files,
caches embeddings to disk, and answers natural-language queries via a vLLM
server using retrieval-augmented generation.
"""

# SISTEMA CLI COMPLETO PARA DATOS TURÍSTICOS + vLLM - CACHE PRIORITARIO
import pandas as pd
import json
import numpy as np
import hashlib
from pathlib import Path
from sentence_transformers import SentenceTransformer
import faiss
from langchain_openai import ChatOpenAI
from langchain.schema import HumanMessage
import os
import argparse
from datetime import datetime
from typing import List, Dict, Any, Tuple
import torch
from tourism.extractor import UltimateTourismExtractor


# Force CPU usage for all PyTorch operations
os.environ['CUDA_VISIBLE_DEVICES'] = ''
torch.set_default_tensor_type('torch.FloatTensor')

# If you want to be extra sure, add this too:
os.environ['TOKENIZERS_PARALLELISM'] = 'false'


class TourismDataProcessor:
    """Extracts and normalises tourism records from Excel spreadsheets."""

    def __init__(self):
        self.all_data = []

    def extract_all_excel_data(excel_files: List[str]) -> List[Dict]:
        """VERSIÓN MEJORADA - Reemplaza tu función existente"""
        extractor = UltimateTourismExtractor()
        all_records = []

        for excel_file in excel_files:
            records = extractor.extract_from_excel(excel_file)
            all_records.extend(records)
            print(f"✅ {Path(excel_file).name}: {len(records)} registros")

        print(f"📊 TOTAL RAG: {len(all_records)} registros optimizados")
        return all_records


    def _get_fecha_fuente(self, periodo):
        """Return a date string (DD/MM/YYYY) derived from the given period name."""
        if not periodo:
            return "01/01/2025"

        # Mapeo de meses
        meses = {
            'enero': '01', 'febrero': '02', 'marzo': '03', 'abril': '04',
            'mayo': '05', 'junio': '06', 'julio': '07', 'agosto': '08',
            'septiembre': '09', 'octubre': '10', 'noviembre': '11', 'diciembre': '12'
        }

        periodo_lower = periodo.lower()
        for mes_nombre, mes_num in meses.items():
            if mes_nombre in periodo_lower:
                return f"01/{mes_num}/2025"

        # Si encuentra un número de mes
        for i in range(1, 13):
            if f"{i:02d}" in periodo or str(i) in periodo:
                return f"01/{i:02d}/2025"

        return "01/01/2025"

    def _detect_file_type(self, file_path, df):
        """Map the Excel filename prefix (01-21) to a human-readable tourism type."""
        filename = Path(file_path).stem.lower()

        # Mapeo directo para los 21 archivos específicos
        file_types = {
            '01_total': 'Total Turistas',
            '02_espanoles': 'Turistas Españoles',
            '03_andaluces': 'Turistas Andaluces',
            '04_resto': 'Resto España',
            '05_extranjeros': 'Turistas Extranjeros',
            '06_britanicos': 'Turistas Británicos',
            '07_alemanes': 'Turistas Alemanes',
            '08_otros': 'Otros Mercados',
            '09_litoral': 'Turismo Litoral',
            '10_interior': 'Turismo Interior',
            '11_cruceros': 'Turismo Cruceros',
            '12_ciudad': 'Turismo Ciudad',
            '13_cultural': 'Turismo Cultural',
            '14_almeria': 'Provincia Almería',
            '15_cadiz': 'Provincia Cádiz',
            '16_cordoba': 'Provincia Córdoba',
            '17_granada': 'Provincia Granada',
            '18_huelva': 'Provincia Huelva',
            '19_jaen': 'Provincia Jaén',
            '20_malaga': 'Provincia Málaga',
            '21_sevilla': 'Provincia Sevilla'
        }

        for prefix, type_name in file_types.items():
            if filename.startswith(prefix):
                return type_name

        return filename.replace('_', ' ').title()

    def _is_data_row(self, row_clean):
        """Return True if the cleaned row looks like a data row with an indicator and numeric fields."""
        if len(row_clean) < 2:
            return False

        # Primer campo debe ser descriptivo (indicador)
        if len(row_clean[0]) < 5:
            return False

        # Al menos un campo debe tener números, porcentajes o fechas
        has_data = False
        for cell in row_clean[1:]:
            cell_str = str(cell).replace(',', '').replace(' ', '')
            if (cell_str.replace('.', '').replace('-', '').isdigit() or
                    '%' in cell_str or
                    any(month in cell_str.lower() for month in
                        ['enero', 'febrero', 'marzo', 'abril', 'mayo', 'junio']) or
                    any(year in cell_str for year in ['2024', '2025'])):
                has_data = True
                break

        return has_data


class TourismRAG:
    """FAISS-backed vector store for tourism data with disk-cached embeddings."""

    def __init__(self, data_list, cache_dir="./cache", force_regenerate=False, use_cache_only=False):
        print("🔧 Inicializando sistema RAG (CPU ONLY)...")
        self.data = data_list
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True)
        self.force_regenerate = force_regenerate
        self.use_cache_only = use_cache_only

        # FORCE CPU for SentenceTransformers
        self.model = SentenceTransformer(
            'paraphrase-multilingual-MiniLM-L12-v2',
            device='cpu'  # EXPLICITLY force CPU
        )
        self.setup_vector_store()

    def _get_data_hash(self):
        """Return a short MD5 hash of the current data for cache key purposes."""
        import hashlib
        data_str = json.dumps(self.data, sort_keys=True, ensure_ascii=False)
        return hashlib.md5(data_str.encode()).hexdigest()[:8]

    def _find_any_compatible_cache(self):
        """Locate the most recent compatible cache files in the cache directory."""
        try:
            embedding_files = list(self.cache_dir.glob("embeddings_*.npy"))
            if not embedding_files:
                return None, None, None, None

            # Usar el cache más reciente
            latest_file = max(embedding_files, key=lambda f: f.stat().st_mtime)
            hash_part = latest_file.stem.split('_')[1]

            index_file = self.cache_dir / f"faiss_index_{hash_part}.index"
            texts_file = self.cache_dir / f"texts_{hash_part}.json"
            data_file = self.cache_dir / f"data_{hash_part}.json"  # NUEVO: buscar datos originales

            if index_file.exists() and texts_file.exists():
                return latest_file, index_file, texts_file, data_file

        except Exception as e:
            print(f"⚠️ Error buscando cache compatible: {e}")

        return None, None, None, None

    def setup_vector_store(self):
        """Build or load the FAISS index, prioritising cached embeddings."""
        data_hash = self._get_data_hash()
        data_hash = '3b7a0a5f'
        embeddings_file = self.cache_dir / f"embeddings_{data_hash}.npy"
        index_file = self.cache_dir / f"faiss_index_{data_hash}.index"
        texts_file = self.cache_dir / f"texts_{data_hash}.json"
        data_file = self.cache_dir / f"data_{data_hash}.json"  # NUEVO: archivo de datos

        # MODO 1: Solo usar cache (no regenerar nunca)
        if self.use_cache_only:
            if all(f.exists() for f in [embeddings_file, index_file, texts_file]):
                success = self._load_cache(embeddings_file, index_file, texts_file, data_file)
                if success:
                    return

            # Buscar cualquier cache compatible
            alt_emb, alt_idx, alt_txt, alt_data = self._find_any_compatible_cache()
            if alt_emb:
                print(f"🔄 Usando cache alternativo: {alt_emb.name}")
                success = self._load_cache(alt_emb, alt_idx, alt_txt, alt_data)
                if success:
                    return

            raise Exception("❌ Modo cache-only activado pero no se encontró cache válido")

        # MODO 2: Forzar regeneración
        if self.force_regenerate:
            print("🔄 Forzando regeneración de embeddings...")
            self._generate_embeddings_and_cache(data_hash)
            return

        # MODO 3: PRIORIDAD AL CACHE (comportamiento por defecto mejorado)
        print(f"🔍 Buscando cache con hash: {data_hash}")

        # Intentar cargar cache específico
        if all(f.exists() for f in [embeddings_file, index_file, texts_file]):
            success = self._load_cache(embeddings_file, index_file, texts_file, data_file)
            if success:
                return

        # Buscar cache compatible con datos similares
        print("🔍 Buscando cache compatible...")
        alt_emb, alt_idx, alt_txt, alt_data = self._find_any_compatible_cache()
        if alt_emb:
            print(f"🔄 Encontrado cache alternativo: {alt_emb.name}")
            success = self._load_cache(alt_emb, alt_idx, alt_txt, alt_data)
            if success:
                print("ℹ️ Usando cache de datos similares. Para cache exacto, use --clear-cache")
                return

        # Solo regenerar si no hay cache disponible
        print("🔧 No se encontró cache válido, generando embeddings...")
        self._generate_embeddings_and_cache(data_hash)

    def _load_cache(self, embeddings_file, index_file, texts_file, data_file=None):
        """Load cached embeddings, FAISS index, and optionally the original data."""
        try:
            print(f"📂 Cargando cache: {embeddings_file.name}")

            # Cargar embeddings
            self.embeddings = np.load(embeddings_file)

            # Cargar índice FAISS
            self.index = faiss.read_index(str(index_file))

            # Verificar consistencia básica con textos
            with open(texts_file, 'r', encoding='utf-8') as f:
                cached_texts = json.load(f)

            # NUEVO: Cargar datos originales si están disponibles
            if data_file and data_file.exists():
                try:
                    with open(data_file, 'r', encoding='utf-8') as f:
                        cached_data = json.load(f)

                    # Usar datos originales del cache
                    self.data = cached_data
                    print(f"✅ Cache completo cargado: {len(cached_data)} registros + embeddings")
                    return True

                except Exception as e:
                    print(f"⚠️ Error cargando datos del cache: {e}, usando datos actuales")

            # Verificación flexible para compatibilidad con cache sin datos
            if abs(len(cached_texts) - len(self.data)) <= 10:  # Tolerancia de 10 registros
                print(f"✅ Cache de embeddings cargado: {len(cached_texts)} embeddings")
                if len(cached_texts) != len(self.data):
                    print(f"ℹ️ Diferencia de {abs(len(cached_texts) - len(self.data))} registros (tolerado)")
                return True
            else:
                print(f"⚠️ Cache inconsistente: {len(cached_texts)} vs {len(self.data)} registros")
                return False

        except Exception as e:
            print(f"⚠️ Error cargando cache: {e}")
            return False

    def _generate_embeddings_and_cache(self, data_hash):
        """Encode all data texts, build the FAISS index, and persist to cache."""
        print(f"🏗️ Generando embeddings para {len(self.data)} textos...")

        # Crear textos descriptivos para embedding
        texts = []
        for item in self.data:
            text = item.get('texto_completo',
                            f"{item.get('tipo_turismo', '')} {item.get('periodo', '')} {item.get('valor', 0)}")
            # text = f"{item['tipo_turismo']} {item['indicador']} {item['periodo']} {item['valor']} {item['variacion']}"
            texts.append(text)

        # Generar embeddings
        self.embeddings = self.model.encode(texts, show_progress_bar=True)

        # Crear índice FAISS
        dimension = self.embeddings.shape[1]
        self.index = faiss.IndexFlatIP(dimension)
        self.index.add(self.embeddings.astype('float32'))

        # Guardar en cache
        self._save_cache(data_hash, texts)
        print("✅ Índice vectorial creado y cacheado")

    def _save_cache(self, data_hash, texts):
        """Persist embeddings, FAISS index, texts, and source data to disk."""
        try:
            embeddings_file = self.cache_dir / f"embeddings_{data_hash}.npy"
            index_file = self.cache_dir / f"faiss_index_{data_hash}.index"
            texts_file = self.cache_dir / f"texts_{data_hash}.json"
            data_file = self.cache_dir / f"data_{data_hash}.json"  # NUEVO: guardar datos originales

            print(f"💾 Guardando cache completo: {embeddings_file.name}")

            # Guardar archivos
            np.save(embeddings_file, self.embeddings)
            faiss.write_index(self.index, str(index_file))

            with open(texts_file, 'w', encoding='utf-8') as f:
                json.dump(texts, f, ensure_ascii=False, indent=2)

            # NUEVO: Guardar datos originales para evitar reprocesamiento
            with open(data_file, 'w', encoding='utf-8') as f:
                json.dump(self.data, f, ensure_ascii=False, indent=2)

            # Limpiar caches antiguos (mantener los 3 más recientes)
            self._cleanup_old_caches(data_hash, keep_recent=3)
            print(f"✅ Cache completo guardado (embeddings + datos originales)")

        except Exception as e:
            print(f"⚠️ Error guardando cache: {e}")

    def _cleanup_old_caches(self, current_hash, keep_recent=3):
        """Remove stale cache files, keeping only the *keep_recent* most recent sets."""
        try:
            # Obtener todos los archivos de embeddings con timestamps
            embedding_files = []
            for file in self.cache_dir.glob("embeddings_*.npy"):
                if current_hash not in file.name:
                    embedding_files.append((file.stat().st_mtime, file))

            # Ordenar por fecha (más reciente primero) y mantener solo los primeros
            embedding_files.sort(reverse=True)
            files_to_keep = set()

            for i, (_, file) in enumerate(embedding_files[:keep_recent]):
                hash_part = file.stem.split('_')[1]
                files_to_keep.add(f"embeddings_{hash_part}.npy")
                files_to_keep.add(f"faiss_index_{hash_part}.index")
                files_to_keep.add(f"texts_{hash_part}.json")
                files_to_keep.add(f"data_{hash_part}.json")  # NUEVO: incluir datos originales

            # Eliminar archivos antiguos
            removed_count = 0
            patterns = ["embeddings_*.npy", "faiss_index_*.index", "texts_*.json",
                        "data_*.json"]  # NUEVO: incluir datos
            for pattern in patterns:
                for file in self.cache_dir.glob(pattern):
                    if (current_hash not in file.name and
                            file.name not in files_to_keep):
                        file.unlink()
                        removed_count += 1

            if removed_count > 0:
                print(f"🧹 Limpiados {removed_count} archivos de cache antiguos")

        except Exception as e:
            print(f"⚠️ Error limpiando cache: {e}")

    def search(self, query, k=5):
        """Return the top-*k* most relevant data records for the given query."""
        query_embedding = self.model.encode([query])
        scores, indices = self.index.search(query_embedding.astype('float32'), k)

        results = []
        for i, idx in enumerate(indices[0]):
            results.append({
                'data': self.data[idx],
                'score': float(scores[0][i])
            })

        return results


class TourismQA:
    """End-to-end tourism QA system combining TourismRAG retrieval with vLLM generation."""

    def __init__(self, data_list, cache_dir="./cache", force_regenerate=False, use_cache_only=False):
        print("🤖 Conectando con vLLM server...")

        # Configuración vLLM server
        base_url = os.getenv("VLLM_BASE_URL", "http://172.24.250.17:8000/v1")
        api_key = os.getenv("VLLM_API_KEY", "NoLoNecesitas")
        chat_model = os.getenv("VLLM_MODEL", "gemma-3-12b-it")

        self.rag = TourismRAG(data_list, cache_dir=cache_dir,
                              force_regenerate=force_regenerate,
                              use_cache_only=use_cache_only)
        self.llm = ChatOpenAI(
            model=chat_model,
            base_url=base_url,
            api_key=api_key,
            temperature=0.2
        )

        print(f"✅ Conectado a {base_url} con modelo {chat_model}")

    def _build_context_and_sources(self, question: str, top_k: int = 12):
        """Retrieve relevant records and build the prompt context string and source list."""

        # Búsqueda directa usando el RAG existente
        results = self.rag.search(question, k=top_k)

        # Buscar también por palabras clave específicas
        keywords = self._extract_keywords(question)
        if keywords:
            keyword_query = " ".join(keywords)
            keyword_results = self.rag.search(keyword_query, k=top_k // 2)
            # Combinar resultados únicos
            seen_indices = {r['data'].get('archivo_origen', '') + str(r['data'].get('valor', '')) for r in results}
            for kr in keyword_results:
                key = kr['data'].get('archivo_origen', '') + str(kr['data'].get('valor', ''))
                if key not in seen_indices:
                    results.append(kr)
                    seen_indices.add(key)

        # Construir contexto y fuentes
        context_parts = []
        sources = []

        for i, result in enumerate(results[:top_k]):
            data = result['data']
            texto = data.get('texto_completo', '')

            if texto:
                context_parts.append(texto)

                # Crear fuente estructurada
                sources.append({
                    'fuente': data.get('fuente', f"Nexus Andalucía - {data.get('tipo_turismo', '').title()}"),
                    'publish_date': f"01/{data.get('mes', '01')}/2025",
                    'ref': texto[:200] + "..." if len(texto) > 200 else texto,
                    'archivo_origen': data.get('archivo_origen', ''),
                    'tipo_dato': data.get('tipo_registro', '')
                })

        return "\n\n".join(context_parts), sources


    def _extract_keywords(self, question: str) -> List[str]:
        """Extract domain-specific keywords (nationalities, metrics, months) from the question."""
        keywords = []
        q_lower = question.lower()

        # Nacionalidades
        if any(x in q_lower for x in ['británic', 'britanic']):
            keywords.append('británicos')
        if any(x in q_lower for x in ['alemán', 'aleman']):
            keywords.append('alemanes')

        # Métricas
        if any(x in q_lower for x in ['turistas', 'viajeros']):
            keywords.append('turistas')
        if 'pernoctaciones' in q_lower:
            keywords.append('pernoctaciones')

        # Meses
        meses = ['enero', 'febrero', 'marzo', 'abril', 'mayo', 'junio']
        for mes in meses:
            if mes in q_lower:
                keywords.append(mes)

        # Años
        import re
        years = re.findall(r'20\d{2}', question)
        keywords.extend(years)

        return keywords

    def answer(self, question):
        """Answer a tourism question using RAG retrieval and the vLLM server."""
        print(f"\n❓ Pregunta: {question}")

        # Construir contexto y fuentes usando la nueva función
        context, fuentes = self._build_context_and_sources(question, top_k=6)

        # Si no hay contexto, respuesta de error
        if not context:
            return {
                'question': question,
                'answer': "No se encontró información suficiente para responder con las fuentes disponibles.",
                'fuentes': [],
                'context_used': "",
                'code': 1,
                'error': "Sin datos relevantes encontrados"
            }

        # Crear prompt mejorado
        prompt = f"""Eres un asistente experto en turismo de Andalucía. Responde basándote ÚNICAMENTE en estos datos de Nexus Andalucía:

DATOS DISPONIBLES:
{context}

PREGUNTA: {question}

INSTRUCCIONES:
- Usa solo los datos proporcionados de Nexus Andalucía
- Sé específico con números y porcentajes exactos
- Menciona el período de los datos claramente
- Si no tienes información suficiente, dilo claramente
- Responde en español de forma clara y concisa

RESPUESTA:"""

        # Generar respuesta
        print("🔄 Generando respuesta...")
        try:
            message = HumanMessage(content=prompt)
            response = self.llm.invoke([message])
            answer = response.content.strip()

            return {
                'question': question,
                'answer': answer,
                'fuentes': fuentes,
                'context_used': context,
                'code': 0,
                'error': ""
            }

        except Exception as e:
            return {
                'question': question,
                'answer': f"Error al generar respuesta: {str(e)}",
                'fuentes': fuentes,
                'context_used': context,
                'code': 2,
                'error': str(e)
            }

    def query_json(self, question: str) -> Dict[str, Any]:
        """Return an answer in the JSON format expected by machiavelli_query.py."""
        result = self.answer(question)

        return {
            "code": result.get('code', 0),
            "data": {
                "interim_answer": "",
                "final_answer": result['answer'],
                "fuentes": result['fuentes']
            },
            "error": result.get('error', ""),
            "is_done": True
        }


def get_excel_files_from_dir(source_dir: str) -> List[str]:
    """Return a list of Excel file paths (.xlsx, .xls) found in *source_dir*."""
    excel_dir = Path(source_dir)
    if not excel_dir.exists():
        print(f"❌ El directorio {source_dir} no existe")
        return []

    excel_files = list(excel_dir.glob("*.xlsx")) + list(excel_dir.glob("*.xls"))
    excel_files = [str(f) for f in excel_files]

    print(f"📁 Encontrados {len(excel_files)} archivos Excel en {source_dir}")
    for f in excel_files[:5]:  # Mostrar primeros 5
        print(f"  - {Path(f).name}")
    if len(excel_files) > 5:
        print(f"  ... y {len(excel_files) - 5} más")

    return excel_files


def has_valid_cache(cache_dir):
    """Return True if a complete cache (embeddings + index + texts + data) exists."""
    cache_path = Path(cache_dir)
    if not cache_path.exists():
        return False

    cache_files = list(cache_path.glob("embeddings_*.npy"))
    if not cache_files:
        return False

    # Verificar que existan TODOS los archivos necesarios (incluyendo datos)
    latest_cache = max(cache_files, key=lambda f: f.stat().st_mtime)
    hash_part = latest_cache.stem.split('_')[1]

    index_file = cache_path / f"faiss_index_{hash_part}.index"
    texts_file = cache_path / f"texts_{hash_part}.json"
    data_file = cache_path / f"data_{hash_part}.json"  # NUEVO: verificar datos originales

    return index_file.exists() and texts_file.exists() and data_file.exists()


def load_cached_data(cache_dir):
    """Load the original data records from the most recent cache file."""
    cache_path = Path(cache_dir)
    cache_files = list(cache_path.glob("embeddings_*.npy"))
    latest_cache = max(cache_files, key=lambda f: f.stat().st_mtime)
    hash_part = latest_cache.stem.split('_')[1]

    data_file = cache_path / f"data_{hash_part}.json"

    if data_file.exists():
        print(f"📂 Cargando datos originales desde cache: {data_file.name}")
        with open(data_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        print(f"✅ Cargados {len(data)} registros desde cache")
        return data
    else:
        print("⚠️ Cache de datos no encontrado, generando datos dummy")
        return []  # Fallback para compatibilidad


def main():
    """CLI entry point: parse arguments, load data or cache, and start interactive QA."""
    parser = argparse.ArgumentParser(description='Sistema Q&A Turismo Andalucía - Cache Prioritario')
    parser.add_argument('--source-dir', default='./nexus/Excels_Limpios/',
                        help='Directorio con archivos Excel (default: ./nexus/Excels_Limpios/)')
    parser.add_argument('--cache-dir', default='./cache',
                        help='Directorio para cache de embeddings (default: ./cache)')
    parser.add_argument('--files', nargs='+', default=None,
                        help='Archivos Excel específicos a procesar (sobrescribe --source-dir)')
    parser.add_argument('--save-data', help='Guardar datos procesados en JSON')
    parser.add_argument('--load-data', help='Cargar datos desde JSON')
    parser.add_argument('--query', '-q', help='Hacer una consulta directa y salir')
    parser.add_argument('--json-output', action='store_true',
                        help='Salida en formato JSON (compatible con machiavelli_query)')

    # OPCIONES DE CACHE MEJORADAS
    parser.add_argument('--clear-cache', action='store_true',
                        help='Limpiar cache y regenerar embeddings')
    parser.add_argument('--force-regenerate', action='store_true',
                        help='Forzar regeneración de embeddings (mantiene cache anterior)')
    parser.add_argument('--cache-only', action='store_true',
                        help='Solo usar cache existente (error si no existe)')
    parser.add_argument('--no-cache', action='store_true',
                        help='No usar cache (regenerar siempre)')
    parser.add_argument('--force-excel', action='store_true',
                        help='Forzar procesamiento de Excel aunque exista cache')

    args = parser.parse_args()

    print("🏖️ SISTEMA Q&A TURISMO ANDALUCÍA - CACHE PRIORITARIO")
    print("=" * 60)

    # Gestión de opciones de cache
    use_cache_only = args.cache_only
    force_regenerate = args.force_regenerate or args.no_cache
    force_excel = args.force_excel

    # Limpiar cache si se solicita
    if args.clear_cache:
        cache_path = Path(args.cache_dir)
        if cache_path.exists():
            print(f"🗑️ Limpiando cache completo en {args.cache_dir}")
            removed = 0
            patterns = ["embeddings_*.npy", "faiss_index_*.index", "texts_*.json",
                        "data_*.json"]  # NUEVO: incluir datos
            for pattern in patterns:
                for file in cache_path.glob(pattern):
                    file.unlink()
                    removed += 1
            print(f"✅ Cache limpiado: {removed} archivos eliminados")
            force_regenerate = True
            force_excel = True

    # Mostrar información de cache ANTES de procesar
    cache_path = Path(args.cache_dir)
    has_cache = has_valid_cache(args.cache_dir)

    if has_cache:
        cache_files = list(cache_path.glob("embeddings_*.npy"))
        latest_cache = max(cache_files, key=lambda f: f.stat().st_mtime)
        cache_date = datetime.fromtimestamp(latest_cache.stat().st_mtime)
        print(
            f"💾 Cache válido encontrado: {len(cache_files)} archivos (último: {cache_date.strftime('%Y-%m-%d %H:%M')})")
    else:
        print("💾 No se encontró cache válido")

    # LÓGICA PRINCIPAL: ¿Procesar Excel o usar cache?
    data = []

    if args.load_data:
        print(f"📂 Cargando datos desde JSON: {args.load_data}")
        with open(args.load_data, 'r', encoding='utf-8') as f:
            data = json.load(f)

    elif has_cache and not force_excel and not force_regenerate:
        print("🚀 MODO CACHE: Saltando procesamiento de Excel")
        print("💡 Usa --force-excel si necesitas reprocesar los archivos")
        data = load_cached_data(args.cache_dir)

    else:
        print("📊 MODO EXCEL: Procesando archivos...")

        # Determinar archivos a procesar
        if args.files:
            excel_files = args.files
        else:
            excel_files = get_excel_files_from_dir(args.source_dir)

        if not excel_files:
            print("❌ No se encontraron archivos Excel para procesar")
            return

        processor = TourismDataProcessor()
        data = processor.extract_all_excel_data(excel_files)

        if args.save_data:
            print(f"💾 Guardando datos en: {args.save_data}")
            with open(args.save_data, 'w', encoding='utf-8') as f:
                json.dump(data, f, ensure_ascii=False, indent=2)

    # Inicializar sistema Q&A con opciones de cache
    try:
        qa_system = TourismQA(data, cache_dir=args.cache_dir,
                              force_regenerate=force_regenerate,
                              use_cache_only=use_cache_only or has_cache)
    except Exception as e:
        print(f"❌ Error inicializando sistema: {e}")
        return

    # Si hay consulta directa
    if args.query:
        if args.json_output:
            result = qa_system.query_json(args.query)
            print(json.dumps(result, ensure_ascii=False, indent=2))
        else:
            result = qa_system.answer(args.query)
            print(f"\n🤖 Respuesta:")
            print("-" * 30)
            print(result['answer'])
            print(f"\n📊 Fuentes utilizadas:")
            for fuente in result['fuentes'][:3]:
                print(f"{fuente['idx']}. {fuente['fuente']} ({fuente['publish_date']})")
        return

    print("\n🎯 SISTEMA LISTO - Escribe 'salir' para terminar")
    print("💡 TIP: El sistema usa cache de embeddings automáticamente")
    print("=" * 60)

    # Loop interactivo
    while True:
        try:
            question = input("\n💤 Tu pregunta: ").strip()

            if question.lower() in ['salir', 'exit', 'quit']:
                print("👋 ¡Hasta luego!")
                break

            if not question:
                continue

            # Responder
            result = qa_system.answer(question)

            print(f"\n🤖 Respuesta:")
            print("-" * 30)
            print(result['answer'])

            if result['fuentes']:
                print(f"\n📊 Fuentes utilizadas:")
                for fuente in result['fuentes'][:3]:
                    print(f"{fuente['idx']}. {fuente['fuente']} ({fuente['publish_date']})")
                    print(f"   Referencia: {fuente['ref']}")

        except KeyboardInterrupt:
            print("\n\n👋 ¡Hasta luego!")
            break
        except Exception as e:
            print(f"❌ Error: {e}")


# EJEMPLOS DE USO CON CACHE ULTRA-RÁPIDO:
# python tourism_qa.py                                    # Modo super rápido: usa cache automáticamente
# python tourism_qa.py --cache-only                       # Solo cache, error si no existe
# python tourism_qa.py --force-excel                      # Fuerza procesamiento Excel aunque haya cache
# python tourism_qa.py --clear-cache                      # Limpia cache y regenera todo
# python tourism_qa.py --query "¿Cuántos turistas en junio?" # Consulta rápida con cache

# MODO RÁPIDO POR DEFECTO:
# - Si hay cache válido: lo usa SIN procesar Excel (ultra rápido)
# - Si NO hay cache: procesa Excel y crea cache para próximas veces

# CONFIGURACIÓN vLLM SERVER (variables de entorno):
# export VLLM_BASE_URL="http://172.24.250.17:8000/v1"
# export VLLM_API_KEY="NoLoNecesitas"
# export VLLM_MODEL="gemma-3-12b-it"

if __name__ == "__main__":
    main()

# PREGUNTAS DE EJEMPLO:
"""
- ¿Cuántos turistas hubo en junio 2025?
- ¿Cómo evolucionaron las pernoctaciones de turistas españoles?
- ¿Cuál es el gasto medio diario de los andaluces?
- ¿Qué aeropuerto tuvo más llegadas?
- ¿Cuál fue la variación anual en el primer trimestre?
- Compara turistas españoles vs andaluces en junio
"""