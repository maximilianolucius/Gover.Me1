"""Closed-loop adaptive search engine with 6-dimensional quality metrics.

Iteratively searches, evaluates answer quality across completeness, accuracy,
consistency, depth, freshness, and authority, then refines until a target
quality threshold is reached or diminishing returns are detected.
"""

import os
import json
import asyncio
import time
import numpy as np
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Tuple
from enum import Enum
import requests
from bs4 import BeautifulSoup
from collections import deque
import statistics

try:
    from ddgs import DDGS
except ImportError:
    from duckduckgo_search import DDGS

# Configuración
os.environ.update({
    'VLLM_BASE_URL': 'http://172.24.250.17:8000/v1',
    'VLLM_MODEL': 'gemma-3-12b-it'
})


@dataclass
class QualityMetrics:
    """Métricas multidimensionales de calidad"""
    completeness: float = 0.0  # ¿Responde completamente la pregunta?
    accuracy: float = 0.0  # ¿La información es precisa?
    consistency: float = 0.0  # ¿Las fuentes son consistentes?
    depth: float = 0.0  # ¿Nivel de detalle suficiente?
    freshness: float = 0.0  # ¿Información reciente/actualizada?
    authority: float = 0.0  # ¿Fuentes autoritativas?

    def overall_score(self) -> float:
        """Puntuación ponderada general"""
        weights = [0.25, 0.20, 0.15, 0.15, 0.15, 0.10]  # Suma = 1.0
        values = [self.completeness, self.accuracy, self.consistency,
                  self.depth, self.freshness, self.authority]
        return sum(w * v for w, v in zip(weights, values))

    def to_dict(self) -> Dict[str, float]:
        """Return all metric values including the weighted overall score."""
        return {
            'completeness': self.completeness,
            'accuracy': self.accuracy,
            'consistency': self.consistency,
            'depth': self.depth,
            'freshness': self.freshness,
            'authority': self.authority,
            'overall': self.overall_score()
        }


@dataclass
class SearchState:
    """Estado completo del sistema de búsqueda"""
    query: str
    current_answer: str = ""
    evidence: List[Dict] = field(default_factory=list)
    search_history: List[str] = field(default_factory=list)
    quality_history: List[QualityMetrics] = field(default_factory=list)
    iteration: int = 0
    total_tokens: int = 0
    plateau_count: int = 0  # Contador de estancamiento
    last_improvement: int = 0  # Última iteración con mejora
    visited_urls: set = field(default_factory=set)

    def get_quality_trend(self, window=3) -> float:
        """Tendencia de calidad en las últimas iteraciones"""
        if len(self.quality_history) < 2:
            return 0.0

        recent_scores = [q.overall_score() for q in self.quality_history[-window:]]
        if len(recent_scores) < 2:
            return 0.0

        # Calcular tendencia (pendiente)
        x = list(range(len(recent_scores)))
        y = recent_scores
        n = len(x)

        slope = (n * sum(x[i] * y[i] for i in range(n)) - sum(x) * sum(y)) / \
                (n * sum(x[i] ** 2 for i in range(n)) - sum(x) ** 2)

        return slope


class AdaptiveSearchEngine:
    """Motor de búsqueda adaptativo con autoevaluación continua"""

    def __init__(self, quality_threshold=0.75, max_iterations=15, plateau_tolerance=3):
        self.llm = LocalLLM()
        self.ddgs = DDGS()
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (compatible; AdaptiveSearch/2.0)'
        })

        # Configuración de calidad
        self.quality_threshold = quality_threshold
        self.max_iterations = max_iterations
        self.plateau_tolerance = plateau_tolerance

        # Umbrales dinámicos
        self.min_sources_threshold = 2
        self.diminishing_returns_threshold = 0.05  # Mejora mínima por iteración

    async def search_with_feedback_loop(self, query: str) -> Dict[str, Any]:
        """Bucle principal de búsqueda adaptativa"""
        print(f"🎯 Iniciando búsqueda adaptativa: '{query}'")
        print(f"📊 Objetivo: calidad ≥ {self.quality_threshold}")

        state = SearchState(query=query)

        # Búsqueda inicial
        await self._perform_initial_search(state)

        # Bucle de refinamiento
        while not await self._should_terminate(state):
            state.iteration += 1
            print(f"\n🔄 Iteración {state.iteration}")

            # Evaluar estado actual
            current_quality = await self._evaluate_current_state(state)
            state.quality_history.append(current_quality)

            print(f"📈 Calidad actual: {current_quality.overall_score():.3f}")
            self._print_quality_breakdown(current_quality)

            # Detectar áreas de mejora
            improvement_areas = self._identify_improvement_areas(current_quality)
            print(f"🎯 Áreas de mejora: {improvement_areas}")

            # Ejecutar estrategia de mejora
            improvement_made = await self._execute_improvement_strategy(state, improvement_areas)

            if improvement_made:
                state.last_improvement = state.iteration
                state.plateau_count = 0
            else:
                state.plateau_count += 1

            # Mostrar progreso
            await self._show_progress(state)

        # Generar respuesta final
        final_answer = await self._synthesize_final_answer(state)
        final_quality = state.quality_history[-1] if state.quality_history else QualityMetrics()

        return self._build_result_summary(state, final_answer, final_quality)

    async def _perform_initial_search(self, state: SearchState):
        """Búsqueda inicial para establecer baseline"""
        print("🔍 Realizando búsqueda inicial...")

        # Buscar con query original
        results = await self._search_web(state.query)

        # Obtener contenido de las mejores URLs
        for i, result in enumerate(results[:3]):
            content = await self._extract_url_content(result['url'])
            if content:
                state.evidence.append({
                    'url': result['url'],
                    'title': result['title'],
                    'content': content,
                    'iteration_found': 0
                })
                state.visited_urls.add(result['url'])

        # Generar respuesta inicial
        if state.evidence:
            state.current_answer = await self._generate_answer_from_evidence(state)

        state.search_history.append(state.query)

    async def _evaluate_current_state(self, state: SearchState) -> QualityMetrics:
        """Evaluación multidimensional del estado actual"""
        if not state.current_answer:
            return QualityMetrics()

        # Preparar contexto para evaluación
        evidence_summary = "\n".join([
            f"Fuente {i + 1}: {ev['title']} - {ev['content'][:200]}..."
            for i, ev in enumerate(state.evidence)
        ])

        evaluation_prompt = f"""Evalúa la calidad de esta respuesta según múltiples criterios (0.0-1.0):

PREGUNTA: {state.query}

RESPUESTA ACTUAL:
{state.current_answer}

EVIDENCIA DISPONIBLE:
{evidence_summary}

Evalúa cada criterio de 0.0 a 1.0:

1. COMPLETENESS (¿Responde completamente la pregunta?):
   - 1.0: Respuesta completa y exhaustiva
   - 0.5: Respuesta parcial pero útil  
   - 0.0: Respuesta incompleta o evasiva

2. ACCURACY (¿La información es precisa y verificable?):
   - 1.0: Información precisa y bien verificada
   - 0.5: Mayormente precisa con algunos detalles inciertos
   - 0.0: Información inexacta o no verificable

3. CONSISTENCY (¿Las fuentes son consistentes entre sí?):
   - 1.0: Fuentes completamente consistentes
   - 0.5: Mayormente consistentes con algunas diferencias menores
   - 0.0: Fuentes contradictorias o inconsistentes

4. DEPTH (¿Nivel de detalle y profundidad adecuado?):
   - 1.0: Análisis profundo con detalles relevantes
   - 0.5: Nivel medio de detalle
   - 0.0: Superficial o carente de detalles

5. FRESHNESS (¿Información actualizada y reciente?):
   - 1.0: Información muy reciente y actualizada
   - 0.5: Información relativamente reciente
   - 0.0: Información desactualizada

6. AUTHORITY (¿Fuentes autoritativas y confiables?):
   - 1.0: Fuentes altamente autoritativas
   - 0.5: Fuentes moderadamente confiables
   - 0.0: Fuentes poco confiables

Responde SOLO con formato JSON:
{{
  "completeness": 0.X,
  "accuracy": 0.X,
  "consistency": 0.X,
  "depth": 0.X,
  "freshness": 0.X,
  "authority": 0.X
}}"""

        try:
            response = await self.llm.generate(evaluation_prompt, max_tokens=200)

            # Extraer JSON
            if '{' in response:
                json_str = response[response.find('{'):response.rfind('}') + 1]
                scores = json.loads(json_str)

                return QualityMetrics(
                    completeness=scores.get('completeness', 0.5),
                    accuracy=scores.get('accuracy', 0.5),
                    consistency=scores.get('consistency', 0.5),
                    depth=scores.get('depth', 0.5),
                    freshness=scores.get('freshness', 0.5),
                    authority=scores.get('authority', 0.5)
                )
            else:
                print("⚠️ No se pudo parsear evaluación, usando fallback")
                return self._heuristic_evaluation(state)

        except Exception as e:
            print(f"⚠️ Error en evaluación LLM: {e}")
            return self._heuristic_evaluation(state)

    def _heuristic_evaluation(self, state: SearchState) -> QualityMetrics:
        """Evaluación heurística como fallback"""
        answer_length = len(state.current_answer.split())
        evidence_count = len(state.evidence)

        # Métricas básicas basadas en heurísticas
        completeness = min(answer_length / 100, 1.0)  # Longitud como proxy
        accuracy = min(evidence_count / 3, 1.0)  # Más fuentes = más precisión
        consistency = 0.7 if evidence_count >= 2 else 0.4
        depth = min(answer_length / 150, 1.0)
        freshness = 0.6  # Neutral sin información temporal
        authority = 0.6  # Neutral sin análisis de dominios

        return QualityMetrics(completeness, accuracy, consistency,
                              depth, freshness, authority)

    def _identify_improvement_areas(self, quality: QualityMetrics) -> List[str]:
        """Identifica las áreas que necesitan mejora"""
        areas = []
        threshold = 0.6  # Umbral mínimo por dimensión

        if quality.completeness < threshold:
            areas.append('completeness')
        if quality.accuracy < threshold:
            areas.append('accuracy')
        if quality.consistency < threshold:
            areas.append('consistency')
        if quality.depth < threshold:
            areas.append('depth')
        if quality.freshness < threshold:
            areas.append('freshness')
        if quality.authority < threshold:
            areas.append('authority')

        return areas

    async def _execute_improvement_strategy(self, state: SearchState, areas: List[str]) -> bool:
        """Ejecuta estrategias específicas para mejorar áreas deficientes"""
        improvement_made = False

        for area in areas[:2]:  # Máximo 2 mejoras por iteración
            if area == 'completeness':
                improvement_made |= await self._improve_completeness(state)
            elif area == 'accuracy':
                improvement_made |= await self._improve_accuracy(state)
            elif area == 'consistency':
                improvement_made |= await self._improve_consistency(state)
            elif area == 'depth':
                improvement_made |= await self._improve_depth(state)
            elif area == 'freshness':
                improvement_made |= await self._improve_freshness(state)
            elif area == 'authority':
                improvement_made |= await self._improve_authority(state)

        return improvement_made

    async def _improve_completeness(self, state: SearchState) -> bool:
        """Buscar información adicional para completar la respuesta"""
        print("📝 Mejorando completeness...")

        # Identificar gaps en la respuesta actual
        gap_query = f"{state.query} comprehensive complete guide detailed"
        return await self._search_and_add_evidence(state, gap_query)

    async def _improve_accuracy(self, state: SearchState) -> bool:
        """Verificar y mejorar la precisión de la información"""
        print("🎯 Mejorando accuracy...")

        # Buscar fuentes verificables y oficiales
        verify_query = f"{state.query} official data statistics facts verified"
        return await self._search_and_add_evidence(state, verify_query)

    async def _improve_consistency(self, state: SearchState) -> bool:
        """Buscar fuentes adicionales para verificar consistencia"""
        print("⚖️ Mejorando consistency...")

        consistency_query = f"{state.query} multiple sources comparison analysis"
        return await self._search_and_add_evidence(state, consistency_query)

    async def _improve_depth(self, state: SearchState) -> bool:
        """Añadir más profundidad y detalles"""
        print("🔬 Mejorando depth...")

        depth_query = f"{state.query} detailed analysis in-depth explanation"
        return await self._search_and_add_evidence(state, depth_query)

    async def _improve_freshness(self, state: SearchState) -> bool:
        """Buscar información más reciente"""
        print("🕐 Mejorando freshness...")

        fresh_query = f"{state.query} 2024 2025 latest recent current update"
        return await self._search_and_add_evidence(state, fresh_query)

    async def _improve_authority(self, state: SearchState) -> bool:
        """Buscar fuentes más autoritativas"""
        print("🏛️ Mejorando authority...")

        authority_query = f"{state.query} site:gov OR site:edu OR site:org official research"
        return await self._search_and_add_evidence(state, authority_query)

    async def _search_and_add_evidence(self, state: SearchState, query: str) -> bool:
        """Buscar y añadir nueva evidencia"""
        if query in state.search_history:
            return False  # Evitar búsquedas duplicadas

        state.search_history.append(query)
        results = await self._search_web(query)

        new_evidence_added = False
        for result in results[:2]:  # Máximo 2 nuevas fuentes por mejora
            if result['url'] not in state.visited_urls:
                content = await self._extract_url_content(result['url'])
                if content and len(content.split()) > 50:  # Contenido sustancial
                    state.evidence.append({
                        'url': result['url'],
                        'title': result['title'],
                        'content': content,
                        'iteration_found': state.iteration
                    })
                    state.visited_urls.add(result['url'])
                    new_evidence_added = True

        # Regenerar respuesta si se añadió nueva evidencia
        if new_evidence_added:
            state.current_answer = await self._generate_answer_from_evidence(state)

        return new_evidence_added

    async def _should_terminate(self, state: SearchState) -> bool:
        """Decide si terminar la búsqueda"""
        # Límite de iteraciones
        if state.iteration >= self.max_iterations:
            print(f"🛑 Límite de iteraciones alcanzado ({self.max_iterations})")
            return True

        # Calidad suficiente alcanzada
        if state.quality_history:
            current_quality = state.quality_history[-1].overall_score()
            if current_quality >= self.quality_threshold:
                print(f"✅ Calidad objetivo alcanzada ({current_quality:.3f} ≥ {self.quality_threshold})")
                return True

        # Estancamiento detectado
        if state.plateau_count >= self.plateau_tolerance:
            print(f"📉 Estancamiento detectado ({state.plateau_count} iteraciones sin mejora)")
            return True

        # Rendimientos decrecientes
        if len(state.quality_history) >= 3:
            trend = state.get_quality_trend()
            if trend < self.diminishing_returns_threshold:
                print(f"📊 Rendimientos decrecientes detectados (tendencia: {trend:.4f})")
                return True

        # Búsqueda estéril (sin fuentes útiles encontradas)
        if state.iteration >= 3 and len(state.evidence) < self.min_sources_threshold:
            print("🚫 Búsqueda estéril: fuentes insuficientes encontradas")
            return True

        return False

    async def _search_web(self, query: str) -> List[Dict]:
        """Buscar en web con manejo de errores"""
        try:
            results = list(self.ddgs.text(query, max_results=5))
            return [
                {
                    'title': r.get('title', ''),
                    'url': r.get('href', '') or r.get('link', ''),
                    'snippet': r.get('body', '') or r.get('snippet', '')
                }
                for r in results if r.get('href') or r.get('link')
            ]
        except Exception as e:
            print(f"⚠️ Error en búsqueda web: {e}")
            return []

    async def _extract_url_content(self, url: str) -> Optional[str]:
        """Extraer contenido de URL con validación"""
        if not url.startswith(('http://', 'https://')):
            return None

        try:
            response = self.session.get(url, timeout=10)
            if response.status_code != 200:
                return None

            soup = BeautifulSoup(response.content, 'html.parser')

            # Limpiar HTML
            for tag in soup(['script', 'style', 'nav', 'footer', 'aside']):
                tag.decompose()

            text = soup.get_text(separator=' ', strip=True)

            # Truncar texto muy largo
            words = text.split()
            if len(words) > 800:
                text = ' '.join(words[:800])

            return text if len(words) > 30 else None

        except Exception:
            return None

    async def _generate_answer_from_evidence(self, state: SearchState) -> str:
        """Generar respuesta basada en evidencia actual"""
        if not state.evidence:
            return "No se encontró información suficiente para responder la pregunta."

        evidence_text = "\n\n".join([
            f"Fuente {i + 1} ({ev['url']}):\n{ev['content'][:600]}..."
            for i, ev in enumerate(state.evidence)
        ])

        synthesis_prompt = f"""Basándote en la siguiente evidencia, proporciona una respuesta completa y bien estructurada para: {state.query}

EVIDENCIA:
{evidence_text}

Instrucciones:
1. Sintetiza la información de todas las fuentes
2. Resuelve contradicciones si las hay
3. Organiza la respuesta de forma lógica
4. Incluye detalles específicos y datos relevantes
5. Menciona limitaciones si las hay

Respuesta:"""

        return await self.llm.generate(synthesis_prompt, max_tokens=500)

    async def _synthesize_final_answer(self, state: SearchState) -> str:
        """Síntesis final optimizada"""
        if not state.current_answer:
            state.current_answer = await self._generate_answer_from_evidence(state)

        # Verificación final y pulido
        final_prompt = f"""Revisa y mejora esta respuesta final para la pregunta: {state.query}

RESPUESTA ACTUAL:
{state.current_answer}

TOTAL DE FUENTES: {len(state.evidence)}
ITERACIONES: {state.iteration}

Mejora la respuesta asegurándote de que:
1. Sea completa y directa
2. Esté bien organizada
3. Incluya información clave
4. Sea clara y concisa

Respuesta mejorada:"""

        return await self.llm.generate(final_prompt, max_tokens=400)

    def _print_quality_breakdown(self, quality: QualityMetrics):
        """Mostrar desglose de calidad"""
        metrics = quality.to_dict()
        for metric, score in metrics.items():
            status = "✅" if score >= 0.7 else "⚠️" if score >= 0.5 else "❌"
            print(f"  {status} {metric}: {score:.3f}")

    async def _show_progress(self, state: SearchState):
        """Mostrar progreso del sistema"""
        print(f"📊 Progreso: {len(state.evidence)} fuentes, {len(state.search_history)} búsquedas")

        if len(state.quality_history) >= 2:
            trend = state.get_quality_trend()
            trend_icon = "📈" if trend > 0 else "📉" if trend < 0 else "➡️"
            print(f"{trend_icon} Tendencia: {trend:.4f}")

    def _build_result_summary(self, state: SearchState, final_answer: str, final_quality: QualityMetrics) -> Dict[
        str, Any]:
        """Construir resumen final de resultados"""
        return {
            'query': state.query,
            'final_answer': final_answer,
            'quality_score': final_quality.overall_score(),
            'quality_breakdown': final_quality.to_dict(),
            'iterations': state.iteration,
            'sources_found': len(state.evidence),
            'searches_performed': len(state.search_history),
            'plateau_count': state.plateau_count,
            'quality_trend': state.get_quality_trend(),
            'termination_reason': self._get_termination_reason(state),
            'evidence_sources': [ev['url'] for ev in state.evidence]
        }

    def _get_termination_reason(self, state: SearchState) -> str:
        """Determinar razón de terminación"""
        if state.quality_history:
            current_quality = state.quality_history[-1].overall_score()
            if current_quality >= self.quality_threshold:
                return "Quality threshold reached"

        if state.iteration >= self.max_iterations:
            return "Maximum iterations reached"

        if state.plateau_count >= self.plateau_tolerance:
            return "Quality plateau detected"

        if len(state.quality_history) >= 3 and state.get_quality_trend() < self.diminishing_returns_threshold:
            return "Diminishing returns detected"

        if state.iteration >= 3 and len(state.evidence) < self.min_sources_threshold:
            return "Insufficient sources found"

        return "Unknown"


# LLM Client (igual que antes)
class LocalLLM:
    """Thin client for a local vLLM server exposing an OpenAI-compatible API."""

    def __init__(self):
        self.base_url = os.getenv('VLLM_BASE_URL')
        self.model = os.getenv('VLLM_MODEL')

    async def generate(self, prompt: str, max_tokens: int = 500) -> str:
        """Send a chat completion request and return the generated text."""
        try:
            payload = {
                "model": self.model,
                "messages": [{"role": "user", "content": prompt}],
                "max_tokens": max_tokens,
                "temperature": 0.7,
                "stream": False
            }

            response = requests.post(
                f"{self.base_url}/chat/completions",
                json=payload,
                timeout=30
            )

            if response.status_code == 200:
                data = response.json()
                return data['choices'][0]['message']['content']
            else:
                return "Error generating response"

        except Exception as e:
            print(f"LLM error: {e}")
            return "Error generating response"


# Demo
async def main():
    """Run demo adaptive searches with preconfigured test queries."""
    print("🚀 Sistema de Búsqueda Adaptativa - Bucle Cerrado")
    print("=" * 60)

    # Configuración del sistema
    search_engine = AdaptiveSearchEngine(
        quality_threshold=0.75,  # Calidad objetivo
        max_iterations=12,  # Máximo iteraciones
        plateau_tolerance=3  # Tolerancia al estancamiento
    )

    # Queries de prueba
    test_queries = [
        "What is the current inflation rate in Argentina 2024?",
        "How does quantum computing work and what are its current limitations?",
        "What are the main tourist attractions in Buenos Aires?"
    ]
    test_queries = ['G19, actividad 6201 - Programación informática, Tipo SL  modelo de tributacion España']


    for i, query in enumerate(test_queries, 1):
        print(f"\n{'=' * 60}")
        print(f"PRUEBA {i}: {query}")
        print('=' * 60)

        start_time = time.time()
        result = await search_engine.search_with_feedback_loop(query)
        duration = time.time() - start_time

        print(f"\n📋 RESULTADOS FINALES:")
        print(f"Respuesta: {result['final_answer']}...")
        print(f"Calidad final: {result['quality_score']:.3f}")
        print(f"Iteraciones: {result['iterations']}")
        print(f"Fuentes: {result['sources_found']}")
        print(f"Razón de parada: {result['termination_reason']}")
        print(f"Tiempo: {duration:.1f}s")

        print("\n📊 Desglose de calidad:")
        for metric, score in result['quality_breakdown'].items():
            if metric != 'overall':
                print(f"  {metric}: {score:.3f}")

        await asyncio.sleep(2)


if __name__ == "__main__":
    asyncio.run(main())