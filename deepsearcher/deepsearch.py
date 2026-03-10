"""Advanced search engine using a local vLLM server and reinforcement learning.

Combines a Thompson Sampling multi-armed bandit for query-strategy selection
with an LLM-based planner that decides when to search, open URLs, or answer.
"""

# DeepSearch Starter Project
# Simple implementation using local vLLM + basic RL

import os
import json
import asyncio
import time
import random
import re
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional
from enum import Enum
import requests
from bs4 import BeautifulSoup

try:
    from ddgs import DDGS
except ImportError:
    from duckduckgo_search import DDGS

# Set up your credentials
os.environ.update({
    'VLLM_BASE_URL': 'http://172.24.250.17:8000/v1',
    'VLLM_MODEL': 'gemma-3-12b-it'
})


@dataclass
class SearchResult:
    """A single web search result with title, URL, and snippet."""

    title: str
    url: str
    snippet: str


@dataclass
class Evidence:
    """Extracted page content with a relevance score relative to the query."""

    content: str
    url: str
    relevance: float


@dataclass
class State:
    """Mutable state for a single search session tracking results, evidence, and budget."""

    query: str
    results: List[SearchResult] = field(default_factory=list)
    evidence: List[Evidence] = field(default_factory=list)
    clicks: int = 0
    tokens: int = 0
    done: bool = False
    visited_urls: set = field(default_factory=set)  # Track visited URLs


class Action(Enum):
    """Possible agent actions during a search episode."""

    SEARCH = "search"
    OPEN = "open"
    ANSWER = "answer"


# 1. vLLM Client
class LocalLLM:
    """Thin client for a local vLLM server exposing an OpenAI-compatible API."""

    def __init__(self):
        self.base_url = os.getenv('VLLM_BASE_URL')
        self.model = os.getenv('VLLM_MODEL')
        print(f"Using vLLM at {self.base_url} with model {self.model}")

    async def generate(self, prompt: str, max_tokens: int = 500) -> str:
        """Generate using local vLLM server"""
        try:
            # Use OpenAI-compatible API
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
                print(f"vLLM error: {response.status_code} - {response.text}")
                return "Error generating response"

        except Exception as e:
            print(f"LLM generation error: {e}")
            return "Error generating response"


# 2. Simple Search Environment
class SearchEnv:
    """Environment that wraps DuckDuckGo search and web page content extraction."""

    def __init__(self, max_clicks: int = 20):
        self.max_clicks = max_clicks
        self.ddgs = DDGS()
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (compatible; DeepSearch/1.0)'
        })

    def reset(self, query: str) -> State:
        """Create a fresh search state for a new query."""
        return State(query=query, visited_urls=set())

    def search(self, state: State, query: str) -> tuple[State, float]:
        """Execute search action"""
        try:
            results = list(self.ddgs.text(query, max_results=6))
            state.results = [
                SearchResult(
                    title=r.get('title', ''),
                    url=r.get('href', '') or r.get('link', ''),
                    snippet=r.get('body', '') or r.get('snippet', '')
                ) for r in results if r.get('href') or r.get('link')
            ]
            reward = 0.2 if state.results else -0.1
            print(f"🔍 Search '{query}' found {len(state.results)} results")

            # Debug: print first few results
            for i, r in enumerate(state.results[:3]):
                print(f"  {i + 1}. {r.title[:50]}... -> {r.url[:60]}")

            return state, reward
        except Exception as e:
            print(f"Search error: {e}")
            return state, -0.2

    def open_url(self, state: State, url: str) -> tuple[State, float]:
        """Extract content from URL"""
        if state.clicks >= self.max_clicks:
            return state, -0.3

        # Validate URL
        if not url or not url.startswith(('http://', 'https://')):
            print(f"❌ Invalid URL: {url}")
            return state, -0.2

        # Check if already visited
        if url in state.visited_urls:
            print(f"🔄 Already visited: {url[:60]}...")
            return state, -0.1

        state.clicks += 1
        state.visited_urls.add(url)
        print(f"📄 Opening: {url[:80]}...")

        try:
            response = self.session.get(url, timeout=15)
            if response.status_code != 200:
                print(f"❌ HTTP {response.status_code} for {url}")
                return state, -0.1

            soup = BeautifulSoup(response.content, 'html.parser')

            # Clean HTML more thoroughly
            for tag in soup(['script', 'style', 'nav', 'footer', 'aside', 'header', 'menu']):
                tag.decompose()

            # Remove ads and navigation
            for class_name in ['advertisement', 'ad', 'sidebar', 'navigation', 'nav-menu']:
                for element in soup.find_all(class_=class_name):
                    element.decompose()

            text = soup.get_text(separator=' ', strip=True)

            # Remove extra whitespace
            import re
            text = re.sub(r'\s+', ' ', text).strip()

            # Dynamic truncation based on content quality
            words = text.split()
            if len(words) > 1000:
                # Keep first 800 words for longer articles
                text = ' '.join(words[:800])
            elif len(words) > 500:
                # Keep first 500 words for medium articles
                text = ' '.join(words[:500])
            # Keep all text for shorter articles

            word_count = len(text.split())

            if word_count < 30:  # Too little content
                print(f"⚠️ Little content extracted ({word_count} words) from {url}")
                return state, -0.1

            # Improved relevance scoring
            query_words = set(word.lower() for word in state.query.split() if len(word) > 2)
            content_words = set(word.lower() for word in text.split() if len(word) > 2)

            if not query_words:  # Fallback for very short queries
                relevance = 0.5
            else:
                # Calculate overlap
                overlap = len(query_words & content_words)
                relevance = overlap / len(query_words)

                # Boost for exact phrase matches
                query_lower = state.query.lower()
                if query_lower in text.lower():
                    relevance += 0.2

                # Clamp between 0 and 1
                relevance = min(1.0, max(0.0, relevance))

            evidence = Evidence(content=text, url=url, relevance=relevance)
            state.evidence.append(evidence)
            state.tokens += word_count

            print(f"✅ Extracted {word_count} words (relevance: {relevance:.2f})")
            return state, relevance

        except requests.exceptions.Timeout:
            print(f"⏰ Timeout accessing {url}")
            return state, -0.1
        except requests.exceptions.ConnectionError:
            print(f"🔌 Connection error for {url}")
            return state, -0.1
        except Exception as e:
            print(f"❌ Error opening {url}: {str(e)[:100]}")
            return state, -0.1


# 3. RL Agent with Thompson Sampling
class ThompsonBandit:
    """Multi-armed bandit for search strategies"""

    def __init__(self, strategies: List[str]):
        self.strategies = strategies
        self.alpha = {s: 1.0 for s in strategies}  # Success count
        self.beta = {s: 1.0 for s in strategies}  # Failure count

    def select_strategy(self) -> str:
        """Select strategy using Thompson sampling"""
        samples = {
            strategy: random.betavariate(self.alpha[strategy], self.beta[strategy])
            for strategy in self.strategies
        }
        return max(samples, key=samples.get)

    def update(self, strategy: str, reward: float):
        """Update belief about strategy effectiveness"""
        if reward > 0.3:  # Success threshold
            self.alpha[strategy] += 1
        else:
            self.beta[strategy] += 1

    def get_stats(self) -> Dict[str, float]:
        """Get current strategy preferences"""
        return {
            strategy: self.alpha[strategy] / (self.alpha[strategy] + self.beta[strategy])
            for strategy in self.strategies
        }


# 4. Simple Planning Agent
class PlannerAgent:
    """LLM-driven agent that plans the next search action based on current state."""

    def __init__(self, llm: LocalLLM):
        self.llm = llm

    def classify_query_intent(self, query: str) -> str:
        """Classify query to select best search strategy"""
        query_lower = query.lower()

        if any(word in query_lower for word in
               ['historicos', 'historical', 'museums', 'monuments', 'culture', 'patrimonio']):
            return 'historical'
        elif any(word in query_lower for word in ['dias', 'days', 'itinerary', 'trip', 'viaje', 'ruta', 'visit']):
            return 'travel'
        elif any(word in query_lower for word in ['price', 'cost', 'tiempo', 'time', 'hora', 'when', 'cuanto']):
            return 'factual'
        elif any(word in query_lower for word in ['como', 'how', 'que es', 'what is', 'explain', 'funciona']):
            return 'educational'
        else:
            return 'general'

    def should_stop_search(self, state: State, target_evidence_quality=0.6, min_sources=2) -> bool:
        """Smart stopping condition - decide if we have enough quality evidence"""
        if len(state.evidence) < min_sources:
            return False

        # Check if we have high-quality evidence
        avg_relevance = sum(e.relevance for e in state.evidence) / len(state.evidence)
        if avg_relevance >= target_evidence_quality:
            print(f"🎯 Quality threshold reached (avg: {avg_relevance:.2f})")
            return True

        # Check if we have at least one high-quality source
        max_relevance = max(e.relevance for e in state.evidence) if state.evidence else 0
        if max_relevance >= 0.8:
            print(f"🌟 High-quality source found (max: {max_relevance:.2f})")
            return True

        # Stop if we have 3+ sources with decent quality
        if len(state.evidence) >= 3 and avg_relevance >= 0.4:
            print(f"📚 Sufficient sources with decent quality ({len(state.evidence)} sources)")
            return True

        return False

    def prioritize_urls(self, search_results: List[SearchResult], query: str) -> List[SearchResult]:
        """Prioritize URLs based on query intent and domain authority"""

        # Domain authority scores (higher = more authoritative)
        domain_scores = {
            'wikipedia.org': 0.9,
            'unesco.org': 0.9,
            '.gov': 0.8,
            '.edu': 0.8,
            'turismo': 0.7,
            'tourism': 0.7,
            'viajes': 0.6,
            'travel': 0.6,
            'guia': 0.6,
            'guide': 0.6,
            'oficial': 0.7,
            'official': 0.7,
            'blog': 0.4,
            'forum': 0.3,
            'reddit': 0.3
        }

        query_words = set(query.lower().split())

        scored_urls = []
        for result in search_results:
            score = 0.5  # Base score

            # Domain authority bonus
            for domain, bonus in domain_scores.items():
                if domain in result.url.lower():
                    score += bonus * 0.3
                    break

            # Title relevance bonus
            title_words = set(result.title.lower().split())
            title_overlap = len(query_words & title_words) / max(len(query_words), 1)
            score += title_overlap * 0.4

            # Snippet relevance bonus
            snippet_words = set(result.snippet.lower().split())
            snippet_overlap = len(query_words & snippet_words) / max(len(query_words), 1)
            score += snippet_overlap * 0.3

            # Boost for exact matches in title/snippet
            query_text = query.lower()
            if any(word in result.title.lower() for word in query_text.split() if len(word) > 3):
                score += 0.2

            scored_urls.append((score, result))

        # Sort by score (highest first)
        scored_urls.sort(key=lambda x: x[0], reverse=True)

        # Debug: show prioritization
        print("📊 URL Prioritization:")
        for i, (score, result) in enumerate(scored_urls[:3]):
            print(f"  {i + 1}. (Score: {score:.2f}) {result.title[:40]}...")

        return [result for score, result in scored_urls]

    async def plan_next_action(self, state: State, max_clicks: int) -> tuple[Action, Dict[str, Any]]:
        """Enhanced planning with smart stopping and URL prioritization"""

        # Classify query intent
        query_intent = self.classify_query_intent(state.query)

        # Calculate evidence metrics
        avg_relevance = sum(e.relevance for e in state.evidence) / max(len(state.evidence), 1)
        max_relevance = max([e.relevance for e in state.evidence] + [0])

        print(f"🧠 Planning: Intent={query_intent}, AvgRel={avg_relevance:.2f}, Sources={len(state.evidence)}")

        # Smart stopping condition
        if self.should_stop_search(state) or state.clicks >= max_clicks - 1:
            print("🛑 Stopping: Quality threshold reached or max clicks")
            return Action.ANSWER, {"target": ""}

        # Prioritize URLs if we have search results
        if state.results:
            state.results = self.prioritize_urls(state.results, state.query)

        # Enhanced context with evidence summary
        results_text = "\n".join([
            f"- [{i + 1}] {r.title[:60]}... [URL: {r.url}]"
            for i, r in enumerate(state.results[:20])
        ])

        evidence_text = await self._get_evidence_summary(state.evidence)

        evidence_coverage = "Good" if avg_relevance > 0.5 else "Fair" if avg_relevance > 0.3 else "Poor"

        prompt = f"""You are a research agent. Based on the state below, decide the next action.

QUERY: {state.query}
INTENT: {query_intent}
RESOURCES: {state.clicks}/{max_clicks} clicks, {state.tokens} tokens

SEARCH RESULTS:
{results_text or "None"}

EVIDENCE QUALITY:
{evidence_text}
- Coverage: {evidence_coverage}
- Average relevance: {avg_relevance:.2f}
- Best source: {max_relevance:.2f}

Available actions:
1. SEARCH - search with new/refined query
2. OPEN - open a high-priority unvisited URL
3. ANSWER - synthesize final answer

DECISION CRITERIA:
- Choose ANSWER if: avg_relevance > 0.5 AND sources ≥ 2
- Choose OPEN if: unvisited high-priority URLs available
- Choose SEARCH if: results off-topic or need more specific info

For OPEN: Provide EXACT URL from search results above.
For SEARCH: Provide refined query based on gaps in current evidence.

Respond with JSON only:
{{"action": "SEARCH|OPEN|ANSWER", "target": "exact URL or refined query", "reasoning": "why this action"}}"""

        response = await self.llm.generate(prompt, max_tokens=200)

        try:
            # Parse JSON response
            if '{' in response:
                json_part = response[response.find('{'):response.rfind('}') + 1]
                action_data = json.loads(json_part)
            else:
                action_data = {"action": "ANSWER", "target": "", "reasoning": "fallback"}

            action_type = Action(action_data["action"].lower())
            target = action_data.get("target", "")

            # Enhanced URL selection for OPEN action
            if action_type == Action.OPEN:
                if not target.startswith(('http://', 'https://')):
                    # Find best unvisited URL (already prioritized)
                    available_urls = [r.url for r in state.results
                                      if r.url.startswith(('http://', 'https://'))
                                      and r.url not in state.visited_urls]

                    if available_urls:
                        target = available_urls[0]
                        print(f"🎯 Auto-selected priority URL: {target[:60]}...")
                    else:
                        action_type = Action.SEARCH
                        target = f"{state.query} {query_intent} detallado"
                        print("🔄 No unvisited URLs, switching to refined search")
                elif target in state.visited_urls:
                    available_urls = [r.url for r in state.results
                                      if r.url.startswith(('http://', 'https://'))
                                      and r.url not in state.visited_urls]

                    if available_urls:
                        target = available_urls[0]
                        print(f"🔄 URL already visited, trying: {target[:60]}...")
                    else:
                        action_type = Action.SEARCH
                        target = f"{state.query} información adicional"
                        print("🔄 All URLs visited, refining search")

            params = {"target": target}
            print(f"🤖 Agent decides: {action_type.value} - {action_data.get('reasoning', '')}")
            return action_type, params

        except Exception as e:
            print(f"Planning error: {e}")
            print(f"Raw response: {response}")

            # Smart fallback based on state
            if avg_relevance < 0.4 and state.clicks < 3:
                return Action.SEARCH, {"target": f"{state.query} guía completa"}
            elif len(state.evidence) >= 2:
                return Action.ANSWER, {"target": ""}
            else:
                # Try to open first unvisited URL
                available_urls = [r.url for r in state.results
                                  if r.url.startswith(('http://', 'https://'))
                                  and r.url not in state.visited_urls]
                if available_urls:
                    return Action.OPEN, {"target": available_urls[0]}
                else:
                    return Action.ANSWER, {"target": ""}

    async def _get_evidence_summary(self, evidence_list: List[Evidence]) -> str:
        """Generate intelligent evidence summary using LLM"""
        if not evidence_list:
            return "No evidence collected yet."

        if len(evidence_list) <= 2:
            # Simple summary for few sources
            summaries = []
            for i, e in enumerate(evidence_list, 1):
                domain = e.url.split('/')[2] if '/' in e.url else e.url
                preview = e.content[:150].replace('\n', ' ')
                summaries.append(f"{i}. {domain} (relevance: {e.relevance:.2f}): {preview}...")
            return "\n".join(summaries)

        # Use LLM for intelligent summary of multiple sources
        evidence_text = "\n\n".join([
            f"Source {i + 1} ({e.url}): {e.content[:300]}..."
            for i, e in enumerate(evidence_list[-3:])  # Last 3 for context
        ])

        prompt = f"""Summarize the key information from these sources in 2-3 bullet points:

{evidence_text}

Focus on:
- Main topics covered
- Quality and relevance of information
- Any gaps or inconsistencies

Be concise but informative."""

        summary = await self.llm.generate(prompt, max_tokens=150)
        return f"📋 Evidence Summary:\n{summary}"


# 5. Main DeepSearch System
class DeepSearch:
    """Orchestrates LLM, search environment, planner, and Thompson bandit for deep web research."""

    def __init__(self):
        self.llm = LocalLLM()
        self.env = SearchEnv()
        self.agent = PlannerAgent(self.llm)

        # RL component: strategies for search query formulation
        self.strategies = [
            "direct",  # Use query as-is
            "specific",  # Add specific terms like "2024", "recent"
            "academic",  # Add "study", "research", "data"
            "news"  # Add "news", "latest", "current"
        ]
        self.bandit = ThompsonBandit(self.strategies)

    def _apply_strategy(self, query: str, strategy: str) -> str:
        """Augment the query string according to the selected bandit strategy."""
        if strategy == "specific":
            return f"{query} 2024 recent data"
        elif strategy == "academic":
            return f"{query} study research statistics"
        elif strategy == "news":
            return f"{query} news latest current"
        else:
            return query  # direct

    async def search(self, query: str) -> Dict[str, Any]:
        """Execute complete search session"""
        print(f"\n🎯 Starting search for: '{query}'")

        state = self.env.reset(query)
        episode_reward = 0.0
        strategy_used = self.bandit.select_strategy()

        for step in range(10):  # Max 10 steps per episode
            print(f"\n--- Step {step + 1} ---")

            action, params = await self.agent.plan_next_action(state, self.env.max_clicks)

            if action == Action.SEARCH:
                search_query = params["target"] or query
                if step == 0:  # Apply strategy to first search
                    search_query = self._apply_strategy(search_query, strategy_used)
                    print(f"Using strategy '{strategy_used}': {search_query}")

                state, reward = self.env.search(state, search_query)

            elif action == Action.OPEN:
                target_url = params["target"]
                if not target_url and state.results:
                    target_url = state.results[0].url
                state, reward = self.env.open_url(state, target_url)

            elif action == Action.ANSWER:
                answer = await self._synthesize_answer(state)

                # Evaluate final answer quality
                final_reward = await self._evaluate_answer(query, answer, state.evidence)

                # Update bandit
                total_reward = episode_reward + final_reward
                self.bandit.update(strategy_used, total_reward)

                return {
                    "query": query,
                    "answer": answer,
                    "evidence_count": len(state.evidence),
                    "clicks_used": state.clicks,
                    "tokens_used": state.tokens,
                    "final_reward": final_reward,
                    "strategy_used": strategy_used,
                    "strategy_stats": self.bandit.get_stats()
                }

            episode_reward += reward

            if state.clicks >= self.env.max_clicks:
                print("Max clicks reached, ending search")
                break

        # Fallback if no ANSWER action was taken
        answer = await self._synthesize_answer(state)
        return {
            "query": query,
            "answer": answer,
            "evidence_count": len(state.evidence),
            "clicks_used": state.clicks,
            "tokens_used": state.tokens,
            "final_reward": 0.5,
            "strategy_used": strategy_used
        }

    async def _synthesize_answer(self, state: State) -> str:
        """Generate final answer from collected evidence"""
        if not state.evidence:
            # Special handling for time queries
            if any(word in state.query.lower() for word in ['hora', 'time', 'tiempo', 'clock']):
                prompt = f"""The user asked: {state.query}

This appears to be asking about current time. Even without direct evidence, provide a helpful response explaining:
1. What timezone Alberta, Canada is in
2. How to find current time
3. Any relevant time zone information

Be helpful and informative."""

                return await self.llm.generate(prompt, max_tokens=300)

            return "No sufficient evidence found to answer the query. Please try rephrasing your question or searching for more specific terms."

        evidence_text = "\n\n".join([
            f"Source: {e.url}\nContent: {e.content[:800]}..."
            for e in sorted(state.evidence, key=lambda x: x.relevance, reverse=True)[:3]
        ])

        prompt = f"""Based on the evidence below, provide a comprehensive answer to: {state.query}

EVIDENCE:
{evidence_text}

Provide a well-structured answer with key facts. Be concise but thorough."""

        answer = await self.llm.generate(prompt, max_tokens=400)
        print(f"📝 Generated answer: {answer[:200]}...")
        return answer

    async def _evaluate_answer(self, query: str, answer: str, evidence: List[Evidence]) -> float:
        """Enhanced answer quality evaluation using LLM"""
        if len(answer) < 50:
            return 0.2

        # Use LLM for intelligent evaluation
        evidence_summary = "\n".join([
            f"- {e.url}: {e.content[:200]}... (relevance: {e.relevance:.2f})"
            for e in evidence[:3]
        ])

        evaluation_prompt = f"""Evaluate the quality of this answer for the given query on a scale of 0.0 to 1.0.

QUERY: {query}

ANSWER: {answer}

EVIDENCE USED:
{evidence_summary}

Evaluation criteria:
1. Accuracy: Are the facts correct and verifiable?
2. Completeness: Does it fully address the query?
3. Relevance: Is the information directly related to the question?
4. Clarity: Is it well-structured and easy to understand?
5. Evidence support: Is the answer backed by the provided evidence?

Provide ONLY a score between 0.0 and 1.0, followed by a brief reason.
Format: "0.X - reason"

Score:"""

        try:
            response = await self.llm.generate(evaluation_prompt, max_tokens=100)

            # Extract score from response
            score_match = re.search(r'(\d+\.\d+)', response)
            if score_match:
                llm_score = float(score_match.group(1))
                llm_score = max(0.0, min(1.0, llm_score))
                print(
                    f"🎯 LLM evaluation: {llm_score:.3f} - {response.split('-', 1)[-1].strip() if '-' in response else 'Good quality'}")
            else:
                llm_score = 0.5
                print(f"⚠️ Could not parse LLM score, using default")
        except Exception as e:
            print(f"⚠️ LLM evaluation failed: {e}")
            llm_score = 0.5

        # Combine with heuristic metrics
        heuristic_score = self._heuristic_evaluation(query, answer, evidence)

        # Weighted combination (70% LLM, 30% heuristics)
        final_score = 0.7 * llm_score + 0.3 * heuristic_score

        print(f"📊 Final score: {final_score:.3f} (LLM: {llm_score:.3f}, Heuristic: {heuristic_score:.3f})")
        return final_score

    def _heuristic_evaluation(self, query: str, answer: str, evidence: List[Evidence]) -> float:
        """Backup heuristic evaluation"""
        # Answer length score
        answer_length_score = min(len(answer.split()) / 150, 1.0)

        # Evidence quality score
        evidence_quality_score = sum(e.relevance for e in evidence) / max(len(evidence), 1) if evidence else 0

        # Query coverage score
        query_words = set(query.lower().split())
        answer_words = set(answer.lower().split())
        coverage_score = len(query_words & answer_words) / max(len(query_words), 1)

        # Structural quality (basic checks)
        structure_score = 0.5
        if len(answer) > 100:
            structure_score += 0.2
        if any(marker in answer for marker in [':', '-', '•', '1.', '2.', 'Day 1', 'Day 2']):
            structure_score += 0.2  # Well-structured
        if len(evidence) >= 2:
            structure_score += 0.1  # Multiple sources

        structure_score = min(1.0, structure_score)

        return (answer_length_score + evidence_quality_score + coverage_score + structure_score) / 4


# 6. Demo and Testing
async def main():
    """Run a demo search session with a set of test queries."""
    print("🚀 DeepSearch Starter - Local vLLM Edition")
    print("=" * 50)

    search_system = DeepSearch()

    # Test queries
    queries = [
        "What is the current inflation rate in Argentina?",
        "How many users does TikTok have in 2024?",
        "What are the latest developments in quantum computing?"
    ]

    for i, query in enumerate(queries, 1):
        print(f"\n{'=' * 60}")
        print(f"TEST {i}: {query}")
        print('=' * 60)

        result = await search_system.search(query)

        print("\n📋 RESULTS:")
        print(f"Answer: {result['answer'][:300]}...")
        print(f"Evidence sources: {result['evidence_count']}")
        print(f"Resources: {result['clicks_used']} clicks, {result['tokens_used']} tokens")
        print(f"Quality score: {result['final_reward']:.3f}")
        print(f"Strategy used: {result['strategy_used']}")

        if 'strategy_stats' in result:
            print("\n📈 Strategy Performance:")
            for strategy, score in result['strategy_stats'].items():
                print(f"  {strategy}: {score:.3f}")

        # Wait between tests
        await asyncio.sleep(2)


if __name__ == "__main__":
    # First test the vLLM connection
    async def test_llm():
        llm = LocalLLM()
        response = await llm.generate("Hello! Please respond with 'Connection successful!'")
        print(f"LLM Test: {response}")
        return "successful" in response.lower()


    async def run_tests():
        print("Testing vLLM connection...")
        if await test_llm():
            print("✅ vLLM connection working!")
            await main()
        else:
            print("❌ vLLM connection failed. Check your server and credentials.")


    asyncio.run(run_tests())