"""Simple launcher for the DeepSearch interactive mode.

Starts a REPL that accepts free-text queries, runs the DeepSearch pipeline,
and prints answers with quality and strategy statistics.
"""

import asyncio
import sys
import os

from .deepsearch import DeepSearch


async def interactive_search():
    """REPL loop: read a query, run DeepSearch, and display the answer."""
    print("🔍 DeepSearch Interactive Mode")
    print("Type 'quit' to exit\n")

    search_system = DeepSearch()

    while True:
        query = input("Enter your search query: ").strip()

        if query.lower() in ['quit', 'exit', 'q']:
            print("Goodbye!")
            break

        if not query:
            continue

        print(f"\n🎯 Searching for: '{query}'")
        print("-" * 50)

        try:
            result = await search_system.search(query)

            print(f"\n📝 ANSWER:")
            print(result['answer'])

            print(f"\n📊 STATS:")
            print(f"• Evidence sources: {result['evidence_count']}")
            print(f"• Clicks used: {result['clicks_used']}")
            print(f"• Tokens used: {result['tokens_used']}")
            print(f"• Quality score: {result['final_reward']:.3f}")
            print(f"• Strategy: {result['strategy_used']}")

            if 'strategy_stats' in result:
                print("\n📈 Strategy Performance:")
                for strategy, score in result['strategy_stats'].items():
                    bar = "█" * int(score * 20)
                    print(f"  {strategy:10}: {bar} {score:.3f}")

        except Exception as e:
            print(f"❌ Error: {e}")

        print("\n" + "=" * 60 + "\n")


if __name__ == "__main__":
    asyncio.run(interactive_search())