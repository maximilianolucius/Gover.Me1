"""Debug utility for inspecting raw DuckDuckGo search result fields.

Runs a sample query and prints every key-value pair returned by the
duckduckgo_search library so developers can verify field names and formats.
"""

# Debug script to check DuckDuckGo search results format
# Run this to see what fields are available in search results

from duckduckgo_search import DDGS
import json


def debug_search_results():
    """Execute a test query and print all fields of each result."""
    ddgs = DDGS()
    query = "current time Alberta Canada"

    print(f"Testing search query: '{query}'")
    print("=" * 50)

    try:
        results = list(ddgs.text(query, max_results=3))
        print(f"Found {len(results)} results\n")

        for i, result in enumerate(results, 1):
            print(f"Result {i}:")
            print("-" * 30)

            # Print all available fields
            for key, value in result.items():
                if isinstance(value, str) and len(value) > 100:
                    print(f"  {key}: {value[:100]}...")
                else:
                    print(f"  {key}: {value}")

            print()

            # Check URL field specifically
            url = result.get('href') or result.get('link') or result.get('url')
            print(f"  📎 Extracted URL: {url}")
            print(f"  ✅ Valid URL: {url.startswith(('http://', 'https://')) if url else False}")
            print()

    except Exception as e:
        print(f"Error: {e}")


if __name__ == "__main__":
    debug_search_results()