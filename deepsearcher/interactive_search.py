"""Interactive CLI for the adaptive deep search engine.

Provides a REPL where users can enter queries, adjust quality thresholds
and iteration limits, and review detailed quality-metric breakdowns of
each search session.
"""

import asyncio
import sys
import os
import time
from deepsearcher.adaptive_deepsearch import AdaptiveSearchEngine


async def interactive_adaptive_search():
    """Main REPL loop: read queries, run adaptive search, display results."""
    print("🔍 Adaptive DeepSearch Interactive Mode")
    print("Advanced search with continuous quality optimization")
    print("Type 'quit' to exit or 'config' to adjust settings\n")

    # Initialize with default settings
    search_engine = AdaptiveSearchEngine(
        quality_threshold=0.75,
        max_iterations=12,
        plateau_tolerance=3
    )

    session_stats = {
        'queries': 0,
        'total_time': 0,
        'avg_quality': 0,
        'total_iterations': 0
    }

    while True:
        query = input("🎯 Enter search query: ").strip()

        if query.lower() in ['quit', 'exit', 'q']:
            print_session_summary(session_stats)
            print("Goodbye!")
            break

        if query.lower() == 'config':
            search_engine = configure_engine()
            continue

        if not query:
            continue

        print(f"\n🚀 Searching: '{query}'")
        print("🔄 Starting adaptive optimization loop...")
        print("-" * 60)

        try:
            start_time = time.time()
            result = await search_engine.search_with_feedback_loop(query)
            duration = time.time() - start_time

            # Update session stats
            session_stats['queries'] += 1
            session_stats['total_time'] += duration
            session_stats['avg_quality'] = (
                    (session_stats['avg_quality'] * (session_stats['queries'] - 1) +
                     result['quality_score']) / session_stats['queries']
            )
            session_stats['total_iterations'] += result['iterations']

            print_result(result, duration)

        except KeyboardInterrupt:
            print("\n⏸️  Search interrupted by user")
        except Exception as e:
            print(f"❌ Error: {e}")
            import traceback
            traceback.print_exc()

        print("\n" + "=" * 70 + "\n")


def print_result(result, duration):
    """Print formatted search results"""
    print(f"\n📝 FINAL ANSWER:")
    print("-" * 40)
    print(result['final_answer'])

    print(f"\n📊 QUALITY METRICS:")
    print("-" * 40)
    quality_breakdown = result['quality_breakdown']

    for metric, score in quality_breakdown.items():
        if metric == 'overall':
            continue
        status = get_status_icon(score)
        bar = get_progress_bar(score)
        print(f"{status} {metric.capitalize():12}: {bar} {score:.3f}")

    overall_score = quality_breakdown['overall']
    overall_status = get_status_icon(overall_score)
    overall_bar = get_progress_bar(overall_score)
    print(f"{overall_status} {'Overall':12}: {overall_bar} {overall_score:.3f}")

    print(f"\n🔄 PROCESS STATS:")
    print("-" * 40)
    print(f"• Iterations: {result['iterations']}")
    print(f"• Sources found: {result['sources_found']}")
    print(f"• Searches performed: {result['searches_performed']}")
    print(f"• Termination reason: {result['termination_reason']}")
    print(f"• Quality trend: {result['quality_trend']:+.4f}")
    print(f"• Duration: {duration:.1f}s")

    if result['evidence_sources']:
        print(f"\n🔗 SOURCES:")
        print("-" * 40)
        for i, url in enumerate(result['evidence_sources'][:5], 1):
            domain = url.split('/')[2] if '/' in url else url
            print(f"{i}. {domain}")
        if len(result['evidence_sources']) > 5:
            print(f"   ... and {len(result['evidence_sources']) - 5} more")


def get_status_icon(score):
    """Get status icon based on score"""
    if score >= 0.75:
        return "✅"
    elif score >= 0.5:
        return "⚠️"
    else:
        return "❌"


def get_progress_bar(score, width=15):
    """Generate progress bar for score"""
    filled = int(score * width)
    bar = "█" * filled + "░" * (width - filled)
    return f"[{bar}]"


def configure_engine():
    """Configure search engine parameters"""
    print("\n⚙️  Configuration Menu")
    print("-" * 30)

    try:
        quality_threshold = float(input(f"Quality threshold (0.5-1.0) [0.75]: ") or 0.75)
        quality_threshold = max(0.5, min(1.0, quality_threshold))

        max_iterations = int(input(f"Max iterations (5-20) [12]: ") or 12)
        max_iterations = max(5, min(20, max_iterations))

        plateau_tolerance = int(input(f"Plateau tolerance (2-5) [3]: ") or 3)
        plateau_tolerance = max(2, min(5, plateau_tolerance))

        print(f"\n✅ Configuration updated:")
        print(f"   Quality threshold: {quality_threshold}")
        print(f"   Max iterations: {max_iterations}")
        print(f"   Plateau tolerance: {plateau_tolerance}")

        return AdaptiveSearchEngine(
            quality_threshold=quality_threshold,
            max_iterations=max_iterations,
            plateau_tolerance=plateau_tolerance
        )

    except ValueError:
        print("❌ Invalid input, using defaults")
        return AdaptiveSearchEngine()


def print_session_summary(stats):
    """Print session statistics"""
    if stats['queries'] == 0:
        return

    print(f"\n📈 SESSION SUMMARY")
    print("-" * 30)
    print(f"• Queries processed: {stats['queries']}")
    print(f"• Total time: {stats['total_time']:.1f}s")
    print(f"• Average quality: {stats['avg_quality']:.3f}")
    print(f"• Average iterations: {stats['total_iterations'] / stats['queries']:.1f}")
    print(f"• Time per query: {stats['total_time'] / stats['queries']:.1f}s")


def print_help():
    """Print help information"""
    print("""
🔍 Adaptive DeepSearch Commands:

• Enter any question to start search
• 'config' - Adjust search parameters  
• 'quit' - Exit program

🎯 How it works:
• Continuously evaluates response quality
• Automatically refines search strategy
• Stops when quality threshold reached
• Detects when further search is futile

📊 Quality Metrics:
• Completeness - Answers the full question
• Accuracy - Information is correct
• Consistency - Sources agree
• Depth - Sufficient detail level
• Freshness - Recent information
• Authority - Reliable sources
""")


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] in ['-h', '--help']:
        print_help()
        sys.exit(0)

    try:
        asyncio.run(interactive_adaptive_search())
    except KeyboardInterrupt:
        print("\n👋 Search interrupted. Goodbye!")
    except Exception as e:
        print(f"💥 Fatal error: {e}")
        sys.exit(1)