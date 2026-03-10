"""Deduplication tool for ElevenLabs conversation JSON exports.

Compares two date-based conversation files and removes entries from the
newer file that already appear in the older one, based on conversation IDs.
"""

import json
from typing import Set, Dict, List


def load_json_file(filename: str) -> Dict:
    """Load and parse JSON file"""
    with open(filename, 'r', encoding='utf-8') as f:
        return json.load(f)


def save_json_file(data: Dict, filename: str) -> None:
    """Save data to JSON file"""
    with open(filename, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def extract_conversation_ids(data: Dict) -> Set[str]:
    """Extract all conversation_ids from the conversations data"""
    conversation_ids = set()
    for conv in data.get('conversations', []):
        conv_id = conv.get('conversation_id')
        if conv_id:
            conversation_ids.add(conv_id)
    return conversation_ids


def remove_duplicate_conversations(current_file: str, previous_file: str, output_file: str = None):
    """
    Remove conversations from current_file that already exist in previous_file

    Args:
        current_file: Path to the current day's JSON file (e.g., 'data/8G3toaZ2Tyu7HlUaWRO8_2025-08-27.json')
        previous_file: Path to the previous day's JSON file (e.g., 'data/8G3toaZ2Tyu7HlUaWRO8_2025-08-26.json')
        output_file: Optional output file path. If None, overwrites current_file
    """

    print(f"Loading {current_file}...")
    current_data = load_json_file(current_file)

    print(f"Loading {previous_file}...")
    previous_data = load_json_file(previous_file)

    # Extract conversation IDs from previous day
    previous_conv_ids = extract_conversation_ids(previous_data)
    print(f"Found {len(previous_conv_ids)} conversation IDs in previous file")

    # Filter out duplicate conversations from current data
    original_conversations = current_data.get('conversations', [])
    filtered_conversations = []
    removed_count = 0

    for conv in original_conversations:
        conv_id = conv.get('conversation_id')
        if conv_id not in previous_conv_ids:
            filtered_conversations.append(conv)
        else:
            removed_count += 1
            print(f"Removing duplicate: {conv_id}")

    # Update the data
    current_data['conversations'] = filtered_conversations
    current_data['total_conversations'] = len(filtered_conversations)

    # Save the updated file
    output_path = output_file if output_file else current_file
    save_json_file(current_data, output_path)

    print(f"\n✅ Results:")
    print(f"   - Original conversations: {len(original_conversations)}")
    print(f"   - Removed duplicates: {removed_count}")
    print(f"   - Remaining conversations: {len(filtered_conversations)}")
    print(f"   - Saved to: {output_path}")


if __name__ == "__main__":
    # Configuration
    AGENT_ID = "8G3toaZ2Tyu7HlUaWRO8"
    CURRENT_FILE = f"data/{AGENT_ID}_2025-08-27.json"
    PREVIOUS_FILE = f"data/{AGENT_ID}_2025-08-26.json"

    try:
        remove_duplicate_conversations(CURRENT_FILE, PREVIOUS_FILE)
    except FileNotFoundError as e:
        print(f"❌ Error: File not found - {e}")
        print("Make sure both JSON files exist in the current directory")
    except Exception as e:
        print(f"❌ Error: {e}")