"""ElevenLabs Conversational AI history client.

Downloads and saves conversation transcripts for a given agent from the
ElevenLabs API, with support for date-range filtering and pagination.
"""

import os
import requests
import json
from typing import List, Dict, Optional
from datetime import datetime
import time


class ElevenLabsAgentHistory:
    """Client for the ElevenLabs Conversational AI conversation history API."""

    def __init__(self, api_key: str):
        self.api_key = api_key
        self.base_url = "https://api.elevenlabs.io/v1"
        self.headers = {
            "xi-api-key": api_key,
            "Content-Type": "application/json"
        }

    def list_conversations(self, agent_id: str, page_size: int = 30,
                           start_time: Optional[int] = None,
                           end_time: Optional[int] = None,
                           user_id: Optional[str] = None) -> Dict:
        """
        List all conversations for a specific agent

        Args:
            agent_id: The ID of the agent
            page_size: Number of conversations to return (max 100)
            start_time: Unix timestamp to filter conversations after this date
            end_time: Unix timestamp to filter conversations up to this date
            user_id: Filter conversations by initiating user ID
        """
        params = {
            "agent_id": agent_id,
            "page_size": page_size
        }

        if start_time:
            params["start_time"] = start_time
        if end_time:
            params["end_time"] = end_time
        if user_id:
            params["user_id"] = user_id

        response = requests.get(
            f"{self.base_url}/convai/conversations",
            headers=self.headers,
            params=params
        )
        response.raise_for_status()
        return response.json()

    def get_conversation_details(self, conversation_id: str) -> Dict:
        """
        Get detailed information about a specific conversation including transcript

        Args:
            conversation_id: The ID of the conversation
        """
        response = requests.get(
            f"{self.base_url}/convai/conversations/{conversation_id}",
            headers=self.headers
        )
        response.raise_for_status()
        return response.json()

    def get_conversations_from_date(self, agent_id: str, start_date: str) -> List[Dict]:
        """
        Get all conversations for an agent from a specific date onwards

        Args:
            agent_id: The ID of the agent
            start_date: Date in format 'YYYY-MM-DD' (e.g., '2025-08-25')
        """
        # Convert date to Unix timestamp (start of day)
        start_datetime = datetime.strptime(start_date, '%Y-%m-%d')
        start_timestamp = int(start_datetime.timestamp())

        print(f"Searching for conversations from {start_date} onwards...")
        print(f"Unix timestamp: {start_timestamp}")

        all_conversations = []
        cursor = None

        while True:
            # Get list of conversations with date filter
            params = {
                "agent_id": agent_id,
                "page_size": 100,
                "start_time": start_timestamp
            }
            if cursor:
                params["cursor"] = cursor

            response = requests.get(
                f"{self.base_url}/convai/conversations",
                headers=self.headers,
                params=params
            )
            response.raise_for_status()
            data = response.json()

            conversations = data.get("conversations", [])
            print(f"Found {len(conversations)} conversations in this batch")

            # Get detailed info for all conversations
            for conversation in conversations:
                conv_id = conversation["conversation_id"]

                # Get detailed conversation info
                try:
                    details = self.get_conversation_details(conv_id)
                    status = details.get("status", "unknown")
                    all_conversations.append(details)
                    print(f"✓ Added conversation: {conv_id} (Status: {status})")

                except Exception as e:
                    print(f"Error getting details for conversation {conv_id}: {e}")
                    continue

            # Check if there are more pages
            if not data.get("has_more", False):
                break
            cursor = data.get("next_cursor")

        print(f"\nTotal conversations found: {len(all_conversations)}")
        return all_conversations

    def extract_transcript_text(self, conversation_details: Dict) -> str:
        """
        Extract the full transcript text from conversation details

        Args:
            conversation_details: The detailed conversation data
        """
        transcript_parts = []

        for item in conversation_details.get("transcript", []):
            role = item.get("role", "unknown")
            message = item.get("message", "")
            timestamp = item.get("timestamp", "")

            transcript_parts.append(f"[{timestamp}] {role.upper()}: {message}")

        return "\n".join(transcript_parts)

    def save_conversations_from_date(self, agent_id: str, start_date: str, filename: str = None):
        """
        Save all conversation history for an agent from a specific date to a JSON file

        Args:
            agent_id: The ID of the agent
            start_date: Date in format 'YYYY-MM-DD'
            filename: Optional filename, defaults to data/agent_id_YYYY-MM-DD.json
        """
        if not filename:
            filename = f"data/{agent_id}_{start_date}.json"

        conversations = self.get_conversations_from_date(agent_id, start_date)

        # Create summary information
        summary = {
            "agent_id": agent_id,
            "filter_date": start_date,
            "total_conversations": len(conversations),
            "extraction_date": datetime.now().isoformat(),
            "conversations": conversations
        }

        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)

        print(f"\n📁 Saved {len(conversations)} conversations to {filename}")

        # Print summary statistics
        if conversations:
            print(f"\n📊 Summary:")
            statuses = {}
            for conv in conversations:
                status = conv.get('status', 'unknown')
                statuses[status] = statuses.get(status, 0) + 1

            for status, count in statuses.items():
                print(f"   - {status}: {count} conversations")

        return conversations


# Usage example for your specific requirements
if __name__ == "__main__":
    # Configuration
    ELEVENLABS_API_KEY = os.getenv("ELEVENLABS_API_KEY", "sk_e38cc2b727b1785beda86dd43c056ab5f9847cba571153ee")
    AGENT_ID = "8G3toaZ2Tyu7HlUaWRO8"
    START_DATE = "2025-08-27"  # From 2025-08-25 onwards

    # Initialize the client
    client = ElevenLabsAgentHistory(ELEVENLABS_API_KEY)

    try:
        print(f"🤖 Downloading conversations for agent: {AGENT_ID}")
        print(f"📅 From date: {START_DATE} onwards")
        print(f"🔑 Using API key: {ELEVENLABS_API_KEY[:10]}...")
        print("-" * 60)

        # Download and save all conversations from the specified date
        all_conversations = client.save_conversations_from_date(
            AGENT_ID,
            START_DATE
        )

        # Optional: Print details of first conversation if available
        if all_conversations:
            print(f"\n📋 Sample conversation details:")
            first_conv = all_conversations[0]
            print(f"   Conversation ID: {first_conv['conversation_id']}")
            print(f"   Status: {first_conv['status']}")
            print(f"   Has Audio: {first_conv.get('has_audio', False)}")

            # Extract and print first few lines of transcript
            transcript = client.extract_transcript_text(first_conv)
            if transcript:
                lines = transcript.split('\n')[:3]  # First 3 lines
                print(f"   Transcript preview:")
                for line in lines:
                    print(f"     {line}")
                if len(transcript.split('\n')) > 3:
                    n = len(transcript.split('\n')) - 3
                    print(f"     ... (and {n} more lines)")
        else:
            print("❌ No conversations found for the specified criteria.")

    except requests.exceptions.RequestException as e:
        print(f"❌ API Error: {e}")
    except Exception as e:
        print(f"❌ Error: {e}")

# conv_2401k3ef2etffzdvx0nqws1jrvnz