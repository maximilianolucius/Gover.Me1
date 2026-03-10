history_1.py
history_clean_0.py

simplify_conversation.py

analiz

sleep 21600; while true; do python encuestadora/history_1.py; python encuestadora/history_clean_0.py; python encuestadora/simplify_conversations.py; python encuestadora/analyze_conversations_1.py; sleep 60; done
