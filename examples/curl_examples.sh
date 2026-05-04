#!/bin/bash
# BrainOS — cURL Examples
# Run: chmod +x examples/curl_examples.sh && ./examples/curl_examples.sh

BASE="http://localhost:8000"

echo ""
echo "=========================================="
echo "  BrainOS API — cURL Examples"
echo "=========================================="

# Health
echo ""
echo "--- Health ---"
curl -s "$BASE/health" | python3 -m json.tool

# Ingest text
echo ""
echo "--- Ingest Text ---"
curl -s -X POST "$BASE/ingest" \
  -H "Content-Type: application/json" \
  -d '{
    "content": "The Pomodoro Technique is a time management method where you work for 25 minutes then take a 5-minute break. After 4 cycles, take a longer 15-30 minute break. Great for deep focus work.",
    "input_type": "text",
    "source": "productivity-notes"
  }' | python3 -m json.tool

# Ingest URL
echo ""
echo "--- Ingest URL ---"
curl -s -X POST "$BASE/ingest" \
  -H "Content-Type: application/json" \
  -d '{
    "content": "https://en.wikipedia.org/wiki/Pomodoro_Technique",
    "input_type": "url"
  }' | python3 -m json.tool

# Ask a question
echo ""
echo "--- Ask Brain ---"
curl -s -X POST "$BASE/ask" \
  -H "Content-Type: application/json" \
  -d '{
    "query": "What is the Pomodoro Technique and how does it work?",
    "top_k": 3
  }' | python3 -m json.tool

# Daily summary
echo ""
echo "--- Daily Summary ---"
curl -s "$BASE/summary" | python3 -m json.tool

# Upload a PDF (replace path with your file)
# echo ""
# echo "--- Ingest PDF ---"
# curl -s -X POST "$BASE/ingest/pdf" \
#   -F "file=@/path/to/your/document.pdf" | python3 -m json.tool

# Upload voice note (replace path with your audio)
# echo ""
# echo "--- Ingest Voice ---"
# curl -s -X POST "$BASE/ingest/voice" \
#   -F "file=@/path/to/recording.mp3" | python3 -m json.tool

# Recent memories
echo ""
echo "--- Recent Memories ---"
curl -s "$BASE/memory/recent?n=5" | python3 -m json.tool

# Stats
echo ""
echo "--- Stats ---"
curl -s "$BASE/stats" | python3 -m json.tool

echo ""
echo "Done!"
