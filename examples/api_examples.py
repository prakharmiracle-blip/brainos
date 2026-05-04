"""
BrainOS — Example API Requests
Run these after starting the server with: python main.py
"""

import requests
import json

BASE = "http://localhost:8000"


def pp(data):
    print(json.dumps(data, indent=2))


# ============================================================
# 1. Health check
# ============================================================
print("\n--- Health Check ---")
r = requests.get(f"{BASE}/health")
pp(r.json())


# ============================================================
# 2. Ingest a text note
# ============================================================
print("\n--- Ingest Text Note ---")
r = requests.post(f"{BASE}/ingest", json={
    "content": "Today I learned that Transformers use self-attention mechanisms to process sequences in parallel. The key innovation is the attention formula: softmax(QK^T / sqrt(d_k)) * V. This allows the model to relate positions within the sequence regardless of distance.",
    "input_type": "text",
    "source": "study-session",
    "metadata": {"topic": "deep-learning", "tags": ["transformers", "attention"]}
})
pp(r.json())


# ============================================================
# 3. Ingest a URL
# ============================================================
print("\n--- Ingest URL ---")
r = requests.post(f"{BASE}/ingest", json={
    "content": "https://en.wikipedia.org/wiki/Retrieval-augmented_generation",
    "input_type": "url",
})
pp(r.json())


# ============================================================
# 4. Ingest another note
# ============================================================
print("\n--- Ingest Second Note ---")
r = requests.post(f"{BASE}/ingest", json={
    "content": "Meeting notes: Team agreed to ship the v2 API by Friday. Need to finalize rate limiting strategy. John will handle auth, Sarah takes the database migrations. Follow up on Monday.",
    "input_type": "text",
    "source": "meeting-notes",
    "metadata": {"project": "api-v2"}
})
pp(r.json())


# ============================================================
# 5. Ingest a voice note (audio file)
# ============================================================
print("\n--- Ingest Voice (skip if no audio file) ---")
# Uncomment and set path to test:
# with open("my_recording.mp3", "rb") as f:
#     r = requests.post(f"{BASE}/ingest/voice", files={"file": f})
#     pp(r.json())


# ============================================================
# 6. Ask your Brain
# ============================================================
print("\n--- Ask: Transformers ---")
r = requests.post(f"{BASE}/ask", json={
    "query": "What did I learn about transformers and attention?",
    "top_k": 3
})
data = r.json()
print("Answer:", data["answer"])
print("\nSources:")
for s in data["sources"]:
    print(f"  [{s['input_type']}] score={s['score']} | {s['content'][:80]}...")


# ============================================================
# 7. Ask about meetings
# ============================================================
print("\n--- Ask: Meeting tasks ---")
r = requests.post(f"{BASE}/ask", json={
    "query": "What tasks were assigned in my meetings?",
})
print("Answer:", r.json()["answer"])


# ============================================================
# 8. Daily Summary
# ============================================================
print("\n--- Daily Summary ---")
r = requests.get(f"{BASE}/summary")
data = r.json()
print(f"Date: {data['date']}")
print("\nInsights:")
for i in data["insights"]:
    print(f"  • {i}")
print("\nKey Learnings:")
for l in data["key_learnings"]:
    print(f"  • {l}")
print("\nImportant Notes:")
for n in data["important_notes"]:
    print(f"  • {n}")


# ============================================================
# 9. Recent memories
# ============================================================
print("\n--- Recent Memories (last 3) ---")
r = requests.get(f"{BASE}/memory/recent?n=3")
data = r.json()
print(f"Total chunks: {data['count']}")
for e in data["entries"]:
    print(f"  [{e.get('input_type')}] {e.get('timestamp', '')[:19]} | {e.get('content', '')[:80]}...")


# ============================================================
# 10. Stats
# ============================================================
print("\n--- Stats ---")
r = requests.get(f"{BASE}/stats")
pp(r.json())
