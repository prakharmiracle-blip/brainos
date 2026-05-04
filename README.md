# 🧠 BrainOS — Phase 1 MVP

> **AI-powered personal memory and intelligence system.**
> Capture everything. Query anything. Understand yourself.

---

## What is BrainOS?

BrainOS is a personal AI brain that:
- **Ingests** your notes, voice memos, URLs, and PDFs
- **Stores** them as vector embeddings (fully local with FAISS)
- **Retrieves** the most relevant memories when you ask a question
- **Answers** questions using only YOUR data (RAG — no hallucinations)
- **Summarises** your day with structured insights

---

## Architecture

```
User Input (text / voice / URL / PDF)
        ↓
  Ingestion Pipeline
        ↓
  Text Processor (clean → chunk)
        ↓
  Embedding Engine (sentence-transformers, local)
        ↓
  FAISS Vector Store (persisted to disk)
        ↓
  ┌─────────────────────────────────┐
  │         FastAPI Backend         │
  │                                 │
  │  /ingest   /ask   /summary      │
  └─────────────────────────────────┘
        ↓               ↓
   Memory Retriever   Summary Agent
   (top-k similarity)
        ↓
   LLM (OpenAI / Together / Anthropic)
        ↓
   Grounded Answer / Daily Summary
```

---

## Project Structure

```
brainos/
├── main.py                    # Entry point
├── requirements.txt
├── .env.example               # Copy to .env
├── data/
│   ├── faiss_index/           # Persisted vector store
│   └── uploads/               # Temp audio/PDF uploads
├── logs/
├── examples/
│   ├── api_examples.py        # Python API examples
│   └── curl_examples.sh       # cURL examples
└── src/
    ├── config.py              # Centralised settings
    ├── ingestion/
    │   ├── pipeline.py        # Master ingestion pipeline
    │   ├── voice.py           # Whisper transcription
    │   └── document.py        # URL + PDF extraction
    ├── processing/
    │   ├── text_processor.py  # Clean + chunk text
    │   └── embedder.py        # sentence-transformers
    ├── memory/
    │   ├── vector_store.py    # FAISS + metadata store
    │   └── retriever.py       # Similarity search + context builder
    ├── agents/
    │   ├── chat_agent.py      # RAG Q&A agent
    │   └── summary_agent.py   # Daily summary agent
    ├── api/
    │   └── app.py             # FastAPI routes
    └── utils/
        ├── logger.py          # Loguru logging
        ├── llm_client.py      # LLM abstraction
        └── models.py          # Pydantic models
```

---

## Setup

### 1. Clone / create the project

```bash
git clone <your-repo> brainos
cd brainos
```

### 2. Create virtual environment

```bash
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

> **Note on Whisper**: If you want voice support, also run:
> ```bash
> pip install openai-whisper
> # You may also need ffmpeg:
> # macOS:  brew install ffmpeg
> # Ubuntu: sudo apt install ffmpeg
> # Windows: https://ffmpeg.org/download.html
> ```

### 4. Configure environment

```bash
cp .env.example .env
# Edit .env with your API key and preferences
```

Minimum required in `.env`:

```env
LLM_PROVIDER=openai
OPENAI_API_KEY=sk-your-key-here
```

For **Together AI** (free tier available):
```env
LLM_PROVIDER=together
TOGETHER_API_KEY=your-key
LLM_MODEL=mistralai/Mixtral-8x7B-Instruct-v0.1
```

For **Anthropic Claude**:
```env
LLM_PROVIDER=anthropic
ANTHROPIC_API_KEY=your-key
LLM_MODEL=claude-3-haiku-20240307
```

### 5. Run the server

```bash
python main.py
```

Server starts at: **http://localhost:8000**

Interactive docs at: **http://localhost:8000/docs**

---

## API Reference

### `GET /health`
Check if the server is running.

```bash
curl http://localhost:8000/health
```

---

### `POST /ingest` — Text or URL

```bash
# Text note
curl -X POST http://localhost:8000/ingest \
  -H "Content-Type: application/json" \
  -d '{
    "content": "The human brain has about 86 billion neurons.",
    "input_type": "text",
    "source": "biology-study"
  }'

# URL
curl -X POST http://localhost:8000/ingest \
  -H "Content-Type: application/json" \
  -d '{
    "content": "https://en.wikipedia.org/wiki/Artificial_intelligence",
    "input_type": "url"
  }'
```

---

### `POST /ingest/voice` — Audio File

```bash
curl -X POST http://localhost:8000/ingest/voice \
  -F "file=@recording.mp3"
```

---

### `POST /ingest/pdf` — PDF File

```bash
curl -X POST http://localhost:8000/ingest/pdf \
  -F "file=@document.pdf"
```

---

### `POST /ask` — Query Your Brain

```bash
curl -X POST http://localhost:8000/ask \
  -H "Content-Type: application/json" \
  -d '{
    "query": "What did I learn about neurons?",
    "top_k": 5
  }'
```

**Response:**
```json
{
  "answer": "Based on your notes, you learned that the human brain has approximately 86 billion neurons [1].",
  "sources": [
    {
      "content": "The human brain has about 86 billion neurons.",
      "input_type": "text",
      "source": "biology-study",
      "timestamp": "2025-05-02T14:30:00",
      "score": 0.9234
    }
  ],
  "query": "What did I learn about neurons?"
}
```

---

### `GET /summary` — Daily Summary

```bash
# Today's summary
curl http://localhost:8000/summary

# Specific date
curl "http://localhost:8000/summary?target_date=2025-05-02"
```

**Response:**
```json
{
  "date": "2025-05-02",
  "insights": [
    "You explored neuroscience fundamentals and AI architecture today",
    "Key productivity system (Pomodoro) was studied and noted",
    "Multiple URL resources were captured for later review"
  ],
  "key_learnings": [
    "Human brain contains ~86 billion neurons",
    "Transformers use self-attention for parallel sequence processing"
  ],
  "important_notes": [
    "API v2 deadline is Friday — John handles auth, Sarah does migrations",
    "Follow-up meeting scheduled for Monday"
  ],
  "raw_summary": "..."
}
```

---

### `GET /memory/recent` — Debug

```bash
curl "http://localhost:8000/memory/recent?n=10"
```

---

### `GET /stats` — System Stats

```bash
curl http://localhost:8000/stats
```

---

## Running Examples

```bash
# Python examples (server must be running)
python examples/api_examples.py

# cURL examples
chmod +x examples/curl_examples.sh
./examples/curl_examples.sh
```

---

## Configuration Reference

| Variable | Default | Description |
|---|---|---|
| `LLM_PROVIDER` | `openai` | `openai` / `together` / `anthropic` |
| `OPENAI_API_KEY` | — | Your OpenAI key |
| `EMBEDDING_MODEL` | `all-MiniLM-L6-v2` | Local sentence-transformers model |
| `FAISS_INDEX_PATH` | `data/faiss_index` | Where FAISS saves to disk |
| `WHISPER_MODEL` | `base` | `tiny` / `base` / `small` / `medium` |
| `CHUNK_SIZE` | `512` | Words per chunk |
| `CHUNK_OVERLAP` | `64` | Overlap between chunks |
| `TOP_K_RETRIEVAL` | `5` | Number of chunks to retrieve per query |
| `LOG_LEVEL` | `INFO` | `DEBUG` / `INFO` / `WARNING` |

---

## Phase 2 Roadmap (What's Next)

- [ ] WhatsApp integration (Twilio / Meta Cloud API)
- [ ] Scheduled daily summary (cron / APScheduler)
- [ ] Multi-user support with auth
- [ ] Web dashboard (Streamlit / React)
- [ ] Graph memory (entity extraction)
- [ ] Smart reminders agent
- [ ] Export memories to Notion / Obsidian

---

## License

MIT — Build freely.
