# ⚖️ Legal AI Assistant

A production-ready, full-stack AI legal assistant with RAG, streaming responses, contract risk analysis, legal document generation, and multi-jurisdiction support.

**Live Demo:** [legal-assistant.up.railway.app](https://legal-assistant.up.railway.app)

---

## 🚀 What It Does

| Feature                            | Description                                                                 |
| ---------------------------------- | --------------------------------------------------------------------------- |
| 💬 **Conversational AI**           | Ask legal questions in plain English with context-aware responses           |
| 📄 **Document Intelligence (RAG)** | Upload any PDF contract or legal document and ask questions about it        |
| 🔍 **Contract Risk Analyser**      | Automatically detects risky clauses (HIGH/MEDIUM/LOW) in uploaded documents |
| 📝 **Legal Document Generator**    | Generate NDAs, demand letters, tenancy agreements, and more                 |
| 🌍 **Multi-Jurisdiction**          | Answers cite local law for Nigeria, UK, USA, Canada, Australia              |
| ⚡ **Streaming Responses**         | Text appears word-by-word like ChatGPT                                      |
| 📱 **PWA / Mobile App**            | Install on iOS/Android home screen — works like a native app                |
| 🧠 **Hybrid Retrieval**            | Vector search + BM25 keyword search + CrossEncoder reranking                |

---

## 🏗️ Architecture

```
User Input
    ↓
Intent Classifier (greeting / legal_question / document_query / emotional_distress)
    ↓
┌─────────────────────────────────────────────────────┐
│  Legal Reasoning     │  RAG Pipeline                │
│  ─────────────────   │  ──────────────              │
│  Gemini LLM          │  Query Rewriter              │
│  + jurisdiction      │  → Vector Search (Chroma)    │
│  + today's date      │  → BM25 Keyword Search       │
│  + conversation      │  → CrossEncoder Reranker     │
│  history             │  → Context Builder           │
│                      │  → Gemini LLM                │
└─────────────────────────────────────────────────────┘
    ↓
Streaming Response (SSE) → Frontend
    ↓
SQLite Session Storage
```

---

## 🧩 Tech Stack

### Backend

- **FastAPI** — async REST API + Server-Sent Events streaming
- **LangChain** — LLM orchestration, document loading, text splitting
- **Google Gemini** (`gemini-flash-lite-latest`) — LLM for reasoning and generation
- **Chroma** — vector database for document embeddings
- **HuggingFace** (`all-MiniLM-L6-v2`) — sentence embeddings
- **CrossEncoder** (`ms-marco-MiniLM-L-6-v2`) — reranking retrieved chunks
- **BM25** (`rank-bm25`) — keyword-based retrieval
- **SQLAlchemy + SQLite** — persistent session/message storage

### Frontend

- **Vanilla HTML/CSS/JS** — no framework, fast and lightweight
- **PWA** — installable on mobile, service worker for offline shell
- **Server-Sent Events** — real-time streaming responses
- **LocalStorage** — client-side chat persistence

### Infrastructure

- **Docker** — containerised deployment
- **Railway** — cloud hosting with auto-deploy from GitHub
- **Uvicorn** — ASGI server

---

## 📂 Project Structure

```
legal-ai-assistant/
├── app/
│   ├── main.py                  # FastAPI app, static file serving
│   ├── db.py                    # SQLAlchemy setup (SQLite)
│   ├── index.html               # Frontend UI (PWA)
│   ├── api/
│   │   └── routes.py            # All API endpoints
│   ├── models/
│   │   ├── db_models.py         # Conversation & Message ORM models
│   │   └── schemas.py           # Pydantic request/response schemas
│   ├── services/
│   │   ├── rag_service.py       # Main RAG pipeline + streaming
│   │   ├── intent_classifier.py # Rule-based intent detection
│   │   ├── retriever.py         # Hybrid retrieval (vector + BM25)
│   │   ├── reranker.py          # CrossEncoder reranking
│   │   ├── bm25_index.py        # BM25 keyword index
│   │   ├── context_builder.py   # Context assembly + source tracking
│   │   ├── query_rewriter.py    # Query optimisation for retrieval
│   │   ├── risk_analyser.py     # Contract risk detection
│   │   └── doc_generator.py     # Legal document generation
│   ├── storage/
│   │   ├── vector_db.py         # Chroma vector store loader
│   │   └── pdf_storage.py       # PDF file storage
│   ├── static/
│   │   ├── manifest.json        # PWA manifest
│   │   └── sw.js                # Service worker
│   └── utils/
│       └── session_manager.py   # DB session CRUD
├── Dockerfile
├── railway.toml
├── .env.example
└── requirements.txt
```

---

## ⚙️ Setup

### 1. Clone

```bash
git clone https://github.com/Abelite-cod/legal-ai-assistant.git
cd legal-ai-assistant
```

### 2. Environment Variables

```bash
cp .env.example .env
```

Edit `.env`:

```
GOOGLE_API_KEY=your_google_api_key_here
DATABASE_URL=sqlite:///./legal_ai.db
```

Get a free API key at [aistudio.google.com](https://aistudio.google.com/app/apikey)

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Run

```bash
python -m uvicorn app.main:app --host 0.0.0.0 --port 8000
```

Open **http://localhost:8000**

---

## 🌐 API Endpoints

| Method   | Endpoint          | Description                                  |
| -------- | ----------------- | -------------------------------------------- |
| `GET`    | `/health`         | Health check + model status                  |
| `POST`   | `/ask`            | Ask a legal question (non-streaming)         |
| `POST`   | `/ask/stream`     | Ask a question with streaming response (SSE) |
| `POST`   | `/upload_pdf`     | Upload and index a PDF document              |
| `POST`   | `/analyse`        | Analyse uploaded document for contract risks |
| `POST`   | `/generate`       | Generate a legal document from a template    |
| `GET`    | `/sessions`       | List all chat sessions                       |
| `DELETE` | `/sessions/{id}`  | Delete a chat session                        |
| `GET`    | `/jurisdictions`  | List supported jurisdictions                 |
| `GET`    | `/document_types` | List supported document types                |
| `GET`    | `/docs`           | Interactive API documentation (Swagger UI)   |

---

## 📱 Mobile App (PWA)

**Android (Chrome):**

1. Open the live URL in Chrome
2. Tap menu → "Add to Home screen"

**iPhone (Safari):**

1. Open the live URL in Safari
2. Tap Share → "Add to Home Screen"

---

## 🚀 Deploy to Railway

1. Fork this repo
2. Go to [railway.app](https://railway.app) → New Project → Deploy from GitHub
3. Add environment variables: `GOOGLE_API_KEY` and `DATABASE_URL`
4. Railway auto-builds from the `Dockerfile` and deploys

---

## 🔑 Key Design Decisions

**Why Hybrid Retrieval?**
Pure vector search misses exact keyword matches (e.g. clause numbers, party names). BM25 catches these. The CrossEncoder reranker then scores the combined results for relevance — significantly improving precision over either method alone.

**Why Streaming?**
Legal answers can be long. Streaming via SSE means the user sees the first words in ~1 second instead of waiting 5-10 seconds for the full response.

**Why SQLite?**
For a demo/portfolio project, SQLite requires zero infrastructure. The schema is designed to migrate to PostgreSQL with a single `DATABASE_URL` change.

**Why not LangChain agents?**
The intent classifier + routing approach gives deterministic, fast responses for the most common cases (greetings, simple legal questions) without the overhead of agent planning loops.

---

## ⚠️ Disclaimer

This application is for informational purposes only and does not constitute legal advice. Always consult a qualified legal professional for your specific situation.

---

## 👨‍💻 Author

**Abel Okagbare** — Full-stack developer specialising in AI systems and backend engineering.

- GitHub: [@Abelite-cod](https://github.com/Abelite-cod)
