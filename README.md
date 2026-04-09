# ⚖️ Legal AI Assistant (RAG + Gemini)

A conversational AI-powered legal assistant that helps users understand legal situations, analyze documents, and get practical guidance using Retrieval-Augmented Generation (RAG) and Google Gemini.

---

## 🚀 Overview

This project is a full-stack AI system that combines:

- Conversational AI (chat-based interface)
- Legal reasoning engine
- Document analysis (PDF support)
- Retrieval-Augmented Generation (RAG)
- Multi-session chat system

Users can:

- Ask legal questions in plain English
- Upload documents and query them
- Maintain multiple chat sessions
- Receive contextual, structured legal guidance

---

## 🧠 Core Features

### 💬 Conversational AI

- Natural chat interface
- Intent detection (greeting, legal question, document query, etc.)
- Context-aware responses using conversation history

### ⚖️ Legal Reasoning Engine

- Empathetic and practical responses
- Step-by-step guidance
- Real-world scenario handling (e.g. wrongful termination)

### 📄 RAG (Document Intelligence)

- Upload PDF documents
- Automatic chunking and embedding
- Hybrid retrieval (semantic + keyword)
- Context-grounded answers (no hallucination)

### 🧩 Multi-Session Chat

- Multiple chat threads
- Persistent sessions (via frontend local storage)
- Session-based API interaction

### ⚡ Performance Optimizations

- Context trimming
- Response caching (LRU-style)
- Efficient embedding model

---

## 🏗️ Tech Stack

### Backend

- FastAPI
- LangChain
- Google Gemini (via `langchain_google_genai`)
- Chroma Vector DB
- HuggingFace Embeddings (`all-MiniLM-L6-v2`)

### Frontend

- HTML / CSS / JavaScript (custom UI)
- LocalStorage (chat persistence)

### Dev Tools

- Uvicorn
- Ngrok (for mobile testing)
- Python Dotenv

---

## 📂 Project Structure

```
app/
├── api/                 # FastAPI routes
├── services/            # Core logic (RAG, reasoning, intent)
├── storage/             # Vector DB + PDF storage
├── models/              # DB models
├── db/                  # Database setup
└── main.py              # App entry point
```

---

## ⚙️ How It Works

### 1. User Input

User sends a message via frontend.

### 2. Intent Classification

System determines if it's:

- Conversation
- Legal reasoning
- Document query

### 3. Routing

- Conversation → quick response
- Legal → LLM reasoning
- Document → RAG pipeline

### 4. RAG Flow (if document query)

- Retrieve relevant chunks
- Build context
- Send to LLM with strict prompt

### 5. Response Processing

- Normalize output
- Remove formatting noise
- Return clean response

---

## 🔧 Setup Instructions

### 1. Clone Repo

```bash
git clone <your-repo-url>
cd project
```

### 2. Create Virtual Environment

```bash
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Environment Variables

Create `.env` file:

```
GOOGLE_API_KEY=your_api_key_here
```

### 5. Run Backend

```bash
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

### 6. Test API

Open:

```
http://localhost:8000/docs
```

---

## 🌐 Frontend Usage

- Open the HTML file in browser
- Or connect via ngrok for mobile testing

Update API endpoint in JS:

```js
const API_BASE = "https://your-ngrok-url";
```

---

## 📱 Mobile Testing (Ngrok)

```bash
ngrok http 8000
```

Use generated URL in frontend.

---

## 🚧 Known Improvements

- Better UI (React / Tailwind)
- Authentication system
- Persistent DB chat storage
- File upload UI
- Streaming responses
- Deployment (Vercel + Railway)

---

## 🎯 Future Roadmap

- Convert to full web app
- Add user accounts
- Improve legal accuracy with fine-tuning
- Mobile app (React Native)

---

## ⚠️ Disclaimer

This application is for informational purposes only and does not constitute legal advice.

---

## 👨‍💻 Author

Built by a full-stack developer exploring AI-powered legal systems.

---

## ⭐ If you like this project

Give it a star and share feedback!
