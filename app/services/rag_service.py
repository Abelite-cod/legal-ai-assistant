from app.services.retriever import hybrid_retrieve, reset_bm25
from app.services.context_builder import build_context
from app.storage.vector_db import load_vectorstore
from app.storage.pdf_storage import store_pdf
from app.services.intent_classifier import classify_intent
from app.services.query_rewriter import rewrite_query

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

from langchain_core.messages import HumanMessage, SystemMessage
from typing import List, Dict, Any, Optional
from dotenv import load_dotenv

import hashlib
import re

load_dotenv()

CHROMA_DIR = "chroma_db"
LLM_MODEL = "gemini-flash-lite-latest"

MAX_CONTEXT_CHARS = 12000
MAX_CACHE_SIZE = 200


# -------------------------
# CLEAN OUTPUT (IMPORTANT FIX)
# -------------------------
def clean_text(text: str) -> str:
    """
    Removes markdown symbols so frontend doesn't break UI
    """
    text = re.sub(r"(\*\*|\*)", "", text)  # remove bold/italic
    text = re.sub(r"(#{1,6}\s*)", "", text)  # remove headings
    return text.strip()


# -------------------------
# HELPERS
# -------------------------
def normalize(text: str) -> str:
    return text.lower().strip()


def make_cache_key(question: str, history: list) -> str:
    raw = question + str(history)
    return hashlib.md5(raw.encode()).hexdigest()


def normalize_llm_output(response) -> str:
    content = getattr(response, "content", response)

    if isinstance(content, str):
        return clean_text(content)

    if isinstance(content, list):
        return clean_text(" ".join(
            item.get("text", "") if isinstance(item, dict) else str(item)
            for item in content
        ))

    if isinstance(content, dict):
        return clean_text(content.get("text", str(content)))

    return clean_text(str(content))


def trim_documents(docs, max_chars=MAX_CONTEXT_CHARS):
    trimmed = []
    total = 0

    for doc in docs:
        text = doc.page_content
        if total + len(text) > max_chars:
            break
        trimmed.append(doc)
        total += len(text)

    return trimmed


# -------------------------
# CONVERSATION HANDLER
# -------------------------
def handle_conversation(intent: str, question: str) -> Optional[str]:
    q = normalize(question)

    if intent == "greeting" and len(q.split()) <= 3:
        return "Hey 👋 I’m your legal assistant. How can I help you today?"

    if intent == "small_talk":
        return "I'm here to help with legal questions or documents."

    return None


# -------------------------
# LEGAL REASONING (CLEAN + NO REPETITION)
# -------------------------
def handle_legal_reasoning(llm, question: str, history: str = "") -> str:
    prompt = f"""
You are a senior legal assistant.

Rules:
- Be clear and structured
- Do not repeat phrases
- Do not use markdown symbols (*, #)
- Be concise

Conversation:
{history}

User:
{question}
"""

    response = llm.invoke([HumanMessage(content=prompt)])
    return normalize_llm_output(response)


# -------------------------
# SYSTEM PROMPT
# -------------------------
SYSTEM_PROMPT = """
You are a legal document assistant.

Rules:
- Be empathetic, practical and clean plain text
- Use ONLY provided context
- Never hallucinate
- If missing info say:
  The document does not contain this information.
- Do not use markdown symbols like ** or #
"""


# -------------------------
# MAIN SERVICE
# -------------------------
class GeminiRAGService:
    def __init__(self):
        self.cache: Dict[str, Dict[str, Any]] = {}
        self._llm = None
        self._embeddings = None
        self.vectorstore = None  # loaded lazily on first use

    def _get_llm(self):
        """Lazy-load the LLM so startup never crashes on missing API key."""
        if self._llm is None:
            import os
            api_key = os.getenv("GOOGLE_API_KEY")
            if not api_key:
                raise RuntimeError(
                    "GOOGLE_API_KEY is not set. Add it to your .env file."
                )
            self._llm = ChatGoogleGenerativeAI(
                model=LLM_MODEL,
                temperature=0.2
            )
        return self._llm

    def _get_embeddings(self):
        """Lazy-load embeddings model."""
        if self._embeddings is None:
            self._embeddings = HuggingFaceEmbeddings(
                model_name="sentence-transformers/all-MiniLM-L6-v2"
            )
        return self._embeddings

    def _get_vectorstore(self):
        """Lazy-load vectorstore."""
        if self.vectorstore is None:
            self.vectorstore = load_vectorstore()
        return self.vectorstore

    @property
    def llm(self):
        return self._get_llm()

    # -------------------------
    # CACHE
    # -------------------------
    def _cache_set(self, key, value):
        self.cache[key] = value

        if len(self.cache) > MAX_CACHE_SIZE:
            self.cache.pop(next(iter(self.cache)))

    # -------------------------
    # PDF INGESTION
    # -------------------------
    async def upload_pdf(self, file):
        file_path = await store_pdf(file)

        loader = PyPDFLoader(file_path)
        docs = loader.load()

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=400,
            chunk_overlap=100
        )

        chunks = splitter.split_documents(docs)

        vs = self._get_vectorstore()
        if vs:
            vs.add_documents(chunks)
            self.vectorstore = vs
        else:
            self.vectorstore = Chroma.from_documents(
                documents=chunks,
                embedding=self._get_embeddings(),
                persist_directory=CHROMA_DIR
            )

        try:
            self.vectorstore.persist()
        except Exception:
            pass

        reset_bm25()
        return file_path

    # -------------------------
    # POST PROCESS (FIXED)
    # -------------------------
    def post_process_answer(self, answer: str, intent: str) -> str:
        if intent == "document_query":
            return answer + "\n\nDo you want me to extract key clauses?"

        if intent == "legal_question":
            return answer

        return answer

    # -------------------------
    # MAIN PIPELINE
    # -------------------------
    async def ask_question(
        self,
        question: str,
        history: List[Dict[str, str]] = None
    ) -> Dict[str, Any]:

        history = history or []
        key = make_cache_key(question, history)

        if key in self.cache:
            return self.cache[key]

        intent = classify_intent(question)

        conversation_text = "\n".join(
            f"{m['role'].upper()}: {m['content']}" for m in history
        )

        # -------------------------
        # CONVERSATION
        # -------------------------
        convo_reply = handle_conversation(intent, question)
        if convo_reply:
            result = {
                "answer": convo_reply,
                "sources": [],
                "intent_score": 0.9,
                "confidence": "High"
            }
            self._cache_set(key, result)
            return result

        # -------------------------
        # LEGAL REASONING
        # -------------------------
        if intent in ["legal_question", "emotional_distress"]:
            answer = handle_legal_reasoning(
                self.llm,
                question,
                conversation_text
            )

            if not answer:
                answer = "I couldn't generate a proper response."

            answer = self.post_process_answer(answer, intent)

            result = {
                "answer": answer,
                "sources": [],
                "intent_score": 0.85,
                "confidence": "High"
            }

            self._cache_set(key, result)
            return result

        # -------------------------
        # RAG PIPELINE
        # -------------------------
        if intent == "document_query":

            if not self.vectorstore:
                return {
                    "answer": "Please upload a document first.",
                    "sources": [],
                    "intent_score": 0.0,
                    "confidence": "Low"
                }

            # Rewrite the query for better retrieval
            try:
                search_query = rewrite_query(self.llm, question)
            except Exception:
                search_query = question  # fallback to original if rewriter fails

            retrieved_docs = hybrid_retrieve(search_query, self.vectorstore)

            if not retrieved_docs:
                return {
                    "answer": "No relevant information found.",
                    "sources": [],
                    "intent_score": 0.0,
                    "confidence": "Low"
                }

            trimmed_docs = trim_documents(retrieved_docs)
            context, sources = build_context(trimmed_docs)

            messages = [
                SystemMessage(content=SYSTEM_PROMPT),
                HumanMessage(content=f"""
Context:
{context}

Question:
{question}
""")
            ]

            response = self.llm.invoke(messages)
            answer = normalize_llm_output(response)

            if not answer:
                answer = "The document does not contain this information."

            answer = self.post_process_answer(answer, intent)

            result = {
                "answer": answer,
                "sources": sources,
                "intent_score": 0.95,
                "confidence": "High"
            }

            self._cache_set(key, result)
            return result

        # -------------------------
        # FALLBACK
        # -------------------------
        return {
            "answer": "Could you clarify your question?",
            "sources": [],
            "intent_score": 0.3,
            "confidence": "Medium"
        }

    # -------------------------
    # STREAMING PIPELINE
    # -------------------------
    async def ask_question_stream(
        self,
        question: str,
        history: List[Dict[str, str]] = None
    ):
        """
        Async generator that yields text chunks for streaming responses.
        Falls back to non-streaming for conversation/greeting intents.
        """
        history = history or []
        intent = classify_intent(question)

        conversation_text = "\n".join(
            f"{m['role'].upper()}: {m['content']}" for m in history
        )

        # Quick conversation replies — yield whole string at once
        convo_reply = handle_conversation(intent, question)
        if convo_reply:
            yield convo_reply
            return

        # Legal reasoning — stream via Gemini
        if intent in ["legal_question", "emotional_distress"]:
            prompt = f"""You are a senior legal assistant.

Rules:
- Be clear and structured
- Use numbered lists where appropriate
- Do not use markdown symbols (*, #)
- Be concise

Conversation:
{conversation_text}

User:
{question}
"""
            for chunk in self.llm.stream([HumanMessage(content=prompt)]):
                text = getattr(chunk, "content", "")
                if text:
                    # Strip markdown symbols inline
                    text = re.sub(r"(\*\*|\*|#{1,6}\s*)", "", text)
                    yield text
            return

        # Document query — stream after retrieval
        if intent == "document_query":
            vs = self._get_vectorstore()
            if not vs:
                yield "Please upload a document first."
                return

            try:
                search_query = rewrite_query(self.llm, question)
            except Exception:
                search_query = question

            retrieved_docs = hybrid_retrieve(search_query, vs)
            if not retrieved_docs:
                yield "No relevant information found in the document."
                return

            trimmed_docs = trim_documents(retrieved_docs)
            context, _ = build_context(trimmed_docs)

            messages = [
                SystemMessage(content=SYSTEM_PROMPT),
                HumanMessage(content=f"Context:\n{context}\n\nQuestion:\n{question}")
            ]

            for chunk in self.llm.stream(messages):
                text = getattr(chunk, "content", "")
                if text:
                    text = re.sub(r"(\*\*|\*|#{1,6}\s*)", "", text)
                    yield text
            return

        # Fallback
        yield "Could you clarify your question?"