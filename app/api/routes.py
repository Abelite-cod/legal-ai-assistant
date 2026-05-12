import os
from fastapi import APIRouter, UploadFile, File, Depends, HTTPException
from fastapi.responses import JSONResponse
from app.models.schemas import AskRequest, AskResponse
from app.services.rag_service import GeminiRAGService
from app.utils.session_manager import get_history, add_message, get_all_sessions, delete_session

router = APIRouter()
rag_service = GeminiRAGService()


# Dependency
def get_rag_service():
    return rag_service


# ─────────────────────────────────────────
# HEALTH CHECK
# ─────────────────────────────────────────
@router.get("/health")
async def health_check():
    """Quick liveness check — confirms the API is running and Gemini key is set."""
    api_key_set = bool(os.getenv("GOOGLE_API_KEY"))
    vectorstore_ready = rag_service.vectorstore is not None

    return {
        "status": "ok",
        "google_api_key_set": api_key_set,
        "vectorstore_ready": vectorstore_ready,
        "model": "gemini-flash-lite-latest"
    }


# ─────────────────────────────────────────
# PDF UPLOAD
# ─────────────────────────────────────────
@router.post("/upload_pdf")
async def upload_pdf(
    file: UploadFile = File(...),
    rag_service: GeminiRAGService = Depends(get_rag_service)
):
    """Upload a PDF and index it for retrieval."""
    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are supported.")

    file_location = await rag_service.upload_pdf(file)
    return {
        "status": "success",
        "filename": file.filename,
        "file_location": file_location,
        "message": "Document indexed successfully. You can now ask questions about it."
    }


# ─────────────────────────────────────────
# ASK
# ─────────────────────────────────────────
@router.post("/ask", response_model=AskResponse)
async def ask_endpoint(
    request: AskRequest,
    rag_service: GeminiRAGService = Depends(get_rag_service)
):
    """Send a question and receive a legal AI response."""
    session_id = request.session_id
    question = request.question.strip()

    if not session_id or not question:
        raise HTTPException(status_code=400, detail="Missing session_id or question.")

    history = get_history(session_id)
    add_message(session_id, "user", question)

    try:
        result = await rag_service.ask_question(
            question=question,
            history=history
        )
    except RuntimeError as e:
        raise HTTPException(status_code=503, detail=str(e))
    except Exception as e:
        err_str = str(e)
        if "429" in err_str or "RESOURCE_EXHAUSTED" in err_str:
            raise HTTPException(
                status_code=429,
                detail="The AI model is temporarily rate-limited. Please wait a moment and try again."
            )
        raise HTTPException(status_code=500, detail="An unexpected error occurred. Please try again.")

    add_message(session_id, "assistant", result["answer"])

    return result


# ─────────────────────────────────────────
# SESSIONS
# ─────────────────────────────────────────
@router.get("/sessions")
async def list_sessions():
    """List all chat sessions with message counts."""
    sessions = get_all_sessions()
    return {"sessions": sessions, "total": len(sessions)}


@router.delete("/sessions/{session_id}")
async def remove_session(session_id: str):
    """Delete a chat session and all its messages."""
    deleted = delete_session(session_id)
    if not deleted:
        raise HTTPException(status_code=404, detail=f"Session '{session_id}' not found.")
    return {"status": "deleted", "session_id": session_id}
