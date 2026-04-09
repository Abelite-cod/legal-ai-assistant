from fastapi import APIRouter, UploadFile, File, Depends, HTTPException
from app.models.schemas import AskRequest, AskResponse
from app.services.rag_service import GeminiRAGService
from app.utils.session_manager import get_history, add_message

router = APIRouter()
rag_service = GeminiRAGService()

# Dependency
def get_rag_service():
    return rag_service


@router.post("/upload_pdf")
async def upload_pdf(
    file: UploadFile = File(...),
    rag_service: GeminiRAGService = Depends(get_rag_service)
):
    """
    Upload a PDF and store it for retrieval.
    """
    file_location = await rag_service.upload_pdf(file)
    return {"status": "success", "file_location": file_location}


@router.post("/ask", response_model=AskResponse)
async def ask_endpoint(
    request: AskRequest,
    rag_service: GeminiRAGService = Depends(get_rag_service)
):
    session_id = request.session_id
    question = request.question.strip()

    if not session_id or not question:
        raise HTTPException(status_code=400, detail="Missing session_id or question")

    history = get_history(session_id)

    add_message(session_id, "user", question)

    result = await rag_service.ask_question(
        question=question,
        history=history
    )

    add_message(session_id, "assistant", result["answer"])

    return result