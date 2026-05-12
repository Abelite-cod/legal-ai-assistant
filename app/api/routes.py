import os
from fastapi import APIRouter, UploadFile, File, Depends, HTTPException
from fastapi.responses import JSONResponse, StreamingResponse
from app.models.schemas import (
    AskRequest, AskResponse,
    AnalyseRequest, AnalyseResponse,
    GenerateRequest, GenerateResponse,
    JURISDICTIONS
)
from app.services.rag_service import GeminiRAGService
from app.services.risk_analyser import analyse_contract_risks
from app.services.doc_generator import generate_legal_document, DOCUMENT_TYPES
from app.storage.vector_db import load_vectorstore
from app.services.context_builder import build_context
from app.utils.session_manager import get_history, add_message, get_all_sessions, delete_session
import json

router = APIRouter()
rag_service = GeminiRAGService()


def get_rag_service():
    return rag_service


# ─────────────────────────────────────────
# HEALTH CHECK
# ─────────────────────────────────────────
@router.get("/health")
async def health_check():
    api_key_set = bool(os.getenv("GOOGLE_API_KEY"))
    vectorstore_ready = rag_service.vectorstore is not None
    return {
        "status": "ok",
        "google_api_key_set": api_key_set,
        "vectorstore_ready": vectorstore_ready,
        "model": "gemini-flash-lite-latest"
    }


# ─────────────────────────────────────────
# JURISDICTIONS LIST
# ─────────────────────────────────────────
@router.get("/jurisdictions")
async def list_jurisdictions():
    return {"jurisdictions": JURISDICTIONS}


# ─────────────────────────────────────────
# DOCUMENT TYPES LIST
# ─────────────────────────────────────────
@router.get("/document_types")
async def list_document_types():
    return {"document_types": DOCUMENT_TYPES}


# ─────────────────────────────────────────
# PDF UPLOAD
# ─────────────────────────────────────────
@router.post("/upload_pdf")
async def upload_pdf(
    file: UploadFile = File(...),
    rag_service: GeminiRAGService = Depends(get_rag_service)
):
    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are supported.")

    file_location = await rag_service.upload_pdf(file)
    return {
        "status": "success",
        "filename": file.filename,
        "file_location": file_location,
        "message": "Document indexed. You can now ask questions, analyse risks, or generate documents."
    }


# ─────────────────────────────────────────
# ASK (non-streaming fallback)
# ─────────────────────────────────────────
@router.post("/ask", response_model=AskResponse)
async def ask_endpoint(
    request: AskRequest,
    rag_service: GeminiRAGService = Depends(get_rag_service)
):
    session_id = request.session_id
    question = request.question.strip()
    jurisdiction = request.jurisdiction or "General"

    if not session_id or not question:
        raise HTTPException(status_code=400, detail="Missing session_id or question.")

    history = get_history(session_id)[-6:]
    add_message(session_id, "user", question)

    try:
        result = await rag_service.ask_question(
            question=question,
            history=history,
            jurisdiction=jurisdiction
        )
    except RuntimeError as e:
        raise HTTPException(status_code=503, detail=str(e))
    except Exception as e:
        err_str = str(e)
        if "429" in err_str or "RESOURCE_EXHAUSTED" in err_str:
            raise HTTPException(status_code=429, detail="Rate limited. Please wait and try again.")
        raise HTTPException(status_code=500, detail="An unexpected error occurred.")

    add_message(session_id, "assistant", result["answer"])
    return result


# ─────────────────────────────────────────
# STREAMING ASK
# ─────────────────────────────────────────
@router.post("/ask/stream")
async def ask_stream(
    request: AskRequest,
    rag_service: GeminiRAGService = Depends(get_rag_service)
):
    session_id = request.session_id
    question = request.question.strip()
    jurisdiction = request.jurisdiction or "General"

    if not session_id or not question:
        raise HTTPException(status_code=400, detail="Missing session_id or question.")

    history = get_history(session_id)[-6:]
    add_message(session_id, "user", question)

    async def event_stream():
        full_answer = ""
        try:
            async for chunk in rag_service.ask_question_stream(
                question=question,
                history=history,
                jurisdiction=jurisdiction
            ):
                full_answer += chunk
                yield f"data: {json.dumps({'chunk': chunk})}\n\n"

            yield f"data: {json.dumps({'done': True, 'sources': []})}\n\n"
            add_message(session_id, "assistant", full_answer)

        except Exception as e:
            err = str(e)
            if "429" in err or "RESOURCE_EXHAUSTED" in err:
                yield f"data: {json.dumps({'error': 'Rate limited. Please wait and try again.'})}\n\n"
            else:
                yield f"data: {json.dumps({'error': 'An error occurred. Please try again.'})}\n\n"

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"}
    )


# ─────────────────────────────────────────
# CONTRACT RISK ANALYSER
# ─────────────────────────────────────────
@router.post("/analyse")
async def analyse_contract(
    request: AnalyseRequest,
    rag_service: GeminiRAGService = Depends(get_rag_service)
):
    """Analyse the uploaded document for risky clauses."""
    vs = rag_service._get_vectorstore()
    if not vs:
        raise HTTPException(status_code=400, detail="No document uploaded. Please upload a PDF first.")

    try:
        # Get all document content for analysis
        data = vs.get()
        if not data.get("documents"):
            raise HTTPException(status_code=400, detail="Document is empty or could not be read.")

        # Use first 6000 chars for analysis (enough for risk detection)
        full_text = "\n".join(data["documents"])[:6000]

        result = analyse_contract_risks(
            llm=rag_service.llm,
            context=full_text,
            jurisdiction=request.jurisdiction or "General"
        )

        # Post result as a chat message
        risk_count = len(result["risks"])
        high_risks = [r for r in result["risks"] if r["level"] == "HIGH"]
        summary_msg = f"Contract Risk Analysis complete. Found {risk_count} issue(s)"
        if high_risks:
            summary_msg += f" including {len(high_risks)} HIGH risk clause(s)"
        summary_msg += f". Jurisdiction: {result['jurisdiction']}."
        add_message(request.session_id, "assistant", summary_msg)

        return result

    except HTTPException:
        raise
    except Exception as e:
        err = str(e)
        if "429" in err or "RESOURCE_EXHAUSTED" in err:
            raise HTTPException(status_code=429, detail="Rate limited. Please wait and try again.")
        raise HTTPException(status_code=500, detail=f"Analysis failed: {err[:200]}")


# ─────────────────────────────────────────
# LEGAL DOCUMENT GENERATOR
# ─────────────────────────────────────────
@router.post("/generate")
async def generate_document(
    request: GenerateRequest,
    rag_service: GeminiRAGService = Depends(get_rag_service)
):
    """Generate a legal document from a template."""
    if not request.document_type or not request.details:
        raise HTTPException(status_code=400, detail="document_type and details are required.")

    try:
        document = generate_legal_document(
            llm=rag_service.llm,
            document_type=request.document_type,
            details=request.details,
            jurisdiction=request.jurisdiction or "General"
        )

        add_message(
            request.session_id,
            "assistant",
            f"Generated {request.document_type} ({request.jurisdiction}). Document is ready to copy or download."
        )

        return {
            "document": document,
            "document_type": request.document_type,
            "jurisdiction": request.jurisdiction or "General"
        }

    except Exception as e:
        err = str(e)
        if "429" in err or "RESOURCE_EXHAUSTED" in err:
            raise HTTPException(status_code=429, detail="Rate limited. Please wait and try again.")
        raise HTTPException(status_code=500, detail=f"Generation failed: {err[:200]}")


# ─────────────────────────────────────────
# SESSIONS
# ─────────────────────────────────────────
@router.get("/sessions")
async def list_sessions():
    sessions = get_all_sessions()
    return {"sessions": sessions, "total": len(sessions)}


@router.delete("/sessions/{session_id}")
async def remove_session(session_id: str):
    deleted = delete_session(session_id)
    if not deleted:
        raise HTTPException(status_code=404, detail=f"Session '{session_id}' not found.")
    return {"status": "deleted", "session_id": session_id}
