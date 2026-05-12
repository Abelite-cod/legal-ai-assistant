from pydantic import BaseModel
from typing import List, Optional

JURISDICTIONS = ["Nigeria", "United Kingdom", "United States", "Canada", "Australia", "General"]

class AskRequest(BaseModel):
    session_id: str
    question: str
    jurisdiction: Optional[str] = "General"

class AskResponse(BaseModel):
    answer: str
    sources: List[str]
    intent_score: float = 0.0
    confidence: str = "Medium"

class AnalyseRequest(BaseModel):
    session_id: str
    jurisdiction: Optional[str] = "General"

class AnalyseResponse(BaseModel):
    risks: List[dict]
    summary: str
    jurisdiction: str

class GenerateRequest(BaseModel):
    session_id: str
    document_type: str   # e.g. "NDA", "Demand Letter", "Tenancy Agreement"
    details: str         # user-provided context/details
    jurisdiction: Optional[str] = "General"

class GenerateResponse(BaseModel):
    document: str
    document_type: str
    jurisdiction: str
