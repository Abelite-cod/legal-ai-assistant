from pydantic import BaseModel
from typing import List

class AskRequest(BaseModel):
    session_id: str
    question: str

class AskResponse(BaseModel):
    answer: str
    sources: List[str]
    intent_score: float = 0.0
    