"""
Legal Document Generator
Generates legal documents from templates based on user-provided details.
"""
from langchain_core.messages import HumanMessage, SystemMessage
import datetime as _dt


DOCUMENT_TYPES = [
    "Non-Disclosure Agreement (NDA)",
    "Demand Letter",
    "Tenancy Agreement",
    "Employment Offer Letter",
    "Freelance Service Agreement",
    "Simple Loan Agreement",
    "Letter Before Action",
    "Privacy Policy",
]

GENERATOR_SYSTEM_PROMPT = """You are a senior legal document drafter.
Generate professional, legally sound documents based on the details provided.
Use clear, plain English. Structure the document properly with numbered clauses.
Do not use markdown symbols like ** or #.
Always include a disclaimer at the end that this is a template and professional legal advice should be sought."""


def generate_legal_document(llm, document_type: str, details: str, jurisdiction: str = "General") -> str:
    """
    Generate a legal document based on type, details, and jurisdiction.
    Returns the generated document as a string.
    """
    today = _dt.date.today().strftime("%B %d, %Y")
    jurisdiction_note = f" This document should comply with {jurisdiction} law." if jurisdiction != "General" else ""

    prompt = f"""Generate a {document_type} based on the following details.{jurisdiction_note}
Today's date: {today}

Details provided by the user:
{details}

Requirements:
- Use proper legal document structure with numbered clauses
- Include all standard clauses for this document type
- Fill in placeholders like [PARTY NAME], [DATE], [ADDRESS] where specific info is not provided
- Make it professional and legally sound
- End with a disclaimer that this is a template

Generate the complete document now:
"""

    response = llm.invoke([
        SystemMessage(content=GENERATOR_SYSTEM_PROMPT),
        HumanMessage(content=prompt)
    ])

    content = response.content if hasattr(response, "content") else str(response)
    return content.strip()
