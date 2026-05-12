"""
Contract Risk Analyser
Analyses uploaded PDF documents for risky clauses and flags them.
"""
from langchain_core.messages import HumanMessage, SystemMessage
from typing import List, Dict


RISK_CATEGORIES = [
    "Unlimited liability",
    "Automatic renewal without notice",
    "Unilateral amendment rights",
    "Broad indemnification",
    "Restrictive non-compete",
    "Unfair termination clause",
    "Intellectual property assignment",
    "Jurisdiction / governing law disadvantage",
    "Penalty or liquidated damages",
    "Exclusivity clause",
]

RISK_SYSTEM_PROMPT = """You are a senior contract lawyer specialising in risk analysis.
Your job is to identify risky, unfair, or unusual clauses in legal documents.
Be concise. Use plain English. Do not use markdown symbols like ** or #."""


def analyse_contract_risks(llm, context: str, jurisdiction: str = "General") -> Dict:
    """
    Analyse document context for risky clauses.
    Returns a dict with 'risks' list and 'summary'.
    """
    jurisdiction_note = f" Apply {jurisdiction} law where relevant." if jurisdiction != "General" else ""

    prompt = f"""Analyse the following contract text for risky or unfair clauses.{jurisdiction_note}

For each risk found, provide:
- Clause type (e.g. Liability, Termination, IP Assignment)
- Risk level: HIGH, MEDIUM, or LOW
- Brief explanation (1-2 sentences)
- The exact problematic phrase if possible

Format each risk as:
RISK: [clause type] | LEVEL: [HIGH/MEDIUM/LOW] | EXPLANATION: [explanation] | PHRASE: [phrase or N/A]

If no significant risks are found, say: NO_RISKS_FOUND

Contract text:
{context}
"""

    response = llm.invoke([
        SystemMessage(content=RISK_SYSTEM_PROMPT),
        HumanMessage(content=prompt)
    ])

    raw = response.content if hasattr(response, "content") else str(response)
    risks = _parse_risks(raw)

    summary_prompt = f"""Based on this contract analysis, write a 2-3 sentence plain English summary 
of the overall risk level and the most important issues to address.
Do not use markdown symbols.

Analysis:
{raw}
"""
    summary_response = llm.invoke([HumanMessage(content=summary_prompt)])
    summary = summary_response.content if hasattr(summary_response, "content") else str(summary_response)

    return {
        "risks": risks,
        "summary": summary.strip(),
        "jurisdiction": jurisdiction
    }


def _parse_risks(raw: str) -> List[Dict]:
    """Parse the structured risk output from the LLM."""
    if "NO_RISKS_FOUND" in raw:
        return []

    risks = []
    for line in raw.split("\n"):
        line = line.strip()
        if not line.startswith("RISK:"):
            continue
        try:
            parts = {}
            for segment in line.split("|"):
                segment = segment.strip()
                if ":" in segment:
                    key, _, value = segment.partition(":")
                    parts[key.strip()] = value.strip()

            risks.append({
                "clause_type": parts.get("RISK", "Unknown"),
                "level": parts.get("LEVEL", "MEDIUM").upper(),
                "explanation": parts.get("EXPLANATION", ""),
                "phrase": parts.get("PHRASE", "N/A")
            })
        except Exception:
            continue

    return risks
