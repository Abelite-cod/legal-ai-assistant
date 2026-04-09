import re
from typing import Dict, List, Tuple


INTENT_RULES: Dict[str, List[Tuple[str, int]]] = {
    "greeting": [
        (r"\bhi\b|\bhello\b|\bhey\b", 2),
    ],
    "small_talk": [
        (r"how are you|what'?s up|how'?s it going", 2),
    ],
    "emotional_distress": [
        (r"harass|abuse|threat|bullied", 3),
        (r"fired unfairly|terminated unfairly", 4),
    ],
    "document_query": [
        (r"document|pdf|section|page|clause", 3),
    ],
    "legal_question": [
        (r"law|legal|sue|court|rights|contract", 2),
    ],
}


PRIORITY_ORDER = [
    "emotional_distress",  # highest priority
    "document_query",
    "legal_question",
    "greeting",
    "small_talk",
]


def classify_intent(question: str, debug: bool = False) -> str:
    q = question.lower().strip()

    scores: Dict[str, int] = {intent: 0 for intent in INTENT_RULES}

    # -------------------------
    # SCORING
    # -------------------------
    for intent, patterns in INTENT_RULES.items():
        for pattern, weight in patterns:
            if re.search(pattern, q):
                scores[intent] += weight

    # -------------------------
    # SELECT BEST INTENT
    # -------------------------
    max_score = max(scores.values())

    if max_score == 0:
        return "legal_question"  # safe fallback

    # Handle ties using priority
    candidates = [intent for intent, score in scores.items() if score == max_score]

    for intent in PRIORITY_ORDER:
        if intent in candidates:
            if debug:
                print("Scores:", scores)
                print("Chosen:", intent)
            return intent

    return "legal_question"