import re
from typing import Dict, List, Tuple


INTENT_RULES: Dict[str, List[Tuple[str, int]]] = {
    "greeting": [
        (r"\bhi\b|\bhello\b|\bhey\b|\bgreetings\b|\bgood\s+(morning|afternoon|evening)\b", 2),
    ],
    "small_talk": [
        (r"how are you|what'?s up|how'?s it going|who are you|what can you do|what are you", 2),
    ],
    "emotional_distress": [
        (r"\bharass(ed|ment|ing)?\b|\babuse(d)?\b|\bthreaten(ed|ing)?\b|\bbullied?\b", 3),
        (r"fired unfairly|terminated unfairly|wrongful(ly)?\s+(fired|terminated|dismissed)", 4),
        (r"discriminat(ed|ion|ing)|hostile work(place)?|retaliat(ed|ion|ing)", 3),
    ],
    "document_query": [
        (r"\bdocument\b|\bpdf\b|\bsection\b|\bpage\b|\bclause\b|\bcontract\b|\bagreement\b", 3),
        (r"\baccording to\b|\bin the (document|file|pdf|contract)\b|\bwhat does (it|the document|this) say\b", 4),
        (r"\bparagraph\b|\bexhibit\b|\bschedule\b|\bannex\b|\bappendix\b|\bartice\b", 3),
        (r"\bupload(ed)?\b|\bfile\b|\battach(ment|ed)?\b", 2),
    ],
    "legal_question": [
        (r"\blaw\b|\blegal\b|\bsue\b|\blawsuit\b|\bcourt\b|\brights?\b|\bcontract\b", 2),
        (r"\battorney\b|\blawyer\b|\bcounsel\b|\bjudge\b|\bjury\b|\bverdict\b|\bappeal\b", 3),
        (r"\bliabilit(y|ies)\b|\bnegligence\b|\bdamages?\b|\bcompensation\b|\bsettlement\b", 3),
        (r"\btermination\b|\bdismissal\b|\beviction\b|\blandlord\b|\btenant\b|\blease\b", 2),
        (r"\bintellectual property\b|\bcopyright\b|\btrademark\b|\bpatent\b", 3),
        (r"\bcriminal\b|\barrest\b|\bcharge(d|s)?\b|\bprosecution\b|\bdefendant\b|\bplaintiff\b", 3),
        (r"\bemployment\b|\bworkplace\b|\bwage(s)?\b|\bsalary\b|\bbenefits?\b|\bunion\b", 2),
        (r"\bdivorce\b|\bcustody\b|\balimony\b|\bchild support\b|\bmarriage\b|\bspouse\b", 2),
        (r"\bimmigration\b|\bvisa\b|\bdeportation\b|\bcitizenship\b|\bgreen card\b", 2),
        (r"\bprivacy\b|\bdata protection\b|\bGDPR\b|\bconsent\b|\bpersonal data\b", 2),
        (r"\bcan i\b|\bam i allowed\b|\bis it legal\b|\bdo i have the right\b|\bwhat are my options\b", 2),
        (r"\bwhat (happens|should i do|can i do|are my rights)\b", 2),
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

    for intent, patterns in INTENT_RULES.items():
        for pattern, weight in patterns:
            if re.search(pattern, q):
                scores[intent] += weight

    if debug:
        print("Intent scores:", scores)

    max_score = max(scores.values())

    if max_score == 0:
        return "legal_question"  # safe fallback — most queries are legal

    # Handle ties using priority order
    candidates = [intent for intent, score in scores.items() if score == max_score]

    for intent in PRIORITY_ORDER:
        if intent in candidates:
            if debug:
                print("Chosen intent:", intent)
            return intent

    return "legal_question"
