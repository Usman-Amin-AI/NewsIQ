import datetime
import json
import os
import re
from pathlib import Path
from typing import Any, Dict, List

from langchain_core.documents import Document


def _normalize_text(text: str) -> List[str]:
    normalized = text.lower()
    normalized = re.sub(r"[^a-z0-9\s]", " ", normalized)
    tokens = [token for token in normalized.split() if len(token) > 2]
    return tokens


def compute_groundedness(answer: str, source_documents: List[Document]) -> float:
    if not answer or not source_documents:
        return 0.0

    answer_tokens = set(_normalize_text(answer))
    source_text = " ".join(doc.page_content for doc in source_documents)
    source_tokens = set(_normalize_text(source_text))
    if not answer_tokens:
        return 0.0

    overlap = answer_tokens.intersection(source_tokens)
    return float(len(overlap)) / len(answer_tokens)


def evaluate_answer(
    question: str,
    answer: str,
    citations: List[str],
    source_documents: List[Document],
) -> Dict[str, Any]:
    groundedness_score = compute_groundedness(answer, source_documents)
    source_count = len({doc.metadata.get("source") for doc in source_documents if doc.metadata.get("source")})
    citation_count = len(citations)
    evaluation = {
        "question": question,
        "answer": answer,
        "groundedness_score": round(groundedness_score, 3),
        "source_count": source_count,
        "citation_count": citation_count,
        "has_sources": source_count > 0,
        "has_citations": citation_count > 0,
        "summary": "High grounding" if groundedness_score >= 0.45 else "Low grounding, potential hallucination",
    }
    return evaluation


def log_evaluation(
    evaluation: Dict[str, Any],
    log_path: str,
) -> None:
    Path(os.path.dirname(log_path)).mkdir(parents=True, exist_ok=True)
    record = {
        "timestamp": datetime.datetime.utcnow().isoformat() + "Z",
        **evaluation,
    }
    with open(log_path, "a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, ensure_ascii=False) + "\n")
