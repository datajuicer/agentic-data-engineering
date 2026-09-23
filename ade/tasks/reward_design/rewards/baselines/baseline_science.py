from __future__ import annotations

import re
import sys
import unicodedata
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[4]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

def compute_score(
    question_prompt: str,
    response_content: str,
    extracted_answer: str,
    ground_truth: str,
    response_length_tokens: int,
    max_response_length_tokens: int,
) -> float:
    del question_prompt, response_content, response_length_tokens, max_response_length_tokens
    predicted = str(extracted_answer or "")
    if not predicted:
        return 0.0
    expected = str(ground_truth or "").strip()
    return 1.0 if _normalize(predicted) == _normalize(expected) else 0.0


def _normalize(value: Any) -> str:
    text = unicodedata.normalize("NFKC", str(value or "")).strip().lower()
    text = text.replace("\\mu", "μ").replace("µ", "μ").replace("−", "-")
    text = text.replace("\\left", "").replace("\\right", "")
    text = re.sub(r"\s+", "", text)
    text = re.sub(r"(?<=\d)\.0+(?=\D|$)", "", text)
    text = re.sub(r"(?<=\.)(\d*?[1-9])0+(?=\D|$)", r"\1", text)
    return text.rstrip(".")
