"""Shared science answer extraction used by canonical dataset tasks."""

from __future__ import annotations

import re
from typing import Any

from .math_exact import find_boxed_content


def normalize_choice_letter(answer: Any) -> str:
    if answer is None:
        return ""
    return str(answer).strip().upper().rstrip(".").rstrip("/")


def extract_choice_letter(text: str, *, is_reference: bool = False) -> str:
    text = str(text or "").strip()
    if not text:
        return ""
    boxed = find_boxed_content(text)
    if boxed is not None:
        text = boxed
    patterns = [
        r"(?:^|\n)\s*Answer\s*:\s*\$?\s*([A-Z])\b",
        r"(?:^|\n)\s*(?:Final\s+answer|The\s+answer)\s*(?:is|:)?\s*\$?\s*([A-Z])\b",
        r"\b(?:answer is|answer:)\s*\(?([A-Z])\)?\b",
    ]
    if is_reference:
        patterns.extend(
            [
                r"^\s*\$?\s*([A-Z])\s*[\).:\-]",
                r"^\s*\$?\s*([A-Z])\s*$",
            ]
        )
    for pattern in patterns:
        matches = list(re.finditer(pattern, text, flags=re.IGNORECASE))
        if matches:
            return normalize_choice_letter(matches[-1].group(1))
    matches = list(re.finditer(r"\b([A-Z])\b", text))
    return normalize_choice_letter(matches[-1].group(1)) if matches else ""


def extract_gpqa_choice_answer(text: str) -> tuple[str, str]:
    boxed = find_boxed_content(text)
    if boxed is not None:
        predicted = extract_choice_letter(boxed)
        if predicted:
            return predicted, "boxed"
    predicted = _extract_gpqa_boxed_like_choice(text)
    if predicted:
        return predicted, "boxed_like"
    predicted = _extract_gpqa_answer_phrase(text)
    return predicted, "answer_phrase" if predicted else "none"


def extract_supergpqa_choice_answer(text: str) -> tuple[str, str]:
    boxed = find_boxed_content(text)
    if boxed is not None:
        predicted = extract_choice_letter(boxed)
        if predicted and re.fullmatch(r"[A-J]", predicted):
            return predicted, "boxed"
    predicted = _extract_a_to_j_answer_phrase(text)
    return predicted, "answer_phrase" if predicted else "none"


def extract_boxed_science_answer(text: str) -> tuple[str, str]:
    boxed = find_boxed_content(text)
    return (boxed, "boxed") if boxed is not None else ("", "none")


def _answer_tail(text: str) -> str:
    tail = str(text or "").split("</think>")[-1].strip()
    return re.sub(r"(?:<\|im_end\|>|<\|endoftext\|>)\s*$", "", tail).strip()


def _extract_gpqa_boxed_like_choice(text: str) -> str:
    tail = _answer_tail(text)
    matches: list[tuple[int, str]] = []
    for match in re.finditer(
        r"(?<!\\)\bboxed\s*\{\s*([A-D])\s*\}", tail, flags=re.IGNORECASE
    ):
        matches.append((match.start(), match.group(1).upper()))
    braced_tail = re.search(
        r"(?:^|\s)\{\s*([A-D])\s*\}\s*$", tail, flags=re.IGNORECASE
    )
    if braced_tail:
        matches.append((braced_tail.start(), braced_tail.group(1).upper()))
    return max(matches, key=lambda item: item[0])[1] if matches else ""


def _extract_gpqa_answer_phrase(text: str) -> str:
    tail = _answer_tail(text)
    letter = r"\**\$?\s*\(?([A-D])\)?\**(?:\)|\b)"
    patterns = [
        rf"(?:^|\n)\s*(?:#{{1,6}}\s*)?(?:[-*]\s*)?(?:\*\*)?\s*(?:final\s+answer|answer)\s*(?:\*\*)?\s*(?:is|:)?\s*{letter}",
        rf"\b(?:final\s+answer|answer)\s*(?:is|:)\s*{letter}",
        rf"\b(?:the\s+)?correct\s+(?:answer|option|choice)\s*(?:is|would\s+be|should\s+be|:)\s*{letter}",
        rf"\b(?:therefore|thus|hence|so)\b[^\n.]{{0,180}}\b(?:the\s+)?(?:correct\s+)?(?:answer|option|choice)\s*(?:is|:)\s*{letter}",
        r"\b(?:therefore|thus|hence|so)\b[^\n.]{0,180}\b(?:option|choice)\s*\(?([A-D])\)?\s+(?:is|would\s+be)\s+(?:the\s+)?(?:correct|best|closest)\b",
        rf"\b(?:I\s+choose|I\s+select|choose|select)\s+(?:option\s+)?{letter}",
    ]
    return _latest_choice_match(tail, patterns, label_range="A-D")


def _extract_a_to_j_answer_phrase(text: str) -> str:
    tail = str(text or "").split("</think>")[-1].strip()
    letter = r"\**\$?\s*\(?([A-J])\)?\**(?:\)|\b)"
    patterns = [
        rf"(?:^|\n)\s*(?:#{{1,6}}\s*)?(?:[-*]\s*)?(?:\*\*)?\s*(?:final\s+answer|answer)\s*(?:\*\*)?\s*(?:is|:)?\s*{letter}",
        rf"\b(?:final\s+answer|answer)\s*(?:is|:)\s*{letter}",
        rf"\b(?:the\s+)?correct\s+(?:answer|option|choice)\s*(?:is|would\s+be|should\s+be|:)\s*{letter}",
        rf"\b(?:I\s+choose|I\s+select|choose|select)\s+(?:option\s+)?{letter}",
    ]
    return _latest_choice_match(tail, patterns, label_range="A-J")


def _latest_choice_match(text: str, patterns: list[str], *, label_range: str) -> str:
    matches: list[tuple[int, str]] = []
    for pattern in patterns:
        for match in re.finditer(pattern, text, flags=re.IGNORECASE):
            for value in reversed(match.groups()):
                if value and re.fullmatch(f"[{label_range}]", value, flags=re.IGNORECASE):
                    matches.append((match.start(), value.upper()))
                    break
    return max(matches, key=lambda item: item[0])[1] if matches else ""
