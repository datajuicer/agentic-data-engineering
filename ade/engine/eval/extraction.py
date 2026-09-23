from __future__ import annotations


def split_thinking_answer(
    text: str,
    *,
    enabled: bool,
    open_tag: str = "<think>",
    close_tag: str = "</think>",
) -> tuple[str, str]:
    text = text or ""
    if not enabled:
        return "", text
    if close_tag in text:
        before, answer = text.rsplit(close_tag, 1)
        thinking = before.split(open_tag, 1)[-1] if open_tag in before else before
        return thinking, answer
    return "", text
