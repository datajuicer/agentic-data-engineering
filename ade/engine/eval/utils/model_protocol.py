"""Validate a resolved checkpoint response protocol against its tokenizer."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping


def validate_thinking_protocol(
    tokenizer: Any,
    model_protocol: Mapping[str, object] | None,
) -> tuple[str, str] | None:
    """Return resolved tags when thinking-tag response splitting is enabled.

    The protocol is resolved by the experiment compiler. Evaluation only
    verifies the checkpoint; it never mutates tokenizer state.
    """

    if model_protocol is None:
        return None
    if set(model_protocol) != {"thinking"}:
        raise ValueError("model_protocol requires exactly one thinking section")
    thinking = model_protocol["thinking"]
    if not isinstance(thinking, Mapping):
        raise ValueError("model_protocol.thinking must be a mapping")
    mode = thinking.get("mode")
    if mode == "disabled":
        if set(thinking) != {"mode", "reasoning_parser"}:
            raise ValueError(
                "disabled thinking protocol requires mode and reasoning_parser"
            )
        if thinking["reasoning_parser"] != "none":
            raise ValueError("disabled thinking protocol requires reasoning_parser=none")
        return None
    required = {
        "mode",
        "tag_encoding",
        "activation",
        "open_tag",
        "close_tag",
        "reasoning_parser",
    }
    if mode != "tagged" or set(thinking) != required:
        raise ValueError("tagged thinking protocol is incomplete")

    open_tag = _required_text(thinking, "open_tag")
    close_tag = _required_text(thinking, "close_tag")
    if open_tag == close_tag:
        raise ValueError("thinking tags must be distinct")
    tag_encoding = _required_text(thinking, "tag_encoding")
    activation = _required_text(thinking, "activation")
    reasoning_parser = _required_text(thinking, "reasoning_parser")
    if reasoning_parser not in {"none", "qwen3"}:
        raise ValueError("thinking reasoning_parser must be none or qwen3")
    if tag_encoding == "text" and reasoning_parser != "none":
        raise ValueError("text/chat-template thinking requires reasoning_parser=none")
    tags = (open_tag, close_tag)
    encoded = {
        tag: tokenizer.encode(tag, add_special_tokens=False)
        for tag in tags
    }
    special_by_tag = {
        tag: _is_special_token(tokenizer, tag, encoded[tag][0])
        if len(encoded[tag]) == 1
        else False
        for tag in tags
    }
    named_special_tokens = set(tokenizer.all_special_tokens)
    additional_special_tokens = set(tokenizer.additional_special_tokens)

    if (tag_encoding, activation) == ("special_tokens", "learned"):
        invalid = [
            tag
            for tag in tags
            if (
                len(encoded[tag]) != 1
                or not special_by_tag[tag]
                or tag not in named_special_tokens
                or tag not in additional_special_tokens
            )
        ]
        if invalid:
            raise ValueError(
                "tagged special-token checkpoint protocol mismatch: "
                f"invalid={invalid}; encoded_as={encoded}"
            )
        return tags

    if (tag_encoding, activation) == ("text", "chat_template"):
        template = str(tokenizer.chat_template or "")
        missing = [tag for tag in tags if tag not in template]
        unexpectedly_special = [
            tag
            for tag in tags
            if (
                special_by_tag[tag]
                or tag in named_special_tokens
                or tag in additional_special_tokens
            )
        ]
        if missing or unexpectedly_special:
            raise ValueError(
                "tagged text/chat-template checkpoint protocol mismatch: "
                f"missing_from_template={missing}; "
                f"unexpectedly_special={unexpectedly_special}"
            )
        return tags

    raise ValueError(
        "thinking protocol requires special_tokens+learned or "
        "text+chat_template"
    )


def _required_text(value: Mapping[str, object], key: str) -> str:
    item = value.get(key)
    if not isinstance(item, str) or not item:
        raise ValueError(f"model_protocol.thinking.{key} must be non-empty")
    return item


def is_special_token(tokenizer: Any, token: str, token_id: int) -> bool:
    """Recognize both named special tokens and persisted AddedToken specials."""

    return _is_special_token(tokenizer, token, token_id)


def validate_embedding_range(
    tokenizer: Any,
    model_path: str,
    thinking_tags: tuple[str, str] | None,
) -> None:
    """Fail closed when persisted thinking-token IDs exceed model embeddings."""

    if thinking_tags is None:
        return
    from transformers import AutoConfig

    config = AutoConfig.from_pretrained(
        model_path,
        local_files_only=Path(model_path).exists(),
        trust_remote_code=True,
    )
    embedding_rows = getattr(config, "vocab_size", None)
    if type(embedding_rows) is not int or embedding_rows < 1:
        raise ValueError("checkpoint config does not expose a valid vocab_size")
    token_ids = {
        tag: tokenizer.encode(tag, add_special_tokens=False)[0]
        for tag in thinking_tags
    }
    invalid = {
        tag: token_id
        for tag, token_id in token_ids.items()
        if token_id < 0 or token_id >= embedding_rows
    }
    if invalid:
        raise ValueError(
            "thinking-token IDs exceed checkpoint embedding range: "
            f"invalid={invalid}; embedding_rows={embedding_rows}"
        )


def _is_special_token(tokenizer: Any, token: str, token_id: int) -> bool:
    if token in set(tokenizer.all_special_tokens):
        return True
    decoder = getattr(tokenizer, "added_tokens_decoder", {})
    added = decoder.get(token_id) if hasattr(decoder, "get") else None
    return bool(getattr(added, "special", False))
