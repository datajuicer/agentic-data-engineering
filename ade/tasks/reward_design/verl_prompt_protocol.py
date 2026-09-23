"""VERL consumers for ADE's raw-completion prompt protocol."""

from __future__ import annotations

from typing import Any

import datasets

from verl.experimental.agent_loop.single_turn_agent_loop import SingleTurnAgentLoop
from verl.utils.dataset.rl_dataset import RLHFDataset
from verl.utils.tokenizer import normalize_token_ids


def raw_user_content(messages: object) -> str:
    """Return the sole user message without adding protocol tokens."""

    if not isinstance(messages, list) or len(messages) != 1:
        raise ValueError("raw_completion requires exactly one prompt message")
    message = messages[0]
    if not isinstance(message, dict) or set(message) != {"role", "content"}:
        raise ValueError("raw_completion prompt message requires exactly role/content")
    if message.get("role") != "user":
        raise ValueError("raw_completion prompt message must have role=user")
    content = message.get("content")
    if not isinstance(content, str) or not content:
        raise ValueError("raw_completion user content must be non-empty text")
    return content


class RawCompletionDataset(RLHFDataset):
    """Filter using the same raw tokenization consumed by the rollout loop."""

    def maybe_filter_out_long_prompts(
        self, dataframe: datasets.Dataset | None = None
    ) -> datasets.Dataset:
        if dataframe is None:
            dataframe = self.dataframe
        if not self.filter_overlong_prompts:
            return dataframe

        def prompt_length(row: dict[str, Any]) -> int:
            content = raw_user_content(row[self.prompt_key])
            encoded = self.tokenizer(
                content,
                add_special_tokens=False,
                return_attention_mask=False,
            )
            return len(normalize_token_ids(encoded["input_ids"]))

        filtered = dataframe.filter(
            lambda row: prompt_length(row) <= self.max_prompt_length,
            num_proc=self.num_workers,
            desc=(
                "Filtering raw-completion prompts longer than "
                f"{self.max_prompt_length} tokens"
            ),
        )
        print(f"filter dataset len: {len(filtered)}")
        return filtered


class RawCompletionAgentLoop(SingleTurnAgentLoop):
    """Tokenize the raw user content with no system or chat template."""

    async def apply_chat_template(
        self,
        messages: list[dict],
        tools: list[dict] | None = None,
        images: list[Any] | None = None,
        videos: list[Any] | None = None,
        audios: list[Any] | None = None,
        mm_processor_kwargs: dict[str, Any] | None = None,
        remove_system_prompt: bool = False,
    ) -> list[int]:
        del mm_processor_kwargs
        if tools or images or videos or audios or remove_system_prompt:
            raise ValueError("raw_completion supports text-only single-turn prompts")
        content = raw_user_content(messages)
        encoded = await self.loop.run_in_executor(
            None,
            lambda: self.tokenizer(
                content,
                add_special_tokens=False,
                return_attention_mask=False,
            )["input_ids"],
        )
        prompt_ids = normalize_token_ids(encoded)
        if len(prompt_ids) > self.prompt_length:
            raise ValueError(
                f"raw_completion prompt has {len(prompt_ids)} tokens, exceeding "
                f"rollout.prompt_length={self.prompt_length}"
            )
        return prompt_ids
