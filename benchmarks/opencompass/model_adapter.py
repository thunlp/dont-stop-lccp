"""OpenCompass Hugging Face adapter with thinking explicitly disabled."""

from typing import List, Optional

import torch

from opencompass.models.huggingface_above_v4_33 import (
    HuggingFacewithChatTemplate,
    _convert_chat_messages,
    _get_stopping_criteria,
)
from opencompass.registry import MODELS


@MODELS.register_module()
class NoThinkingChatModel(HuggingFacewithChatTemplate):
    """Apply the checkpoint chat template with ``enable_thinking=False``."""

    batch_padding = True

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if (getattr(self.model.config, "attention_sink_enabled", False)
                and getattr(self.model.config, "_attn_implementation", None) == "flash_attention_2"):
            from lccp.backend import require_native_sink
            require_native_sink(self.model)

    def _render(self, inputs: List[str]) -> list[str]:
        messages = _convert_chat_messages(inputs)
        return [
            self.tokenizer.apply_chat_template(
                message,
                add_generation_prompt=True,
                tokenize=False,
                enable_thinking=False,
            )
            for message in messages
        ]

    def generate(
        self,
        inputs: List[str],
        max_out_len: int,
        min_out_len: Optional[int] = None,
        stopping_criteria: Optional[List[str]] = None,
        **kwargs,
    ) -> List[str]:
        rendered = self._render(inputs)
        batch_size = len(rendered)
        tokens = self.tokenizer.batch_encode_plus(
            rendered,
            return_tensors="pt",
            padding=True,
            truncation=True,
            add_special_tokens=False,
            max_length=self.max_seq_len,
        )
        tokens = {key: value.to(self.model.device) for key, value in tokens.items()}

        generation_kwargs = self.generation_kwargs.copy()
        generation_kwargs.update(kwargs)
        stop_words = list(set((stopping_criteria or []) + self.stop_words))
        if stop_words:
            generation_kwargs["stopping_criteria"] = _get_stopping_criteria(
                stop_words, self.tokenizer, batch_size
            )
        generation_kwargs["max_new_tokens"] = max_out_len
        if min_out_len is not None:
            generation_kwargs["min_new_tokens"] = min_out_len
        generation_kwargs["pad_token_id"] = self.tokenizer.pad_token_id

        with torch.inference_mode():
            outputs = self.model.generate(**tokens, **generation_kwargs)
        outputs = outputs[:, tokens["input_ids"].shape[1] :]
        decoded = self.tokenizer.batch_decode(outputs, skip_special_tokens=True)
        for stop_word in stop_words:
            decoded = [text.split(stop_word)[0] for text in decoded]
        return decoded

    def get_token_len(self, prompt: str) -> int:
        rendered = self._render([prompt])[0]
        return len(
            self.tokenizer(
                rendered,
                add_special_tokens=False,
                truncation=False,
            ).input_ids
        )
