"""
Open Source Assistant — Qwen 2.5 via Hugging Face Inference API.

Uses the HF Inference API for zero-setup inference, with fallback
to local transformers pipeline if configured.
"""

from __future__ import annotations
from typing import Optional
from huggingface_hub import InferenceClient

from assistants.base import BaseAssistant
from observability.logger import ObservabilityLogger


class OSSAssistant(BaseAssistant):
    """
    Open-source model assistant using Hugging Face Inference API.
    Default model: Qwen/Qwen2.5-0.5B-Instruct
    """

    def __init__(
        self,
        model_id: str = "Qwen/Qwen2.5-0.5B-Instruct",
        hf_token: str = "",
        system_prompt: str = "",
        max_memory_turns: int = 20,
        enable_guardrails: bool = True,
        enable_tools: bool = True,
        logger: Optional[ObservabilityLogger] = None,
    ):
        super().__init__(
            model_name=model_id,
            provider="oss",
            system_prompt=system_prompt,
            max_memory_turns=max_memory_turns,
            enable_guardrails=enable_guardrails,
            enable_tools=enable_tools,
            logger=logger,
        )
        self.model_id = model_id
        self.client = InferenceClient(
            model=model_id,
            token=hf_token or None,
        )

    def _generate(self, messages: list[dict]) -> tuple[str, dict]:
        """Generate response using HF Inference API chat completion."""
        response = self.client.chat_completion(
            messages=messages,
            max_tokens=1024,
            temperature=0.7,
            top_p=0.9,
        )

        text = response.choices[0].message.content.strip()
        metadata = {
            "input_tokens": getattr(response.usage, "prompt_tokens", 0),
            "output_tokens": getattr(response.usage, "completion_tokens", 0),
            "model": self.model_id,
        }
        return text, metadata
