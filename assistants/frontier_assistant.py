"""
Frontier Model Assistant — Google Gemini / OpenAI GPT-4.

Supports both Gemini (primary) and OpenAI (fallback) as frontier
model providers, with automatic provider detection from config.
"""

from __future__ import annotations
from typing import Optional

from assistants.base import BaseAssistant
from observability.logger import ObservabilityLogger


class FrontierAssistant(BaseAssistant):
    """
    Frontier model assistant with support for Gemini and OpenAI.
    """

    def __init__(
        self,
        provider: str = "gemini",
        gemini_api_key: str = "",
        gemini_model: str = "gemini-2.0-flash",
        openai_api_key: str = "",
        openai_model: str = "gpt-4.1-mini",
        system_prompt: str = "",
        max_memory_turns: int = 20,
        enable_guardrails: bool = True,
        enable_tools: bool = True,
        logger: Optional[ObservabilityLogger] = None,
    ):
        # Determine active provider and model
        if provider == "gemini" and gemini_api_key:
            active_model = gemini_model
            active_provider = "gemini"
        elif openai_api_key:
            active_model = openai_model
            active_provider = "openai"
        else:
            active_model = gemini_model
            active_provider = "gemini"

        super().__init__(
            model_name=active_model,
            provider=active_provider,
            system_prompt=system_prompt,
            max_memory_turns=max_memory_turns,
            enable_guardrails=enable_guardrails,
            enable_tools=enable_tools,
            logger=logger,
        )

        self._active_provider = active_provider
        self._client = None

        # Initialize the appropriate client
        if active_provider == "gemini":
            from google import genai
            self._client = genai.Client(api_key=gemini_api_key)
            self._model_id = gemini_model
        elif active_provider == "openai":
            from openai import OpenAI
            self._client = OpenAI(api_key=openai_api_key)
            self._model_id = openai_model

    def _generate(self, messages: list[dict]) -> tuple[str, dict]:
        """Generate response using the configured frontier API."""
        if self._active_provider == "gemini":
            return self._generate_gemini(messages)
        elif self._active_provider == "openai":
            return self._generate_openai(messages)
        raise ValueError(f"No frontier provider configured")

    def _generate_gemini(self, messages: list[dict]) -> tuple[str, dict]:
        """Generate using Google Gemini API."""
        # Gemini uses a different message format
        # Extract system instruction and convert messages
        system_instruction = None
        gemini_contents = []

        for msg in messages:
            if msg["role"] == "system":
                if system_instruction is None:
                    system_instruction = msg["content"]
                else:
                    system_instruction += "\n" + msg["content"]
            else:
                role = "user" if msg["role"] == "user" else "model"
                gemini_contents.append({
                    "role": role,
                    "parts": [{"text": msg["content"]}],
                })

        config = {
            "temperature": 0.7,
            "top_p": 0.9,
            "max_output_tokens": 1024,
        }
        if system_instruction:
            config["system_instruction"] = system_instruction

        response = self._client.models.generate_content(
            model=self._model_id,
            contents=gemini_contents,
            config=config,
        )

        text = response.text.strip() if response.text else ""
        metadata = {
            "input_tokens": getattr(response.usage_metadata, "prompt_token_count", 0) if response.usage_metadata else 0,
            "output_tokens": getattr(response.usage_metadata, "candidates_token_count", 0) if response.usage_metadata else 0,
            "model": self._model_id,
        }
        return text, metadata

    def _generate_openai(self, messages: list[dict]) -> tuple[str, dict]:
        """Generate using OpenAI API."""
        response = self._client.chat.completions.create(
            model=self._model_id,
            messages=messages,
            max_tokens=1024,
            temperature=0.7,
            top_p=0.9,
        )

        text = response.choices[0].message.content.strip()
        metadata = {
            "input_tokens": response.usage.prompt_tokens if response.usage else 0,
            "output_tokens": response.usage.completion_tokens if response.usage else 0,
            "model": self._model_id,
        }
        return text, metadata
