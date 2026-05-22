"""
Frontier Model Assistant — with automatic provider fallback.

Tries providers in order and falls back to the next if one fails.
Supports Groq, Cerebras, Together, NVIDIA, Gemini, and OpenAI.
"""

from __future__ import annotations
from typing import Optional

from assistants.base import BaseAssistant
from observability.logger import ObservabilityLogger


class FrontierAssistant(BaseAssistant):
    """
    Frontier model assistant with multi-provider fallback.
    Uses larger models (70B+) for the comparison benchmark.
    """

    def __init__(
        self,
        provider: str = "groq",
        model_id: str = "",
        api_key: str = "",
        base_url: str | None = None,
        fallback_chain: list | None = None,
        # Legacy params
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
        # Resolve primary
        if not model_id:
            if provider == "gemini":
                model_id = gemini_model
            elif provider == "openai" and not api_key:
                model_id = openai_model
                api_key = openai_api_key

        super().__init__(
            model_name=model_id,
            provider=provider,
            system_prompt=system_prompt,
            max_memory_turns=max_memory_turns,
            enable_guardrails=enable_guardrails,
            enable_tools=enable_tools,
            logger=logger,
        )

        self._model_id = model_id
        self._primary_provider = provider
        self._fallback_chain = fallback_chain or []

        # Build client chain
        self._clients = []
        self._clients.append(self._make_client(provider, api_key, base_url, model_id))

        for fb in self._fallback_chain:
            self._clients.append(self._make_client(
                fb.name, fb.api_key, fb.base_url, fb.frontier_model
            ))

    @staticmethod
    def _make_client(provider: str, api_key: str, base_url: str | None, model_id: str) -> dict:
        if provider in ("groq", "cerebras", "together", "nvidia", "openai"):
            from openai import OpenAI
            return {
                "type": "openai_compat",
                "client": OpenAI(api_key=api_key, base_url=base_url),
                "model": model_id,
                "provider": provider,
            }
        elif provider == "gemini":
            from google import genai
            return {
                "type": "gemini",
                "client": genai.Client(api_key=api_key),
                "model": model_id,
                "provider": "gemini",
            }
        raise ValueError(f"Unsupported provider: {provider}")

    def _generate(self, messages: list[dict]) -> tuple[str, dict]:
        """Generate with automatic fallback."""
        last_error = None

        for client_info in self._clients:
            try:
                if client_info["type"] == "openai_compat":
                    return self._gen_openai(client_info, messages)
                elif client_info["type"] == "gemini":
                    return self._gen_gemini(client_info, messages)
            except Exception as e:
                last_error = e
                p = client_info["provider"]
                print(f"  ⚠ {p} failed: {e.__class__.__name__}: {str(e)[:80]}")
                continue

        raise RuntimeError(f"All providers failed. Last error: {last_error}")

    @staticmethod
    def _gen_openai(info: dict, messages: list[dict]) -> tuple[str, dict]:
        resp = info["client"].chat.completions.create(
            model=info["model"],
            messages=messages,
            max_tokens=1024,
            temperature=0.7,
            top_p=0.9,
        )
        text = resp.choices[0].message.content.strip()
        meta = {
            "input_tokens": resp.usage.prompt_tokens if resp.usage else 0,
            "output_tokens": resp.usage.completion_tokens if resp.usage else 0,
            "model": info["model"],
            "provider": info["provider"],
        }
        return text, meta

    @staticmethod
    def _gen_gemini(info: dict, messages: list[dict]) -> tuple[str, dict]:
        system_instruction = None
        contents = []
        for msg in messages:
            if msg["role"] == "system":
                system_instruction = (system_instruction or "") + "\n" + msg["content"]
            else:
                role = "user" if msg["role"] == "user" else "model"
                contents.append({"role": role, "parts": [{"text": msg["content"]}]})

        config = {"temperature": 0.7, "top_p": 0.9, "max_output_tokens": 1024}
        if system_instruction:
            config["system_instruction"] = system_instruction.strip()

        resp = info["client"].models.generate_content(
            model=info["model"], contents=contents, config=config,
        )
        text = resp.text.strip() if resp.text else ""
        meta = {
            "input_tokens": getattr(resp.usage_metadata, "prompt_token_count", 0) if resp.usage_metadata else 0,
            "output_tokens": getattr(resp.usage_metadata, "candidates_token_count", 0) if resp.usage_metadata else 0,
            "model": info["model"],
            "provider": info["provider"],
        }
        return text, meta
