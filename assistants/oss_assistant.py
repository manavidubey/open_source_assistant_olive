"""
Open Source Assistant — with automatic provider fallback.

Tries providers in order (Groq → Cerebras → Together → NVIDIA → HF)
and falls back to the next if one fails.
"""

from __future__ import annotations
from typing import Optional

from assistants.base import BaseAssistant
from observability.logger import ObservabilityLogger


class OSSAssistant(BaseAssistant):
    """
    Open-source model assistant with multi-provider fallback.
    """

    def __init__(
        self,
        model_id: str = "",
        provider: str = "groq",
        api_key: str = "",
        base_url: str | None = None,
        fallback_chain: list | None = None,
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
        self._provider = provider
        self._fallback_chain = fallback_chain or []

        # Build primary client
        self._clients = []
        self._clients.append(self._make_client(provider, api_key, base_url, model_id))

        # Build fallback clients
        for fb in self._fallback_chain:
            self._clients.append(self._make_client(
                fb.name, fb.api_key, fb.base_url, fb.oss_model
            ))

    @staticmethod
    def _make_client(provider: str, api_key: str, base_url: str | None, model_id: str) -> dict:
        """Create a client dict for a provider."""
        if provider in ("groq", "cerebras", "together", "nvidia", "openai"):
            from openai import OpenAI
            return {
                "type": "openai_compat",
                "client": OpenAI(api_key=api_key, base_url=base_url),
                "model": model_id,
                "provider": provider,
            }
        else:
            from huggingface_hub import InferenceClient
            return {
                "type": "hf",
                "client": InferenceClient(model=model_id, token=api_key or None),
                "model": model_id,
                "provider": provider,
            }

    def _generate(self, messages: list[dict]) -> tuple[str, dict]:
        """Generate with automatic fallback across providers."""
        last_error = None

        for client_info in self._clients:
            try:
                if client_info["type"] == "openai_compat":
                    return self._gen_openai(client_info, messages)
                else:
                    return self._gen_hf(client_info, messages)
            except Exception as e:
                last_error = e
                provider = client_info["provider"]
                print(f"  ⚠ {provider} failed: {e.__class__.__name__}: {str(e)[:80]}")
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
    def _gen_hf(info: dict, messages: list[dict]) -> tuple[str, dict]:
        resp = info["client"].chat_completion(
            messages=messages, max_tokens=1024, temperature=0.7, top_p=0.9,
        )
        text = resp.choices[0].message.content.strip()
        meta = {
            "input_tokens": getattr(resp.usage, "prompt_tokens", 0),
            "output_tokens": getattr(resp.usage, "completion_tokens", 0),
            "model": info["model"],
            "provider": info["provider"],
        }
        return text, meta
