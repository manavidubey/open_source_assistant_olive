"""
AI Personal Assistant — Configuration Module

Supports Groq, Cerebras, Together AI, NVIDIA, Gemini, OpenAI, and HF.
Provides automatic fallback chain across providers.
"""

import os
from dataclasses import dataclass, field
from dotenv import load_dotenv

load_dotenv()

# ── Provider base URLs (all OpenAI-compatible) ──────────────────
PROVIDER_URLS = {
    "groq": "https://api.groq.com/openai/v1",
    "cerebras": "https://api.cerebras.ai/v1",
    "together": "https://api.together.xyz/v1",
    "nvidia": "https://integrate.api.nvidia.com/v1",
    "openai": "https://api.openai.com/v1",
}

# ── Default models per provider ─────────────────────────────────
DEFAULT_MODELS = {
    "groq": {
        "oss": "llama-3.1-8b-instant",
        "frontier": "llama-3.3-70b-versatile",
    },
    "cerebras": {
        "oss": "llama-3.1-8b",
        "frontier": "llama-3.3-70b",
    },
    "together": {
        "oss": "Qwen/Qwen2.5-7B-Instruct-Turbo",
        "frontier": "meta-llama/Llama-3.3-70B-Instruct-Turbo",
    },
    "nvidia": {
        "oss": "meta/llama-3.1-8b-instruct",
        "frontier": "meta/llama-3.3-70b-instruct",
    },
    "openai": {
        "oss": "gpt-4.1-mini",
        "frontier": "gpt-4.1",
    },
    "gemini": {
        "oss": "gemini-2.0-flash-lite",
        "frontier": "gemini-2.0-flash",
    },
    "huggingface": {
        "oss": "Qwen/Qwen2.5-0.5B-Instruct",
        "frontier": None,
    },
}


@dataclass
class ProviderConfig:
    """Configuration for a single provider."""
    name: str
    api_key: str
    base_url: str | None
    oss_model: str
    frontier_model: str


@dataclass
class Config:
    """Application configuration loaded from environment variables."""

    # --- Provider Keys ---
    groq_api_key: str = ""
    cerebras_api_key: str = ""
    together_api_key: str = ""
    nvidia_api_key: str = ""
    gemini_api_key: str = ""
    openai_api_key: str = ""
    hf_api_token: str = ""

    # --- Model Overrides (optional) ---
    oss_model: str = ""
    frontier_model: str = ""

    # --- Legacy ---
    hf_model_id: str = "Qwen/Qwen2.5-0.5B-Instruct"

    # --- Memory ---
    max_memory_turns: int = 20
    summary_threshold: int = 15

    # --- Safety ---
    enable_guardrails: bool = True

    # --- Observability ---
    log_level: str = "INFO"
    log_file: str = "logs/assistant.log"

    # --- System Prompt ---
    system_prompt: str = (
        "You are a helpful, harmless, and honest AI personal assistant. "
        "You provide accurate, well-reasoned answers. If you don't know "
        "something, you say so rather than making up information. "
        "You refuse to help with harmful, illegal, or unethical requests. "
        "You are respectful of all people regardless of their background."
    )

    @classmethod
    def from_env(cls) -> "Config":
        """Load configuration from environment variables."""
        return cls(
            groq_api_key=os.getenv("GROQ_API_KEY", ""),
            cerebras_api_key=os.getenv("CEREBRAS_API_KEY", ""),
            together_api_key=os.getenv("TOGETHER_API_KEY", ""),
            nvidia_api_key=os.getenv("NVIDIA_API_KEY", ""),
            gemini_api_key=os.getenv("GEMINI_API_KEY", ""),
            openai_api_key=os.getenv("OPENAI_API_KEY", ""),
            hf_api_token=os.getenv("HF_API_TOKEN", ""),
            oss_model=os.getenv("OSS_MODEL", ""),
            frontier_model=os.getenv("FRONTIER_MODEL", ""),
            hf_model_id=os.getenv("HF_MODEL_ID", "Qwen/Qwen2.5-0.5B-Instruct"),
            max_memory_turns=int(os.getenv("MAX_MEMORY_TURNS", "20")),
            summary_threshold=int(os.getenv("SUMMARY_THRESHOLD", "15")),
            enable_guardrails=os.getenv("ENABLE_GUARDRAILS", "true").lower() == "true",
            log_level=os.getenv("LOG_LEVEL", "INFO"),
            log_file=os.getenv("LOG_FILE", "logs/assistant.log"),
        )

    def _get_provider_chain(self) -> list[ProviderConfig]:
        """
        Build ordered list of available providers for fallback.
        Priority: Groq → Cerebras → Together → NVIDIA → Gemini → OpenAI → HF
        """
        chain = []
        provider_keys = [
            ("groq", self.groq_api_key),
            ("cerebras", self.cerebras_api_key),
            ("together", self.together_api_key),
            ("nvidia", self.nvidia_api_key),
            ("gemini", self.gemini_api_key),
            ("openai", self.openai_api_key),
            ("huggingface", self.hf_api_token),
        ]

        for name, key in provider_keys:
            if key:
                defaults = DEFAULT_MODELS.get(name, {})
                chain.append(ProviderConfig(
                    name=name,
                    api_key=key,
                    base_url=PROVIDER_URLS.get(name),
                    oss_model=self.oss_model or defaults.get("oss", ""),
                    frontier_model=self.frontier_model or defaults.get("frontier", ""),
                ))

        return chain

    @property
    def provider_chain(self) -> list[ProviderConfig]:
        """Get the full fallback chain of available providers."""
        return self._get_provider_chain()

    @property
    def provider(self) -> str:
        """Primary provider name."""
        chain = self.provider_chain
        return chain[0].name if chain else "none"

    @property
    def api_key(self) -> str:
        """Primary provider API key."""
        chain = self.provider_chain
        return chain[0].api_key if chain else ""

    @property
    def base_url(self) -> str | None:
        """Primary provider base URL."""
        chain = self.provider_chain
        return chain[0].base_url if chain else None

    def get_oss_model(self) -> str:
        chain = self.provider_chain
        return chain[0].oss_model if chain else self.hf_model_id

    def get_frontier_model(self) -> str:
        chain = self.provider_chain
        return chain[0].frontier_model if chain else ""

    # Legacy compat
    @property
    def frontier_provider(self) -> str:
        return self.provider

    def describe(self) -> str:
        chain = self.provider_chain
        providers = ", ".join(p.name for p in chain)
        return (
            f"Providers: {providers}\n"
            f"Primary: {self.provider}\n"
            f"OSS Model: {self.get_oss_model()}\n"
            f"Frontier Model: {self.get_frontier_model()}\n"
            f"Fallback depth: {len(chain)}\n"
            f"Guardrails: {'ON' if self.enable_guardrails else 'OFF'}\n"
            f"Memory: {self.max_memory_turns} turns"
        )
