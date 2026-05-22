"""
AI Personal Assistant — Configuration Module

Centralizes all configuration with environment variable overrides.
"""

import os
from dataclasses import dataclass
from dotenv import load_dotenv

load_dotenv()


@dataclass
class Config:
    """Application configuration loaded from environment variables."""

    # --- Hugging Face (Open Source Model) ---
    hf_api_token: str = ""
    hf_model_id: str = "Qwen/Qwen2.5-0.5B-Instruct"

    # --- Google Gemini (Frontier Model — Primary) ---
    gemini_api_key: str = ""
    gemini_model: str = "gemini-2.0-flash"

    # --- OpenAI (Frontier Model — Alternative) ---
    openai_api_key: str = ""
    openai_model: str = "gpt-4.1-mini"

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
            hf_api_token=os.getenv("HF_API_TOKEN", ""),
            hf_model_id=os.getenv("HF_MODEL_ID", "Qwen/Qwen2.5-0.5B-Instruct"),
            gemini_api_key=os.getenv("GEMINI_API_KEY", ""),
            gemini_model=os.getenv("GEMINI_MODEL", "gemini-2.0-flash"),
            openai_api_key=os.getenv("OPENAI_API_KEY", ""),
            openai_model=os.getenv("OPENAI_MODEL", "gpt-4.1-mini"),
            max_memory_turns=int(os.getenv("MAX_MEMORY_TURNS", "20")),
            summary_threshold=int(os.getenv("SUMMARY_THRESHOLD", "15")),
            enable_guardrails=os.getenv("ENABLE_GUARDRAILS", "true").lower() == "true",
            log_level=os.getenv("LOG_LEVEL", "INFO"),
            log_file=os.getenv("LOG_FILE", "logs/assistant.log"),
        )

    @property
    def frontier_provider(self) -> str:
        """Determine which frontier provider is configured."""
        if self.gemini_api_key:
            return "gemini"
        elif self.openai_api_key:
            return "openai"
        return "none"
