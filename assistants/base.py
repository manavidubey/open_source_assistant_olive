"""
Base Assistant — Abstract interface for all assistant implementations.
"""

from __future__ import annotations
from abc import ABC, abstractmethod
import time
from typing import Optional

from memory.conversation_memory import ConversationMemory
from guardrails.safety import SafetyGuardrails, SafetyResult
from tools.tool_manager import ToolManager, ToolResult
from observability.logger import ObservabilityLogger


class BaseAssistant(ABC):
    """
    Abstract base class for AI assistants.
    
    Integrates memory, guardrails, tools, and observability
    into a unified interaction pipeline.
    """

    def __init__(
        self,
        model_name: str,
        provider: str,
        system_prompt: str,
        max_memory_turns: int = 20,
        enable_guardrails: bool = True,
        enable_tools: bool = True,
        logger: Optional[ObservabilityLogger] = None,
    ):
        self.model_name = model_name
        self.provider = provider
        self.system_prompt = system_prompt

        # Memory
        self.memory = ConversationMemory(
            max_turns=max_memory_turns,
            system_prompt=system_prompt,
        )

        # Guardrails
        self.guardrails = SafetyGuardrails(enabled=enable_guardrails)

        # Tools
        self.tools = ToolManager() if enable_tools else None

        # Observability
        self.logger = logger or ObservabilityLogger()

    @abstractmethod
    def _generate(self, messages: list[dict]) -> tuple[str, dict]:
        """
        Generate a response from the model.
        
        Args:
            messages: List of message dicts with 'role' and 'content'
            
        Returns:
            Tuple of (response_text, metadata_dict)
            metadata can include token counts, model info, etc.
        """
        ...

    def chat(self, user_message: str) -> str:
        """
        Full interaction pipeline:
        1. Safety check input
        2. Check for tool invocations
        3. Build context from memory
        4. Generate response
        5. Safety check output
        6. Update memory
        7. Log interaction
        """
        req_id = self.logger.start_request()
        start_time = time.time()
        tool_used = None

        # ── 1. Input safety check ───────────────────────────────
        safety = self.guardrails.check_input(user_message)
        if safety.is_blocked:
            latency = (time.time() - start_time) * 1000
            self.logger.log_interaction(
                request_id=req_id, model_name=self.model_name,
                provider=self.provider, user_message=user_message,
                assistant_response=safety.message, latency_ms=latency,
                safety_blocked=True, safety_rules=safety.triggered_rules,
            )
            self.memory.add_message("user", user_message)
            self.memory.add_message("assistant", safety.message)
            return safety.message

        # ── 2. Tool detection ───────────────────────────────────
        tool_context = ""
        if self.tools:
            result = self.tools.detect_and_invoke(user_message)
            if result and result.success:
                tool_used = result.tool_name
                tool_context = f"\n[Tool Result — {result.tool_name}]: {result.output}\n"

        # ── 3. Build context and generate ───────────────────────
        self.memory.add_message("user", user_message)

        messages = self.memory.get_context()
        if tool_context:
            # Insert tool result before the last user message
            messages.append({"role": "system", "content": tool_context})

        try:
            response, metadata = self._generate(messages)
        except Exception as e:
            latency = (time.time() - start_time) * 1000
            error_msg = f"I encountered an error generating a response. Please try again."
            self.logger.log_interaction(
                request_id=req_id, model_name=self.model_name,
                provider=self.provider, user_message=user_message,
                assistant_response=error_msg, latency_ms=latency,
                error=str(e),
            )
            self.memory.add_message("assistant", error_msg)
            return error_msg

        # ── 4. Output safety check ─────────────────────────────
        out_safety = self.guardrails.check_output(response)
        if out_safety.is_blocked:
            response = out_safety.message

        # ── 5. Update memory ────────────────────────────────────
        self.memory.add_message("assistant", response)

        # ── 6. Log ──────────────────────────────────────────────
        latency = (time.time() - start_time) * 1000
        self.logger.log_interaction(
            request_id=req_id, model_name=self.model_name,
            provider=self.provider, user_message=user_message,
            assistant_response=response, latency_ms=latency,
            input_tokens=metadata.get("input_tokens", 0),
            output_tokens=metadata.get("output_tokens", 0),
            tool_used=tool_used,
            safety_blocked=out_safety.is_blocked,
            safety_rules=out_safety.triggered_rules if out_safety.is_blocked else [],
        )

        return response

    def reset(self) -> None:
        """Clear conversation memory."""
        self.memory.clear()

    def get_stats(self) -> dict:
        """Get combined stats from all subsystems."""
        return {
            "model": self.model_name,
            "provider": self.provider,
            "memory": self.memory.get_stats(),
            "guardrails": self.guardrails.get_stats(),
            "metrics": self.logger.get_metrics(),
        }
