"""
Conversation Memory Manager

Implements a sliding-window memory with automatic summarization
of older context, providing both short-term recall and compressed
long-term awareness.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Optional
import time
import json


@dataclass
class Message:
    """A single conversation message."""
    role: str           # "user", "assistant", or "system"
    content: str
    timestamp: float = field(default_factory=time.time)
    metadata: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "role": self.role,
            "content": self.content,
            "timestamp": self.timestamp,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "Message":
        return cls(
            role=data["role"],
            content=data["content"],
            timestamp=data.get("timestamp", time.time()),
            metadata=data.get("metadata", {}),
        )


class ConversationMemory:
    """
    Manages conversation history with a sliding window and
    optional summarization of older turns.

    Architecture:
    - Active window: Last `max_turns` messages (full fidelity)
    - Summary buffer: Compressed summary of older messages
    - Combined context: Summary + active window sent to model
    """

    def __init__(
        self,
        max_turns: int = 20,
        summary_threshold: int = 15,
        system_prompt: Optional[str] = None,
    ):
        self.max_turns = max_turns
        self.summary_threshold = summary_threshold
        self.system_prompt = system_prompt

        self._messages: list[Message] = []
        self._summary: Optional[str] = None
        self._total_messages: int = 0

    def add_message(self, role: str, content: str, metadata: dict | None = None) -> None:
        """Add a message to the conversation history."""
        msg = Message(
            role=role,
            content=content,
            metadata=metadata or {},
        )
        self._messages.append(msg)
        self._total_messages += 1

        # Trigger summarization if we exceed threshold
        if len(self._messages) > self.max_turns + self.summary_threshold:
            self._compress_history()

    def get_context(self) -> list[dict]:
        """
        Build the full context to send to the model.
        Returns list of message dicts in OpenAI-compatible format.
        """
        context = []

        # System prompt
        if self.system_prompt:
            system_content = self.system_prompt
            if self._summary:
                system_content += (
                    f"\n\n[CONVERSATION SUMMARY — Earlier context]\n{self._summary}"
                )
            context.append({"role": "system", "content": system_content})

        # Active message window (last max_turns messages)
        active = self._messages[-self.max_turns:] if len(self._messages) > self.max_turns else self._messages
        for msg in active:
            context.append({"role": msg.role, "content": msg.content})

        return context

    def get_history(self) -> list[dict]:
        """Get full message history (for display purposes)."""
        return [msg.to_dict() for msg in self._messages]

    def get_display_messages(self) -> list[dict]:
        """Get messages for UI display (user + assistant only)."""
        return [
            {"role": msg.role, "content": msg.content}
            for msg in self._messages
            if msg.role in ("user", "assistant")
        ]

    def _compress_history(self) -> None:
        """
        Compress older messages into a summary.
        Keeps the last `max_turns` messages and summarizes the rest.
        """
        if len(self._messages) <= self.max_turns:
            return

        # Messages to compress
        to_compress = self._messages[:-self.max_turns]
        self._messages = self._messages[-self.max_turns:]

        # Build summary from compressed messages
        summary_parts = []
        if self._summary:
            summary_parts.append(self._summary)

        for msg in to_compress:
            prefix = "User" if msg.role == "user" else "Assistant"
            # Truncate long messages in summary
            content = msg.content[:200] + "..." if len(msg.content) > 200 else msg.content
            summary_parts.append(f"{prefix}: {content}")

        self._summary = "\n".join(summary_parts)

    def clear(self) -> None:
        """Reset conversation memory."""
        self._messages.clear()
        self._summary = None
        self._total_messages = 0

    @property
    def turn_count(self) -> int:
        """Number of user-assistant turn pairs."""
        return sum(1 for m in self._messages if m.role == "user")

    @property
    def total_messages(self) -> int:
        """Total messages processed (including compressed)."""
        return self._total_messages

    @property
    def has_summary(self) -> bool:
        """Whether older context has been summarized."""
        return self._summary is not None

    def get_stats(self) -> dict:
        """Get memory statistics for observability."""
        return {
            "active_messages": len(self._messages),
            "total_processed": self._total_messages,
            "turn_count": self.turn_count,
            "has_summary": self.has_summary,
            "max_turns": self.max_turns,
        }

    def export(self) -> str:
        """Export conversation as JSON string."""
        return json.dumps({
            "messages": [m.to_dict() for m in self._messages],
            "summary": self._summary,
            "total_messages": self._total_messages,
        }, indent=2)
