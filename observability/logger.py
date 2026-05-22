"""
Observability & Structured Logging

Tracks requests, responses, latency, token usage, and safety events
with structured JSON logging for production monitoring.
"""

from __future__ import annotations
import os, json, time, logging
from datetime import datetime
from dataclasses import dataclass, field, asdict
from typing import Optional


@dataclass
class RequestLog:
    """Structured log entry for a single assistant interaction."""
    request_id: str
    timestamp: str
    model_name: str
    provider: str  # "oss" or "frontier"
    user_message: str
    assistant_response: str
    latency_ms: float
    input_tokens: int = 0
    output_tokens: int = 0
    safety_blocked: bool = False
    safety_rules: list[str] = field(default_factory=list)
    tool_used: Optional[str] = None
    error: Optional[str] = None

    def to_dict(self) -> dict:
        return asdict(self)


class ObservabilityLogger:
    """Structured logging for assistant interactions."""

    def __init__(self, log_file: str = "logs/assistant.log", log_level: str = "INFO"):
        self.log_file = log_file
        self._logs: list[RequestLog] = []
        self._request_counter = 0

        # Ensure log directory exists
        os.makedirs(os.path.dirname(log_file) if os.path.dirname(log_file) else "logs", exist_ok=True)

        # Setup file logger
        self._logger = logging.getLogger("assistant")
        self._logger.setLevel(getattr(logging, log_level, logging.INFO))

        if not self._logger.handlers:
            fh = logging.FileHandler(log_file)
            fh.setFormatter(logging.Formatter("%(message)s"))
            self._logger.addHandler(fh)

    def start_request(self) -> str:
        """Start timing a request, returns request_id."""
        self._request_counter += 1
        return f"req_{self._request_counter}_{int(time.time()*1000)}"

    def log_interaction(
        self,
        request_id: str,
        model_name: str,
        provider: str,
        user_message: str,
        assistant_response: str,
        latency_ms: float,
        input_tokens: int = 0,
        output_tokens: int = 0,
        safety_blocked: bool = False,
        safety_rules: list[str] | None = None,
        tool_used: str | None = None,
        error: str | None = None,
    ) -> RequestLog:
        """Log a complete interaction."""
        entry = RequestLog(
            request_id=request_id,
            timestamp=datetime.now().isoformat(),
            model_name=model_name,
            provider=provider,
            user_message=user_message[:500],
            assistant_response=assistant_response[:500],
            latency_ms=round(latency_ms, 2),
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            safety_blocked=safety_blocked,
            safety_rules=safety_rules or [],
            tool_used=tool_used,
            error=error,
        )
        self._logs.append(entry)
        self._logger.info(json.dumps(entry.to_dict()))
        return entry

    def get_metrics(self) -> dict:
        """Compute aggregate metrics from logged interactions."""
        if not self._logs:
            return {"total_requests": 0}

        latencies = [l.latency_ms for l in self._logs if not l.error]
        return {
            "total_requests": len(self._logs),
            "avg_latency_ms": round(sum(latencies) / len(latencies), 2) if latencies else 0,
            "p95_latency_ms": round(sorted(latencies)[int(len(latencies) * 0.95)] if latencies else 0, 2),
            "max_latency_ms": round(max(latencies) if latencies else 0, 2),
            "safety_blocks": sum(1 for l in self._logs if l.safety_blocked),
            "errors": sum(1 for l in self._logs if l.error),
            "tool_uses": sum(1 for l in self._logs if l.tool_used),
            "by_provider": {
                p: sum(1 for l in self._logs if l.provider == p)
                for p in set(l.provider for l in self._logs)
            },
        }

    def get_recent_logs(self, n: int = 20) -> list[dict]:
        """Get the most recent N log entries."""
        return [l.to_dict() for l in self._logs[-n:]]
