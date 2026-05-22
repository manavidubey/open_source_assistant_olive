"""
Safety & Guardrails Layer

Provides input/output safety filtering to prevent harmful content
generation. Uses a multi-layer approach:
1. Pattern-based jailbreak detection
2. Keyword-based harmful content detection
3. Output safety validation
"""

from __future__ import annotations
import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional


class SafetyLevel(Enum):
    """Safety check result levels."""
    SAFE = "safe"
    WARNING = "warning"
    BLOCKED = "blocked"


@dataclass
class SafetyResult:
    """Result of a safety check."""
    level: SafetyLevel
    message: str = ""
    triggered_rules: list[str] = field(default_factory=list)
    original_content: str = ""

    @property
    def is_safe(self) -> bool:
        return self.level == SafetyLevel.SAFE

    @property
    def is_blocked(self) -> bool:
        return self.level == SafetyLevel.BLOCKED


# ── Jailbreak patterns ──────────────────────────────────────────────
JAILBREAK_PATTERNS = [
    r"ignore\s+(all\s+)?previous\s+instructions",
    r"ignore\s+(all\s+)?prior\s+instructions",
    r"disregard\s+(all\s+)?(previous|prior|above)\s+instructions",
    r"forget\s+(all\s+)?(previous|prior|your)\s+(instructions|rules|guidelines)",
    r"you\s+are\s+(now\s+)?DAN",
    r"do\s+anything\s+now",
    r"act\s+as\s+(an?\s+)?(unrestricted|unfiltered|uncensored)",
    r"pretend\s+(you\s+)?(are|have)\s+no\s+(restrictions|rules|limits|guidelines|filters)",
    r"bypass\s+(your\s+)?(safety|content|ethical)\s+(filters?|guidelines?|restrictions?)",
    r"jailbreak",
    r"developer\s+mode\s+(enabled|activated|on)",
    r"override\s+(your\s+)?(safety|content)\s+(protocols?|settings?|filters?)",
    r"system\s+prompt\s+(override|bypass|ignore)",
    r"you\s+must\s+(always\s+)?comply",
    r"opposite\s+mode",
    r"evil\s+(mode|version|twin)",
]

# ── Harmful content categories ──────────────────────────────────────
HARMFUL_PATTERNS = {
    "violence": [
        r"how\s+to\s+(make|build|create)\s+(a\s+)?(bomb|explosive|weapon)",
        r"how\s+to\s+(kill|murder|assassinate|harm)\s+(someone|a\s+person|people)",
        r"instructions\s+for\s+(making|creating|building)\s+(a\s+)?weapon",
    ],
    "illegal_activity": [
        r"how\s+to\s+(hack|break\s+into)\s+(someone|a)\s+(account|computer|system)",
        r"how\s+to\s+(steal|forge|counterfeit)",
        r"how\s+to\s+(make|cook|produce)\s+(meth|drugs|illegal\s+substances)",
    ],
    "self_harm": [
        r"how\s+to\s+(commit\s+)?suicide",
        r"ways\s+to\s+(hurt|harm|injure)\s+(myself|yourself|oneself)",
        r"methods\s+(of|for)\s+self[- ]harm",
    ],
    "csam": [
        r"child\s+(sexual|pornograph)",
        r"minor\s+(sexual|intimate|explicit)",
    ],
}

# ── Output safety patterns ──────────────────────────────────────────
OUTPUT_UNSAFE_PATTERNS = [
    r"here\s+(is|are)\s+(the\s+)?instructions?\s+(for|to|on)\s+(making|creating|building)\s+(a\s+)?(bomb|weapon|explosive)",
    r"step\s+\d+.*detonate",
    r"here\'?s?\s+how\s+(you\s+can|to)\s+(hack|steal|forge|kill|harm)",
]


class SafetyGuardrails:
    """
    Multi-layer safety system for input and output filtering.

    Layer 1: Jailbreak detection (input)
    Layer 2: Harmful content detection (input)
    Layer 3: Output safety validation (output)
    """

    def __init__(self, enabled: bool = True):
        self.enabled = enabled
        self._jailbreak_patterns = [
            re.compile(p, re.IGNORECASE) for p in JAILBREAK_PATTERNS
        ]
        self._harmful_patterns = {
            cat: [re.compile(p, re.IGNORECASE) for p in patterns]
            for cat, patterns in HARMFUL_PATTERNS.items()
        }
        self._output_patterns = [
            re.compile(p, re.IGNORECASE) for p in OUTPUT_UNSAFE_PATTERNS
        ]

    def check_input(self, text: str) -> SafetyResult:
        """
        Validate user input for safety.
        Returns SafetyResult with level and details.
        """
        if not self.enabled:
            return SafetyResult(level=SafetyLevel.SAFE, original_content=text)

        triggered = []

        # Layer 1: Jailbreak detection
        for pattern in self._jailbreak_patterns:
            if pattern.search(text):
                triggered.append(f"jailbreak:{pattern.pattern[:50]}")

        if triggered:
            return SafetyResult(
                level=SafetyLevel.BLOCKED,
                message=(
                    "I detected a prompt injection / jailbreak attempt. "
                    "I'm designed to be helpful within safe boundaries. "
                    "Please rephrase your request."
                ),
                triggered_rules=triggered,
                original_content=text,
            )

        # Layer 2: Harmful content detection
        for category, patterns in self._harmful_patterns.items():
            for pattern in patterns:
                if pattern.search(text):
                    triggered.append(f"harmful:{category}")

        if triggered:
            return SafetyResult(
                level=SafetyLevel.BLOCKED,
                message=(
                    "I can't help with that request as it involves potentially "
                    "harmful content. I'm here to assist with safe and constructive "
                    "tasks. How else can I help you?"
                ),
                triggered_rules=triggered,
                original_content=text,
            )

        return SafetyResult(level=SafetyLevel.SAFE, original_content=text)

    def check_output(self, text: str) -> SafetyResult:
        """
        Validate model output for safety.
        Returns SafetyResult with level and details.
        """
        if not self.enabled:
            return SafetyResult(level=SafetyLevel.SAFE, original_content=text)

        triggered = []

        for pattern in self._output_patterns:
            if pattern.search(text):
                triggered.append(f"output_unsafe:{pattern.pattern[:50]}")

        if triggered:
            return SafetyResult(
                level=SafetyLevel.BLOCKED,
                message=(
                    "I apologize, but I can't provide that information as it "
                    "could be harmful. Let me help you with something else."
                ),
                triggered_rules=triggered,
                original_content=text,
            )

        return SafetyResult(level=SafetyLevel.SAFE, original_content=text)

    def get_stats(self) -> dict:
        """Get guardrails configuration stats."""
        return {
            "enabled": self.enabled,
            "jailbreak_patterns": len(self._jailbreak_patterns),
            "harmful_categories": list(self._harmful_patterns.keys()),
            "output_patterns": len(self._output_patterns),
        }
