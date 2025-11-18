from __future__ import annotations

from typing import Any, Protocol


class BaseReasoningTool(Protocol):
    """Interface required by BaseAgent for reasoning outputs."""

    def to_log_summary(self) -> dict[str, Any]:
        ...

    def next_step_text(self) -> str:
        ...
