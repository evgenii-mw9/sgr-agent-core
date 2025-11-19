from __future__ import annotations

from typing import Protocol, TYPE_CHECKING

from sgr_deep_research.core.base_tool import BaseTool

if TYPE_CHECKING:
    from sgr_deep_research.core.base_agent import BaseAgent


class ToolSelectionPolicy(Protocol):
    """Strategy protocol for selecting tools per agent phase."""

    def tools_for_reasoning(self, agent: "BaseAgent") -> list[type[BaseTool]]:
        ...

    def tools_for_action(self, agent: "BaseAgent") -> list[type[BaseTool]]:
        ...


class DefaultToolSelectionPolicy:
    """Default policy returning the agent toolkit unchanged."""

    def tools_for_reasoning(self, agent: "BaseAgent") -> list[type[BaseTool]]:
        return list(agent.toolkit)

    def tools_for_action(self, agent: "BaseAgent") -> list[type[BaseTool]]:
        return list(agent.toolkit)
