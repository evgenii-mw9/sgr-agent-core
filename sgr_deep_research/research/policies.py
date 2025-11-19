from __future__ import annotations

from sgr_deep_research.core.policies import DefaultToolSelectionPolicy
from sgr_deep_research.core.tools import ClarificationTool, FinalAnswerTool
from sgr_deep_research.research.tools import (
    CreateReportTool,
    ExtractPageContentTool,
    WebSearchTool,
)


class ResearchToolSelectionPolicy(DefaultToolSelectionPolicy):
    """Tool selection policy that applies research-specific budgets and phases."""

    def tools_for_reasoning(self, agent):
        tools = set(super().tools_for_reasoning(agent))
        context = agent._context
        execution = agent.execution_config

        if getattr(context, "searches_used", 0) >= execution.max_searches:
            tools.discard(WebSearchTool)
        if context.clarifications_used >= execution.max_clarifications:
            tools.discard(ClarificationTool)
        if context.iteration >= execution.max_iterations:
            tools = {CreateReportTool, FinalAnswerTool}

        return list(tools)

    def tools_for_action(self, agent):
        tools = set(super().tools_for_action(agent))
        context = agent._context
        execution = agent.execution_config

        if getattr(context, "searches_used", 0) >= execution.max_searches:
            tools.discard(WebSearchTool)
        if context.clarifications_used >= execution.max_clarifications:
            tools.discard(ClarificationTool)
        if context.iteration >= execution.max_iterations:
            tools = {CreateReportTool, FinalAnswerTool}

        return list(tools)
