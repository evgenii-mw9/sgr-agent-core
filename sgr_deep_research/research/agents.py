from __future__ import annotations

from typing import Type

from openai import AsyncOpenAI

from sgr_deep_research.core.agent_definition import ExecutionConfig, LLMConfig, PromptsConfig
from sgr_deep_research.core.agents import (
    SGRAgent,
    SGRAutoToolCallingAgent,
    SGRSOToolCallingAgent,
    SGRToolCallingAgent,
    ToolCallingAgent,
)
from sgr_deep_research.core.tools import (
    AdaptPlanTool,
    ClarificationTool,
    FinalAnswerTool,
    GeneratePlanTool,
)
from sgr_deep_research.research.context import ResearchContext
from sgr_deep_research.research.policies import ResearchToolSelectionPolicy
from sgr_deep_research.research.tools import CreateReportTool, ExtractPageContentTool, WebSearchTool

RESEARCH_TOOLKIT = [
    ClarificationTool,
    GeneratePlanTool,
    AdaptPlanTool,
    FinalAnswerTool,
    WebSearchTool,
    ExtractPageContentTool,
    CreateReportTool,
]


class BaseResearchAgentMixin:
    context_cls: Type[ResearchContext] = ResearchContext
    interrupting_tool_types = (ClarificationTool,)

    def __init__(
        self,
        task: str,
        openai_client: AsyncOpenAI,
        llm_config: LLMConfig,
        prompts_config: PromptsConfig,
        execution_config: ExecutionConfig,
        toolkit=None,
        **kwargs,
    ):
        toolkit = toolkit or RESEARCH_TOOLKIT
        super().__init__(
            task=task,
            openai_client=openai_client,
            llm_config=llm_config,
            prompts_config=prompts_config,
            execution_config=execution_config,
            toolkit=toolkit,
            tool_selection_policy=ResearchToolSelectionPolicy(),
            **kwargs,
        )


class ResearchSGRAgent(BaseResearchAgentMixin, SGRAgent):
    name = "research_sgr_agent"


class ResearchToolCallingAgent(BaseResearchAgentMixin, ToolCallingAgent):
    name = "research_tool_calling_agent"


class ResearchSGRToolCallingAgent(BaseResearchAgentMixin, SGRToolCallingAgent):
    name = "research_sgr_tool_calling_agent"


class ResearchSGRAutoToolCallingAgent(BaseResearchAgentMixin, SGRAutoToolCallingAgent):
    name = "research_sgr_auto_tool_calling_agent"


class ResearchSGRSOToolCallingAgent(BaseResearchAgentMixin, SGRSOToolCallingAgent):
    name = "research_sgr_so_tool_calling_agent"
