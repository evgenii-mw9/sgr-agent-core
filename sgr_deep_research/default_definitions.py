import sgr_deep_research.core.tools as tools
import sgr_deep_research.research.tools as research_tools
from sgr_deep_research.core.agent_definition import AgentDefinition
from sgr_deep_research.research.agents import (
    ResearchSGRAgent,
    ResearchSGRAutoToolCallingAgent,
    ResearchSGRSOToolCallingAgent,
    ResearchSGRToolCallingAgent,
    ResearchToolCallingAgent,
)

DEFAULT_TOOLKIT = [
    tools.ClarificationTool,
    tools.GeneratePlanTool,
    tools.AdaptPlanTool,
    tools.FinalAnswerTool,
    research_tools.WebSearchTool,
    research_tools.ExtractPageContentTool,
    research_tools.CreateReportTool,
]


def get_default_agents_definitions() -> dict[str, AgentDefinition]:
    """Get default agent definitions.

    This function creates agent definitions lazily to avoid issues with
    configuration initialization order.

    Returns:
        Dictionary of default agent definitions keyed by agent name
    """
    agents = [
        AgentDefinition(
            name="sgr_agent",
            base_class=ResearchSGRAgent,
            tools=DEFAULT_TOOLKIT,
        ),
        AgentDefinition(
            name="tool_calling_agent",
            base_class=ResearchToolCallingAgent,
            tools=DEFAULT_TOOLKIT,
        ),
        AgentDefinition(
            name="sgr_tool_calling_agent",
            base_class=ResearchSGRToolCallingAgent,
            tools=DEFAULT_TOOLKIT,
        ),
        AgentDefinition(
            name="sgr_auto_tool_calling_agent",
            base_class=ResearchSGRAutoToolCallingAgent,
            tools=DEFAULT_TOOLKIT,
        ),
        AgentDefinition(
            name="sgr_so_tool_calling_agent",
            base_class=ResearchSGRSOToolCallingAgent,
            tools=DEFAULT_TOOLKIT,
        ),
    ]
    return {agent.name: agent for agent in agents}
