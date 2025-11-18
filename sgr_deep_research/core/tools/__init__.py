from sgr_deep_research.core.base_tool import (
    BaseTool,
    MCPBaseTool,
)
from sgr_deep_research.core.next_step_tool import (
    NextStepToolsBuilder,
    NextStepToolStub,
)
from sgr_deep_research.core.tools.adapt_plan_tool import AdaptPlanTool
from sgr_deep_research.core.tools.clarification_tool import ClarificationTool
from sgr_deep_research.core.tools.final_answer_tool import FinalAnswerTool
from sgr_deep_research.core.tools.generate_plan_tool import GeneratePlanTool
from sgr_deep_research.core.tools.reasoning_tool import ReasoningTool

# Tool lists for backward compatibility
system_agent_tools = [
    ClarificationTool,
    GeneratePlanTool,
    AdaptPlanTool,
    FinalAnswerTool,
    ReasoningTool,
]

__all__ = [
    # Base classes
    "BaseTool",
    "MCPBaseTool",
    "NextStepToolStub",
    "NextStepToolsBuilder",
    # Individual tools
    "ClarificationTool",
    "GeneratePlanTool",
    "AdaptPlanTool",
    "FinalAnswerTool",
    "ReasoningTool",
    # Tool Collections
    "system_agent_tools",
]
