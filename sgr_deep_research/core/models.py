import asyncio
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field


class AgentStatesEnum(str, Enum):
    INITED = "inited"
    RESEARCHING = "researching"
    WAITING_FOR_CLARIFICATION = "waiting_for_clarification"
    COMPLETED = "completed"
    ERROR = "error"
    FAILED = "failed"

    FINISH_STATES = {COMPLETED, FAILED, ERROR}


class BaseContext(BaseModel):
    """Minimal context shared by all agents."""

    model_config = {"arbitrary_types_allowed": True}

    current_step_reasoning: Any = None
    execution_result: str | None = None

    state: AgentStatesEnum = Field(default=AgentStatesEnum.INITED, description="Current agent state")
    iteration: int = Field(default=0, description="Current iteration number")

    clarifications_used: int = Field(default=0, description="Number of clarifications requested")
    clarification_received: asyncio.Event = Field(
        default_factory=asyncio.Event, description="Event for synchronization of external input"
    )

    counters: dict[str, int] = Field(default_factory=dict, description="Custom counters for domain-specific use")

    def agent_state(self) -> dict:
        return self.model_dump(exclude={"clarification_received"})


class AgentStatistics(BaseModel):
    pass
