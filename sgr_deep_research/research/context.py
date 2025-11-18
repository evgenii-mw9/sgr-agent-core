from typing import Any

from pydantic import Field

from sgr_deep_research.core.models import AgentStatesEnum, BaseContext
from sgr_deep_research.research.models import SearchResult, SourceData


class ResearchContext(BaseContext):
    """Context tailored for deep research flows."""

    searches: list[SearchResult] = Field(default_factory=list, description="List of performed searches")
    sources: dict[str, SourceData] = Field(default_factory=dict, description="Dictionary of found sources")

    searches_used: int = Field(default=0, description="Number of searches performed")

    def agent_state(self) -> dict[str, Any]:
        return self.model_dump(exclude={"searches", "sources", "clarification_received"})
