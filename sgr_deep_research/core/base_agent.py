import json
import json
import logging
import os
import traceback
import uuid
from datetime import datetime
from typing import Type

from openai import AsyncOpenAI
from openai.types.chat import ChatCompletionFunctionToolParam

from sgr_deep_research.core.agent_definition import ExecutionConfig, LLMConfig, PromptsConfig
from sgr_deep_research.core.models import AgentStatesEnum, BaseContext
from sgr_deep_research.core.policies import DefaultToolSelectionPolicy, ToolSelectionPolicy
from sgr_deep_research.core.reasoning import BaseReasoningTool
from sgr_deep_research.core.services.prompt_loader import PromptLoader
from sgr_deep_research.core.services.registry import AgentRegistry
from sgr_deep_research.core.stream import OpenAIStreamingGenerator
from sgr_deep_research.core.tools import BaseTool


class AgentRegistryMixin:
    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        if cls.__name__ not in ("BaseAgent",):
            AgentRegistry.register(cls, name=cls.name)


class BaseAgent(AgentRegistryMixin):
    """Base class for agents."""

    name: str = "base_agent"
    context_cls: Type[BaseContext] = BaseContext
    interrupting_tool_types: tuple[type[BaseTool], ...] = tuple()

    def __init__(
        self,
        task: str,
        openai_client: AsyncOpenAI,
        llm_config: LLMConfig,
        prompts_config: PromptsConfig,
        execution_config: ExecutionConfig,
        toolkit: list[Type[BaseTool]] | None = None,
        tool_selection_policy: ToolSelectionPolicy | None = None,
        **kwargs: dict,
    ):
        self.id = f"{self.name}_{uuid.uuid4()}"
        self.logger = logging.getLogger(f"sgr_deep_research.agents.{self.id}")
        self.creation_time = datetime.now()
        self.task = task
        self.toolkit = toolkit or []

        self._context = self.context_cls()
        self.conversation = []
        self.log = []
        self.max_iterations = execution_config.max_iterations
        self.max_clarifications = execution_config.max_clarifications
        self.execution_config = execution_config

        self.openai_client = openai_client
        self.llm_config = llm_config
        self.prompts_config = prompts_config
        self.tool_selection_policy = tool_selection_policy or DefaultToolSelectionPolicy()

        self.streaming_generator = OpenAIStreamingGenerator(model=self.id)

    async def provide_clarification(self, clarifications: str):
        """Receive clarification from external source (e.g. user input)"""
        self.conversation.append(
            {"role": "user", "content": PromptLoader.get_clarification_template(clarifications, self.prompts_config)}
        )
        self._context.clarifications_used += 1
        self._context.clarification_received.set()
        self._context.state = AgentStatesEnum.RESEARCHING
        self.logger.info(f"✅ Clarification received: {clarifications[:2000]}...")

    def _log_reasoning(self, result: BaseReasoningTool) -> None:
        summary = result.to_log_summary()
        next_step = result.next_step_text()
        self.logger.info(
            f"""
###############################################
🤖 LLM RESPONSE DEBUG:
   ➡️ Next Step: {next_step}
   🧠 Summary: {json.dumps(summary, ensure_ascii=False)[:1500]}...
###############################################"""
        )
        self.log.append(
            {
                "step_number": self._context.iteration,
                "timestamp": datetime.now().isoformat(),
                "step_type": "reasoning",
                "agent_reasoning": summary,
            }
        )

    def _log_tool_execution(self, tool: BaseTool, result: str):
        self.logger.info(
            f"""
###############################################
🛠️ TOOL EXECUTION DEBUG:
    🔧 Tool Name: {tool.tool_name}
    📋 Tool Model: {tool.model_dump_json(indent=2)}
    🔍 Result: '{result[:400]}...'
###############################################"""
        )
        self.log.append(
            {
                "step_number": self._context.iteration,
                "timestamp": datetime.now().isoformat(),
                "step_type": "tool_execution",
                "tool_name": tool.tool_name,
                "agent_tool_context": tool.model_dump(),
                "agent_tool_execution_result": result,
            }
        )

    def _save_agent_log(self):
        from sgr_deep_research.core.agent_config import GlobalConfig

        logs_dir = GlobalConfig().execution.logs_dir
        os.makedirs(logs_dir, exist_ok=True)
        filepath = os.path.join(logs_dir, f"{datetime.now().strftime('%Y%m%d-%H%M%S')}-{self.id}-log.json")
        agent_log = {
            "id": self.id,
            "model_config": self.llm_config.model_dump(exclude={"api_key", "proxy"}),
            "task": self.task,
            "toolkit": [tool.tool_name for tool in self.toolkit],
            "log": self.log,
        }

        json.dump(agent_log, open(filepath, "w", encoding="utf-8"), indent=2, ensure_ascii=False)

    async def _prepare_context(self) -> list[dict]:
        """Prepare conversation context with system prompt."""
        return [
            {"role": "system", "content": PromptLoader.get_system_prompt(self.toolkit, self.prompts_config)},
            *self.conversation,
        ]

    async def _prepare_tools(self) -> list[ChatCompletionFunctionToolParam]:
        """Prepare available tools for current agent state and progress."""
        raise NotImplementedError("_prepare_tools must be implemented by subclass")

    async def _reasoning_phase(self) -> BaseReasoningTool:
        """Call LLM to decide next action based on current context."""
        raise NotImplementedError("_reasoning_phase must be implemented by subclass")

    async def _select_action_phase(self, reasoning: BaseReasoningTool) -> BaseTool:
        """Select most suitable tool for the action decided in reasoning phase.

        Returns the tool suitable for the action.
        """
        raise NotImplementedError("_select_action_phase must be implemented by subclass")

    async def _action_phase(self, tool: BaseTool) -> str:
        """Call Tool for the action decided in select_action phase.

        Returns string or dumped json result of the tool execution.
        """
        raise NotImplementedError("_action_phase must be implemented by subclass")

    def is_interrupting_tool(self, tool: BaseTool) -> bool:
        return isinstance(tool, self.interrupting_tool_types)

    async def execute(self):
        self.logger.info(f"🚀 Starting for task: '{self.task}'")
        self.conversation.extend(
            [
                {
                    "role": "user",
                    "content": PromptLoader.get_initial_user_request(self.task, self.prompts_config),
                }
            ]
        )
        try:
            while self._context.state not in AgentStatesEnum.FINISH_STATES:
                self._context.iteration += 1
                self.logger.info(f"Step {self._context.iteration} started")

                reasoning = await self._reasoning_phase()
                self._context.current_step_reasoning = reasoning
                action_tool = await self._select_action_phase(reasoning)
                await self._action_phase(action_tool)

                if self.is_interrupting_tool(action_tool):
                    self.logger.info("\n⏸️  Research paused - awaiting external input")
                    self._context.state = AgentStatesEnum.WAITING_FOR_CLARIFICATION
                    self.streaming_generator.finish()
                    self._context.clarification_received.clear()
                    await self._context.clarification_received.wait()
                    continue

        except Exception as e:
            self.logger.error(f"❌ Agent execution error: {str(e)}")
            self._context.state = AgentStatesEnum.FAILED
            traceback.print_exc()
        finally:
            if self.streaming_generator is not None:
                self.streaming_generator.finish()
            self._save_agent_log()
