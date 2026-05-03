from __future__ import annotations

from contextlib import suppress
from pathlib import Path
from typing import Any

from mcp.types import CallToolResult
from mcp_agent.agents.agent import Agent
from mcp_agent.agents.agent_spec import AgentSpec
from mcp_agent.app import MCPApp
from mcp_agent.config import Settings
from mcp_agent.workflows.factory import agent_from_spec
from mcp_agent.workflows.llm.augmented_llm import AugmentedLLM

from derail.classifiers import Classifier
from derail.metrics import TraceSpan
from derail.runtime import classifier_context, instrumentation
from derail.runtime.llm_factory import llm_factory


class MCPRunner:
    """Owns the mcp-agent lifecycle for one scenario run."""

    def __init__(
        self,
        *,
        settings: Settings,
        agent_spec: AgentSpec,
        provider: str,
        model: str,
        otel_path: Path,
        classifier: Classifier | None = None,
        scenario_name: str | None = None,
    ) -> None:
        self.settings = settings
        self.agent_spec = agent_spec
        self.provider = provider
        self.model = model
        self.otel_path = otel_path
        self.classifier = classifier
        self.scenario_name = scenario_name

        self.app: MCPApp | None = None
        self.agent: Agent | None = None
        self.llm: AugmentedLLM | None = None
        self._instrumentation_installed = False
        self._classifier_token: object | None = None
        self._scenario_name_token: object | None = None

    async def __aenter__(self) -> MCPRunner:

        instrumentation.install()
        self._instrumentation_installed = True
        if self.classifier is not None:
            self._classifier_token = classifier_context.set_active_classifier(self.classifier)
        if self.scenario_name is not None:
            self._scenario_name_token = classifier_context.set_active_scenario_name(self.scenario_name)
        self.app = MCPApp(settings=self.settings)
        await self.app.initialize()

        self.agent = agent_from_spec(self.agent_spec, context=self.app.context)
        await self.agent.initialize()

        factory = llm_factory(provider=self.provider, model=self.model, context=self.app.context)
        self.llm = factory(self.agent)
        await self.agent.attach_llm(llm=self.llm)
        return self

    async def __aexit__(self, exc_type, exc, tb) -> None:

        if self.agent is not None:
            with suppress(Exception):
                await self.agent.shutdown()
        if self.app is not None:
            with suppress(Exception):
                await self.app.cleanup()

        if self._scenario_name_token is not None:
            classifier_context.reset_active_scenario_name(self._scenario_name_token)
            self._scenario_name_token = None
        if self._classifier_token is not None:
            classifier_context.reset_active_classifier(self._classifier_token)
            self._classifier_token = None
        if self._instrumentation_installed:
            self._instrumentation_installed = False
            instrumentation.uninstall()

    # ---- Public operations --------------------------------------------

    async def call_tool(self, server: str, tool: str, args: dict[str, Any]) -> CallToolResult:
        if self.agent is None:
            raise RuntimeError("MCPRunner is not initialized; use as async context.")

        agent_server_names = set(self.agent_spec.server_names or [])
        if server not in agent_server_names:
            raise RuntimeError(
                f"Cannot call tool on server {server!r}: it is not in the "
                f"agent's server list ({sorted(agent_server_names)!r})."
            )
        return await self.agent.call_tool(name=tool, arguments=args, server_name=server)

    async def run_prompt(self, prompt: str) -> str:
        """Run one agent turn against ``prompt`` and return its final text."""
        if self.llm is None:
            raise RuntimeError("MCPRunner is not initialized; use as async context.")
        response = await self.llm.generate_str(prompt)
        await self._flush_traces()
        return response

    async def _flush_traces(self) -> None:
        """Force the OTEL file exporter to drain before we parse the file."""
        if self.app is None:
            return
        tracing_config = getattr(self.app._context, "tracing_config", None)
        flush = getattr(tracing_config, "flush", None) if tracing_config else None
        if flush is None:
            return
        with suppress(Exception):
            await flush()

    def read_spans(self) -> list[TraceSpan]:
        """Parse the JSONL trace file into ``TraceSpan`` dataclasses."""
        if not self.otel_path.exists():
            return []
        spans: list[TraceSpan] = []
        with self.otel_path.open() as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    spans.append(TraceSpan.from_json(line))
                except Exception:  # pragma: no cover - corrupt line
                    continue
        return spans
