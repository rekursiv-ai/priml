# See typings/harbor/__init__.pyi for why this package exists.
# Derived from harbor 0.24.0 harbor/agents/base.py.
from abc import ABC, abstractmethod
from pathlib import Path
from typing import ClassVar
from uuid import UUID

from harbor.agents.capabilities import AgentCapabilities
from harbor.environments.base import BaseEnvironment
from harbor.models.agent.context import AgentContext

class BaseAgent(ABC):
    logs_dir: Path
    model_name: str | None
    session_id: str | None
    context_id: UUID | None
    capabilities: ClassVar[AgentCapabilities]

    def __init__(
        self,
        logs_dir: Path,
        model_name: str | None = ...,
        **kwargs: object,
    ) -> None: ...
    @staticmethod
    @abstractmethod
    def name() -> str: ...
    @abstractmethod
    def version(self) -> str | None: ...
    @abstractmethod
    async def setup(self, environment: BaseEnvironment) -> None: ...
    @abstractmethod
    async def run(
        self,
        instruction: str,
        environment: BaseEnvironment,
        context: AgentContext,
    ) -> None: ...
