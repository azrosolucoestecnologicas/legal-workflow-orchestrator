"""
Workflow Step Definitions — Building Blocks of Workflow DAGs.

A WorkflowStep is the unit of execution in the engine. It wraps an agent
with execution policy: retries, conditions, timeout, and memory writes.

Step types:
  AgentStep      → runs a BaseAgent, writes output to memory key
  ConditionalStep → evaluates a condition from memory, routes to branches
  HumanGateStep  → pauses execution and waits for human input
  ParallelStep   → runs multiple AgentSteps concurrently

Execution policy (on each AgentStep):
  max_retries: how many times to re-run on FAILED status
  required: if True, workflow fails if this step fails after retries
  condition_fn: callable(memory) → bool; step is SKIPPED if returns False
  memory_key: where to write the step output in WorkflowMemory

Design: steps are pure data structures — they don't know about other steps.
The WorkflowEngine resolves execution order and handles routing.

Step conditions are Python callables that take WorkflowMemory and return bool.
This keeps conditions testable and composable without a DSL.

Example:
  # Only research if classification confidence is high enough
  condition=lambda m: m.get_nested('classification', 'confidence') or 0 > 0.7
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Callable, Any
from enum import Enum

from src.agents.base_agent import BaseAgent
from src.utils.workflow_models import WorkflowMemory


class StepType(str, Enum):
    AGENT = "agent"
    CONDITIONAL = "conditional"
    HUMAN_GATE = "human_gate"
    PARALLEL = "parallel"


@dataclass
class AgentStep:
    """
    A workflow step that executes a BaseAgent.
    """
    step_id: str
    agent: BaseAgent
    memory_key: str                          # Where to write output in WorkflowMemory
    description: str = ""
    max_retries: int = 1                     # Total attempts (1 = no retry)
    required: bool = True                    # Fail workflow if this step fails?
    condition: Optional[Callable[[WorkflowMemory], bool]] = None
    timeout_seconds: Optional[float] = None
    step_type: StepType = StepType.AGENT

    def should_run(self, memory: WorkflowMemory) -> bool:
        """Evaluate whether this step should run."""
        if self.condition is None:
            return True
        try:
            return bool(self.condition(memory))
        except Exception:
            return True  # Default to running if condition errors

    def __repr__(self) -> str:
        return f"AgentStep(id={self.step_id!r}, agent={self.agent.name!r})"


@dataclass
class HumanGateStep:
    """
    A workflow step that pauses execution for human review/approval.

    In interactive mode: prints a summary and waits for input.
    In non-interactive mode: auto-approves (for testing/batch).
    """
    step_id: str
    prompt_fn: Callable[[WorkflowMemory], str]    # Builds the human-readable summary
    require_approval: bool = True                  # If False, auto-approve
    description: str = "Human review gate"
    condition: Optional[Callable[[WorkflowMemory], bool]] = None
    step_type: StepType = StepType.HUMAN_GATE

    def should_run(self, memory: WorkflowMemory) -> bool:
        if self.condition is None:
            return True
        try:
            return bool(self.condition(memory))
        except Exception:
            return True

    def __repr__(self) -> str:
        return f"HumanGateStep(id={self.step_id!r})"


@dataclass
class WorkflowDefinition:
    """
    A complete workflow definition: name + ordered list of steps.

    Steps are executed in order. Conditional branching is handled
    via the condition parameter on each step.

    The engine can also handle parallel execution if steps are
    wrapped in ParallelStep (not implemented in this version,
    but the step_type field is ready for extension).
    """
    name: str
    steps: list[AgentStep | HumanGateStep]
    description: str = ""
    version: str = "1.0"
    metadata: dict = field(default_factory=dict)

    def get_step(self, step_id: str) -> Optional[AgentStep | HumanGateStep]:
        """Find a step by its ID."""
        for step in self.steps:
            if step.step_id == step_id:
                return step
        return None

    def step_ids(self) -> list[str]:
        return [s.step_id for s in self.steps]

    def __repr__(self) -> str:
        return f"WorkflowDefinition(name={self.name!r}, steps={self.step_ids()})"
