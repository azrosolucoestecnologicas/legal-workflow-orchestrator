"""
Workflow Models — Core Data Structures for Orchestration.

Design principles:
  1. Every step output is typed and validated
  2. Memory is a shared mutable store, not passed as arguments
     (avoids the "prompt gets longer with each step" anti-pattern)
  3. Trace captures every decision for auditability
  4. StepResult carries both the output and execution metadata
     (duration, tokens, confidence) so the engine can make routing decisions

Step execution states:
  PENDING   → not yet executed
  RUNNING   → currently executing
  COMPLETED → finished successfully
  FAILED    → raised exception or returned invalid output
  SKIPPED   → condition evaluated to False
  WAITING   → waiting for human approval (human gate)
  CANCELLED → parent step failed, this step was not attempted

Workflow terminal states:
  COMPLETED → all required steps succeeded
  FAILED    → a required step failed and no retry succeeded
  CANCELLED → explicitly cancelled
  WAITING   → awaiting human input
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Optional
from datetime import datetime
import json
import uuid


class StepStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"
    WAITING = "waiting"
    CANCELLED = "cancelled"


class WorkflowStatus(str, Enum):
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    WAITING_HUMAN = "waiting_human"


@dataclass
class StepResult:
    """
    Output of a single workflow step execution.

    Carries both the functional output and execution metadata.
    The engine uses metadata for routing decisions (retry on failure,
    skip next step if confidence is too low, etc.)
    """
    step_id: str
    status: StepStatus
    output: dict = field(default_factory=dict)
    error: Optional[str] = None
    confidence: float = 1.0
    duration_ms: float = 0.0
    llm_calls: int = 0
    tokens_used: int = 0
    agent_name: str = ""
    started_at: str = field(default_factory=lambda: datetime.now().isoformat())
    completed_at: Optional[str] = None
    retry_count: int = 0
    metadata: dict = field(default_factory=dict)

    @property
    def succeeded(self) -> bool:
        return self.status == StepStatus.COMPLETED

    @property
    def failed(self) -> bool:
        return self.status == StepStatus.FAILED

    def to_dict(self) -> dict:
        return {
            "step_id": self.step_id,
            "status": self.status.value,
            "output": self.output,
            "error": self.error,
            "confidence": round(self.confidence, 4),
            "duration_ms": round(self.duration_ms, 1),
            "llm_calls": self.llm_calls,
            "tokens_used": self.tokens_used,
            "agent_name": self.agent_name,
            "started_at": self.started_at,
            "completed_at": self.completed_at,
            "retry_count": self.retry_count,
        }


@dataclass
class WorkflowMemory:
    """
    Shared state store across all workflow steps.

    Provides a dict-like interface with:
      - get/set with dot notation for nested access
      - snapshot for trace capture
      - typed access helpers

    The memory persists for the lifetime of a workflow execution.
    Each step can read all previous outputs and write its own.
    """
    _store: dict = field(default_factory=dict)

    def set(self, key: str, value: Any) -> None:
        self._store[key] = value

    def get(self, key: str, default: Any = None) -> Any:
        return self._store.get(key, default)

    def update(self, data: dict) -> None:
        self._store.update(data)

    def snapshot(self) -> dict:
        """Return a deep copy of current state for trace capture."""
        import copy
        return copy.deepcopy(self._store)

    def get_nested(self, *keys: str, default: Any = None) -> Any:
        """Access nested dict values: memory.get_nested('step1', 'output', 'tipo')"""
        current = self._store
        for key in keys:
            if not isinstance(current, dict) or key not in current:
                return default
            current = current[key]
        return current

    def to_dict(self) -> dict:
        return self._store.copy()

    def __contains__(self, key: str) -> bool:
        return key in self._store

    def __repr__(self) -> str:
        keys = list(self._store.keys())
        return f"WorkflowMemory({keys})"


@dataclass
class StepTrace:
    """Trace record for a single step execution."""
    step_id: str
    agent_name: str
    status: StepStatus
    input_snapshot: dict
    output: dict
    confidence: float
    duration_ms: float
    llm_calls: int
    tokens_used: int
    error: Optional[str]
    started_at: str
    completed_at: Optional[str]
    retry_count: int = 0
    notes: str = ""

    def to_dict(self) -> dict:
        return {
            "step_id": self.step_id,
            "agent_name": self.agent_name,
            "status": self.status.value,
            "input_snapshot": self.input_snapshot,
            "output": self.output,
            "confidence": round(self.confidence, 4),
            "duration_ms": round(self.duration_ms, 1),
            "llm_calls": self.llm_calls,
            "tokens_used": self.tokens_used,
            "error": self.error,
            "started_at": self.started_at,
            "completed_at": self.completed_at,
            "retry_count": self.retry_count,
            "notes": self.notes,
        }


@dataclass
class WorkflowTrace:
    """
    Complete execution trace for a workflow run.

    Provides full auditability: what each agent decided,
    why, with what confidence, and how long it took.
    Essential for debugging, compliance, and model improvement.
    """
    workflow_id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    workflow_name: str = ""
    status: WorkflowStatus = WorkflowStatus.RUNNING
    started_at: str = field(default_factory=lambda: datetime.now().isoformat())
    completed_at: Optional[str] = None
    steps: list[StepTrace] = field(default_factory=list)
    human_gates_encountered: int = 0
    human_gates_approved: int = 0
    total_llm_calls: int = 0
    total_tokens_used: int = 0
    total_duration_ms: float = 0.0
    error: Optional[str] = None

    def add_step(self, step_trace: StepTrace) -> None:
        self.steps.append(step_trace)
        self.total_llm_calls += step_trace.llm_calls
        self.total_tokens_used += step_trace.tokens_used
        self.total_duration_ms += step_trace.duration_ms

    def complete(self, status: WorkflowStatus, error: Optional[str] = None) -> None:
        self.status = status
        self.completed_at = datetime.now().isoformat()
        self.error = error

    def to_dict(self) -> dict:
        return {
            "workflow_id": self.workflow_id,
            "workflow_name": self.workflow_name,
            "status": self.status.value,
            "started_at": self.started_at,
            "completed_at": self.completed_at,
            "steps": [s.to_dict() for s in self.steps],
            "human_gates_encountered": self.human_gates_encountered,
            "human_gates_approved": self.human_gates_approved,
            "total_llm_calls": self.total_llm_calls,
            "total_tokens_used": self.total_tokens_used,
            "total_duration_ms": round(self.total_duration_ms, 1),
            "error": self.error,
        }

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), ensure_ascii=False, indent=indent)


@dataclass
class WorkflowResult:
    """
    Final result of a workflow execution.

    Combines the workflow status, memory state, and trace.
    """
    workflow_id: str
    workflow_name: str
    status: WorkflowStatus
    memory: WorkflowMemory
    trace: WorkflowTrace
    final_output: dict = field(default_factory=dict)
    error: Optional[str] = None

    @property
    def succeeded(self) -> bool:
        return self.status == WorkflowStatus.COMPLETED

    @property
    def failed(self) -> bool:
        return self.status == WorkflowStatus.FAILED
