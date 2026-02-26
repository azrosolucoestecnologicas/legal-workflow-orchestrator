"""
Workflow Engine — Core Orchestration Logic.

The engine is responsible for:
  1. Registering workflow definitions
  2. Executing workflows step by step
  3. Handling step failures with retries
  4. Routing conditional steps (SKIPPED vs executed)
  5. Managing human gates (interactive vs auto-approve)
  6. Writing step outputs to WorkflowMemory
  7. Building the full execution trace
  8. Handling workflow-level errors gracefully

Execution loop:
  for each step in workflow.steps:
    if step.should_run(memory) is False → SKIPPED
    if step is HumanGateStep → pause, get human input, continue
    if step is AgentStep:
      for attempt in range(step.max_retries):
        result = step.agent.execute(step.step_id, memory)
        if result.succeeded → write to memory, record trace, continue
        if result.failed and attempt < max_retries - 1 → retry
      if all attempts failed and step.required → FAIL workflow
      if all attempts failed and not step.required → log warning, continue

Memory contract:
  Each step reads from memory (previous step outputs).
  Each step writes output to memory[step.memory_key].
  The engine also writes execution metadata to memory['_meta'][step_id].

Interactive mode:
  HumanGateStep: prints the prompt_fn(memory) result and asks for approval.
  User types 'yes'/'no'. 'no' cancels the workflow.
  Can be bypassed with interactive=False for batch processing.
"""
from __future__ import annotations

import logging
import time
from typing import Optional

from src.workflows.workflow_steps import (
    WorkflowDefinition, AgentStep, HumanGateStep, StepType
)
from src.utils.workflow_models import (
    WorkflowMemory, WorkflowTrace, WorkflowResult, StepTrace,
    StepResult, StepStatus, WorkflowStatus
)

logger = logging.getLogger(__name__)


class WorkflowEngine:
    """
    Executes registered workflow definitions.
    """

    def __init__(self):
        self._workflows: dict[str, WorkflowDefinition] = {}

    def register(self, workflow: WorkflowDefinition) -> None:
        """Register a workflow definition."""
        self._workflows[workflow.name] = workflow
        logger.debug(f"Registered workflow: {workflow.name} ({len(workflow.steps)} steps)")

    def run(
        self,
        workflow_name: str,
        initial_input: dict,
        interactive: bool = False,
    ) -> WorkflowResult:
        """
        Execute a workflow from initial input.

        Args:
            workflow_name: Name of registered workflow to run.
            initial_input: Initial data to populate WorkflowMemory.
            interactive: If True, pause at HumanGateSteps for approval.

        Returns:
            WorkflowResult with status, memory, and trace.
        """
        if workflow_name not in self._workflows:
            raise ValueError(f"Workflow '{workflow_name}' not registered. "
                             f"Available: {list(self._workflows.keys())}")

        workflow = self._workflows[workflow_name]
        memory = WorkflowMemory()
        memory.update(initial_input)

        trace = WorkflowTrace(workflow_name=workflow_name)

        logger.info(f"Starting workflow: {workflow_name} (id={trace.workflow_id})")

        try:
            status = self._execute_workflow(workflow, memory, trace, interactive)
        except Exception as e:
            logger.error(f"Workflow {workflow_name} crashed: {e}", exc_info=True)
            trace.complete(WorkflowStatus.FAILED, error=str(e))
            status = WorkflowStatus.FAILED

        # Build final output from memory
        final_output = self._build_final_output(memory)

        return WorkflowResult(
            workflow_id=trace.workflow_id,
            workflow_name=workflow_name,
            status=status,
            memory=memory,
            trace=trace,
            final_output=final_output,
        )

    def _execute_workflow(
        self,
        workflow: WorkflowDefinition,
        memory: WorkflowMemory,
        trace: WorkflowTrace,
        interactive: bool,
    ) -> WorkflowStatus:
        """Execute all steps in order. Returns final workflow status."""

        for step in workflow.steps:
            # ── Check condition ───────────────────────────────────────────────
            if not step.should_run(memory):
                logger.info(f"Step {step.step_id} SKIPPED (condition=False)")
                self._record_skipped(step.step_id, memory, trace)
                continue

            # ── Human gate ────────────────────────────────────────────────────
            if isinstance(step, HumanGateStep):
                approved = self._handle_human_gate(step, memory, trace, interactive)
                if not approved:
                    logger.info(f"Workflow cancelled at human gate {step.step_id}")
                    trace.complete(WorkflowStatus.CANCELLED)
                    return WorkflowStatus.CANCELLED
                continue

            # ── Agent step ────────────────────────────────────────────────────
            result = self._execute_agent_step(step, memory, trace)

            if result.failed and step.required:
                error = f"Required step '{step.step_id}' failed: {result.error}"
                logger.error(error)
                trace.complete(WorkflowStatus.FAILED, error=error)
                return WorkflowStatus.FAILED

            if result.failed and not step.required:
                logger.warning(f"Optional step '{step.step_id}' failed — continuing")

        trace.complete(WorkflowStatus.COMPLETED)
        return WorkflowStatus.COMPLETED

    def _execute_agent_step(
        self,
        step: AgentStep,
        memory: WorkflowMemory,
        trace: WorkflowTrace,
    ) -> StepResult:
        """Execute an AgentStep with retry logic."""

        input_snapshot = memory.snapshot()
        last_result = None

        for attempt in range(step.max_retries):
            if attempt > 0:
                logger.info(f"Retrying step {step.step_id} (attempt {attempt + 1}/{step.max_retries})")

            result = step.agent.execute(step.step_id, memory)
            result.retry_count = attempt
            last_result = result

            if result.succeeded:
                # Write output to memory
                memory.set(step.memory_key, result.output)
                logger.info(
                    f"Step {step.step_id} COMPLETED "
                    f"(confidence={result.confidence:.2f}, "
                    f"tokens={result.tokens_used}, "
                    f"{result.duration_ms:.0f}ms)"
                )
                break

            logger.warning(f"Step {step.step_id} failed (attempt {attempt + 1}): {result.error}")

        # Record in trace
        trace.add_step(StepTrace(
            step_id=step.step_id,
            agent_name=step.agent.name,
            status=last_result.status,
            input_snapshot=input_snapshot,
            output=last_result.output,
            confidence=last_result.confidence,
            duration_ms=last_result.duration_ms,
            llm_calls=last_result.llm_calls,
            tokens_used=last_result.tokens_used,
            error=last_result.error,
            started_at=last_result.started_at,
            completed_at=last_result.completed_at,
            retry_count=last_result.retry_count,
        ))

        return last_result

    def _handle_human_gate(
        self,
        step: HumanGateStep,
        memory: WorkflowMemory,
        trace: WorkflowTrace,
        interactive: bool,
    ) -> bool:
        """
        Handle human gate step.
        Returns True if approved (proceed), False if rejected (cancel).
        """
        trace.human_gates_encountered += 1

        if not interactive or not step.require_approval:
            # Auto-approve in non-interactive mode
            trace.human_gates_approved += 1
            logger.info(f"Human gate {step.step_id} AUTO-APPROVED (non-interactive)")
            return True

        # Interactive mode: show summary and ask
        try:
            summary = step.prompt_fn(memory)
        except Exception as e:
            summary = f"[Error building summary: {e}]"

        print("\n" + "=" * 60)
        print(f"HUMAN GATE: {step.description}")
        print("=" * 60)
        print(summary)
        print("=" * 60)

        while True:
            response = input("\nAprovar e continuar? [sim/nao]: ").strip().lower()
            if response in ("sim", "s", "yes", "y"):
                trace.human_gates_approved += 1
                logger.info(f"Human gate {step.step_id} APPROVED by user")
                return True
            elif response in ("nao", "não", "n", "no"):
                logger.info(f"Human gate {step.step_id} REJECTED by user")
                return False
            else:
                print("Por favor, responda 'sim' ou 'nao'")

    def _record_skipped(
        self,
        step_id: str,
        memory: WorkflowMemory,
        trace: WorkflowTrace,
    ) -> None:
        import time
        from datetime import datetime
        trace.add_step(StepTrace(
            step_id=step_id,
            agent_name="",
            status=StepStatus.SKIPPED,
            input_snapshot={},
            output={},
            confidence=1.0,
            duration_ms=0.0,
            llm_calls=0,
            tokens_used=0,
            error=None,
            started_at=datetime.now().isoformat(),
            completed_at=datetime.now().isoformat(),
            notes="Condition evaluated to False",
        ))

    @staticmethod
    def _build_final_output(memory: WorkflowMemory) -> dict:
        """Extract the final deliverables from memory."""
        output = {}
        key_outputs = ["classification", "pesquisa", "analise", "minuta", "revisao"]
        for key in key_outputs:
            val = memory.get(key)
            if val is not None:
                output[key] = val
        return output
