"""
Base Agent — Foundation for All Workflow Agents.

An agent in this system is a stateless callable that:
  1. Receives the current workflow memory as input
  2. Executes one focused task (classify, research, draft, review)
  3. Returns a StepResult with output + execution metadata
  4. Writes its output back to memory via the result

Stateless design:
  Agents do NOT hold state between calls.
  All context comes from WorkflowMemory.
  This makes agents:
    - Testable in isolation (just pass a memory snapshot)
    - Retryable (call again with the same memory state)
    - Composable (same agent can appear in multiple workflows)

LLM usage pattern:
  Each agent has a system prompt that defines its role and output schema.
  The user prompt is constructed from memory contents.
  Temperature = 0 for consistency.
  Output is always JSON so the engine can validate it.

Agent composition rules:
  - Each agent has exactly one responsibility (Single Responsibility)
  - Agents communicate only through WorkflowMemory
  - Agents must not call other agents directly
  - The WorkflowEngine is responsible for sequencing
"""
from __future__ import annotations

import json
import logging
import time
from abc import ABC, abstractmethod
from typing import Optional, Any

from src.utils.workflow_models import StepResult, StepStatus, WorkflowMemory

logger = logging.getLogger(__name__)


class LLMCallStats:
    """Tracks LLM usage for a single agent execution."""
    def __init__(self):
        self.calls = 0
        self.tokens_used = 0
        self.total_cost_usd = 0.0


class AgentLLMClient:
    """
    Lightweight LLM client for agents.

    Wraps the provider SDK with:
    - Temperature=0 enforcement for agent determinism
    - JSON mode when supported
    - Usage tracking
    - Error handling with structured fallback
    """

    def __init__(self, provider: str = "openai", model: str = "gpt-4o-mini"):
        self.provider = provider
        self.model = model
        self._client = self._init_client()

    def _init_client(self):
        import os
        if self.provider == "openai":
            try:
                from openai import OpenAI
                key = os.getenv("OPENAI_API_KEY")
                return OpenAI(api_key=key) if key else None
            except ImportError:
                return None
        elif self.provider == "anthropic":
            try:
                import anthropic
                key = os.getenv("ANTHROPIC_API_KEY")
                return anthropic.Anthropic(api_key=key) if key else None
            except ImportError:
                return None
        return None

    def complete_json(
        self,
        system: str,
        user: str,
        stats: Optional[LLMCallStats] = None,
    ) -> tuple[dict, int]:
        """
        Get a JSON response from the LLM.

        Returns (parsed_dict, tokens_used).
        Returns ({}, 0) on failure.
        """
        if self._client is None:
            logger.warning("No LLM client — returning empty dict")
            return {}, 0

        try:
            if self.provider == "openai":
                resp = self._client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {"role": "system", "content": system},
                        {"role": "user", "content": user},
                    ],
                    temperature=0.0,
                    max_tokens=2000,
                    response_format={"type": "json_object"},
                )
                content = resp.choices[0].message.content
                tokens = resp.usage.total_tokens
            elif self.provider == "anthropic":
                resp = self._client.messages.create(
                    model=self.model,
                    max_tokens=2000,
                    temperature=0.0,
                    system=system + "\n\nRespond ONLY with valid JSON.",
                    messages=[{"role": "user", "content": user}],
                )
                content = resp.content[0].text
                tokens = resp.usage.input_tokens + resp.usage.output_tokens
            else:
                return {}, 0

            if stats:
                stats.calls += 1
                stats.tokens_used += tokens

            # Parse JSON
            content = content.strip()
            if content.startswith("```"):
                content = content.split("```")[1]
                if content.startswith("json"):
                    content = content[4:]
            return json.loads(content.strip()), tokens

        except Exception as e:
            logger.error(f"LLM call failed: {e}")
            return {}, 0


class BaseAgent(ABC):
    """
    Abstract base class for all workflow agents.
    """

    def __init__(
        self,
        llm_client: Optional[AgentLLMClient] = None,
        model: str = "gpt-4o-mini",
    ):
        self._llm = llm_client or AgentLLMClient(model=model)
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")

    @property
    @abstractmethod
    def name(self) -> str:
        """Unique agent identifier."""
        ...

    @property
    @abstractmethod
    def description(self) -> str:
        """What this agent does."""
        ...

    @abstractmethod
    def _build_prompt(self, memory: WorkflowMemory) -> tuple[str, str]:
        """
        Build system and user prompts from memory.

        Returns (system_prompt, user_prompt).
        """
        ...

    @abstractmethod
    def _parse_output(self, raw: dict) -> tuple[dict, float]:
        """
        Parse and validate raw LLM output.

        Returns (validated_output, confidence).
        """
        ...

    def execute(self, step_id: str, memory: WorkflowMemory) -> StepResult:
        """
        Execute this agent and return a StepResult.

        The engine calls this method. Do not override — override
        _build_prompt and _parse_output instead.
        """
        stats = LLMCallStats()
        started_at = time.time()

        self.logger.info(f"Executing {self.name} for step {step_id}")

        # Build prompts
        try:
            system_prompt, user_prompt = self._build_prompt(memory)
        except Exception as e:
            return self._fail(step_id, f"Prompt building failed: {e}", stats, started_at)

        # Call LLM
        raw_output, tokens = self._llm.complete_json(system_prompt, user_prompt, stats)

        if not raw_output:
            return self._fail(step_id, "LLM returned empty response", stats, started_at)

        # Parse and validate
        try:
            output, confidence = self._parse_output(raw_output)
        except Exception as e:
            return self._fail(step_id, f"Output parsing failed: {e}", stats, started_at)

        duration_ms = (time.time() - started_at) * 1000

        return StepResult(
            step_id=step_id,
            status=StepStatus.COMPLETED,
            output=output,
            confidence=confidence,
            duration_ms=duration_ms,
            llm_calls=stats.calls,
            tokens_used=stats.tokens_used,
            agent_name=self.name,
        )

    def _fail(
        self,
        step_id: str,
        error: str,
        stats: LLMCallStats,
        started_at: float,
    ) -> StepResult:
        self.logger.error(f"{self.name} failed: {error}")
        return StepResult(
            step_id=step_id,
            status=StepStatus.FAILED,
            error=error,
            duration_ms=(time.time() - started_at) * 1000,
            llm_calls=stats.calls,
            tokens_used=stats.tokens_used,
            agent_name=self.name,
        )

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(name={self.name!r})"
