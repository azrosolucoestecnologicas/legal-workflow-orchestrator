"""
Unit Tests — Legal Workflow Orchestrator.

Coverage:
  1. WorkflowMemory: get/set/nested/snapshot
  2. WorkflowTrace: step recording, totals
  3. BaseAgent: StepResult building, failure handling
  4. ClassifierAgent: output parsing, validation
  5. ReviewerAgent: issue normalization, approval logic
  6. WorkflowDefinition: step ordering, condition evaluation
  7. WorkflowEngine: full pipeline in mock mode (no LLM)
"""
import sys
import json
import time
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.utils.workflow_models import (
    WorkflowMemory, WorkflowTrace, StepTrace, StepResult,
    StepStatus, WorkflowStatus, WorkflowResult
)
from src.agents.base_agent import BaseAgent, AgentLLMClient
from src.agents.classifier_agent import ClassifierAgent
from src.agents.analyst_agent import AnalystAgent
from src.agents.reviewer_agent import ReviewerAgent
from src.workflows.workflow_steps import AgentStep, HumanGateStep, WorkflowDefinition
from src.workflows.workflow_engine import WorkflowEngine
from src.workflows.definitions import (
    triagem_rapida_workflow, recurso_ordinario_workflow, peticao_inicial_workflow
)


# ── WorkflowMemory Tests ──────────────────────────────────────────────────────

class TestWorkflowMemory:
    def test_set_and_get(self):
        m = WorkflowMemory()
        m.set("key", {"value": 42})
        assert m.get("key") == {"value": 42}

    def test_get_default(self):
        m = WorkflowMemory()
        assert m.get("missing", "default") == "default"
        assert m.get("missing") is None

    def test_update(self):
        m = WorkflowMemory()
        m.update({"a": 1, "b": 2})
        assert m.get("a") == 1
        assert m.get("b") == 2

    def test_get_nested(self):
        m = WorkflowMemory()
        m.set("classification", {"area": "trabalhista", "confidence": 0.92})
        assert m.get_nested("classification", "area") == "trabalhista"
        assert m.get_nested("classification", "confidence") == 0.92
        assert m.get_nested("classification", "missing", default="x") == "x"
        assert m.get_nested("missing", "key") is None

    def test_snapshot_is_deep_copy(self):
        m = WorkflowMemory()
        m.set("data", {"list": [1, 2, 3]})
        snap = m.snapshot()
        snap["data"]["list"].append(4)
        assert m.get("data")["list"] == [1, 2, 3]  # Original unmodified

    def test_contains(self):
        m = WorkflowMemory()
        m.set("exists", True)
        assert "exists" in m
        assert "missing" not in m

    def test_to_dict(self):
        m = WorkflowMemory()
        m.update({"a": 1, "b": 2})
        d = m.to_dict()
        assert d == {"a": 1, "b": 2}


# ── WorkflowTrace Tests ───────────────────────────────────────────────────────

class TestWorkflowTrace:
    def _make_step_trace(self, step_id: str, llm_calls=1, tokens=100) -> StepTrace:
        from datetime import datetime
        return StepTrace(
            step_id=step_id,
            agent_name="TestAgent",
            status=StepStatus.COMPLETED,
            input_snapshot={},
            output={"result": "ok"},
            confidence=0.9,
            duration_ms=500.0,
            llm_calls=llm_calls,
            tokens_used=tokens,
            error=None,
            started_at=datetime.now().isoformat(),
            completed_at=datetime.now().isoformat(),
        )

    def test_add_step_accumulates_totals(self):
        trace = WorkflowTrace(workflow_name="test")
        trace.add_step(self._make_step_trace("step1", llm_calls=1, tokens=100))
        trace.add_step(self._make_step_trace("step2", llm_calls=2, tokens=200))
        assert trace.total_llm_calls == 3
        assert trace.total_tokens_used == 300
        assert len(trace.steps) == 2

    def test_complete_sets_status(self):
        trace = WorkflowTrace(workflow_name="test")
        trace.complete(WorkflowStatus.COMPLETED)
        assert trace.status == WorkflowStatus.COMPLETED
        assert trace.completed_at is not None

    def test_to_json_valid(self):
        trace = WorkflowTrace(workflow_name="test")
        trace.add_step(self._make_step_trace("step1"))
        trace.complete(WorkflowStatus.COMPLETED)
        parsed = json.loads(trace.to_json())
        assert parsed["workflow_name"] == "test"
        assert len(parsed["steps"]) == 1
        assert parsed["status"] == "completed"


# ── ClassifierAgent Tests ─────────────────────────────────────────────────────

class TestClassifierAgent:
    def setup_method(self):
        self.agent = ClassifierAgent()

    def test_parse_valid_output(self):
        raw = {
            "area": "trabalhista",
            "subarea": "rescisao_contratual",
            "urgencia": "media",
            "procedimento": "rito_ordinario",
            "complexidade": "medio",
            "partes": {"requerente": "João", "requerido": "Empresa"},
            "fatos_principais": ["demissão sem justa causa"],
            "confidence": 0.92,
        }
        output, confidence = self.agent._parse_output(raw)
        assert output["area"] == "trabalhista"
        assert output["urgencia"] == "media"
        assert confidence == 0.92

    def test_invalid_area_defaults_to_outros(self):
        raw = {"area": "invalido", "urgencia": "media", "complexidade": "medio",
               "procedimento": "a_definir", "confidence": 0.9}
        output, confidence = self.agent._parse_output(raw)
        assert output["area"] == "outros"
        assert confidence < 0.9  # Penalized

    def test_invalid_urgencia_defaults_to_media(self):
        raw = {"area": "civil", "urgencia": "xyzzy", "complexidade": "medio",
               "procedimento": "a_definir", "confidence": 0.8}
        output, confidence = self.agent._parse_output(raw)
        assert output["urgencia"] == "media"

    def test_confidence_clamped(self):
        raw = {"area": "civil", "urgencia": "media", "complexidade": "medio",
               "procedimento": "a_definir", "confidence": 1.5}
        _, confidence = self.agent._parse_output(raw)
        assert confidence <= 1.0


# ── AnalystAgent Tests ────────────────────────────────────────────────────────

class TestAnalystAgent:
    def setup_method(self):
        self.agent = AnalystAgent()

    def test_parse_probability_to_categoria(self):
        # High probability → favoravel
        raw = {"probabilidade_exito": 0.75, "riscos": [], "confidence": 0.85}
        output, _ = self.agent._parse_output(raw)
        assert output["categoria_exito"] == "favoravel"

    def test_low_probability_categoria(self):
        raw = {"probabilidade_exito": 0.25, "riscos": [], "confidence": 0.8}
        output, _ = self.agent._parse_output(raw)
        assert output["categoria_exito"] == "desfavoravel"

    def test_risk_normalization(self):
        raw = {
            "probabilidade_exito": 0.6,
            "riscos": [
                {"tipo": "prescricao", "descricao": "prazo vencendo", "severidade": "INVALIDA"},
                {"tipo": "prova", "descricao": "falta de documentos", "severidade": "alta"},
            ],
            "confidence": 0.8,
        }
        output, _ = self.agent._parse_output(raw)
        assert output["riscos"][0]["severidade"] == "media"  # Invalid → media
        assert output["riscos"][1]["severidade"] == "alta"


# ── ReviewerAgent Tests ───────────────────────────────────────────────────────

class TestReviewerAgent:
    def setup_method(self):
        self.agent = ReviewerAgent()

    def test_high_severity_forces_not_approved(self):
        raw = {
            "aprovado": True,  # Agent said approved...
            "score_qualidade": 0.9,
            "issues": [
                {"tipo": "citacao", "descricao": "art. 999 não existe", "severidade": "alta"}
            ],
            "recomendacao": "aprovar",
            "confidence": 0.85,
        }
        output, _ = self.agent._parse_output(raw)
        # Should override approval due to high severity issue
        assert output["aprovado"] is False

    def test_low_score_triggers_revision(self):
        raw = {
            "aprovado": False,
            "score_qualidade": 0.55,
            "issues": [{"tipo": "completude", "descricao": "falta pedido", "severidade": "media"}],
            "recomendacao": "aprovar",  # Inconsistent — should be overridden
            "confidence": 0.8,
        }
        output, _ = self.agent._parse_output(raw)
        assert output["recomendacao"] in ("revisar", "rejeitar")

    def test_clean_review_approved(self):
        raw = {
            "aprovado": True,
            "score_qualidade": 0.88,
            "issues": [],
            "sugestoes": ["melhorar introdução"],
            "secoes_ok": ["dos_fatos", "do_direito", "dos_pedidos"],
            "secoes_problematicas": [],
            "recomendacao": "aprovar",
            "confidence": 0.9,
        }
        output, confidence = self.agent._parse_output(raw)
        assert output["aprovado"] is True
        assert output["recomendacao"] == "aprovar"
        assert confidence == 0.9


# ── WorkflowStep Tests ────────────────────────────────────────────────────────

class TestWorkflowSteps:
    def test_agent_step_condition_true(self):
        mock_agent = MagicMock()
        mock_agent.name = "MockAgent"
        step = AgentStep(
            step_id="test",
            agent=mock_agent,
            memory_key="result",
            condition=lambda m: m.get("flag") is True,
        )
        m = WorkflowMemory()
        m.set("flag", True)
        assert step.should_run(m) is True

    def test_agent_step_condition_false(self):
        mock_agent = MagicMock()
        mock_agent.name = "MockAgent"
        step = AgentStep(
            step_id="test",
            agent=mock_agent,
            memory_key="result",
            condition=lambda m: m.get("flag") is True,
        )
        m = WorkflowMemory()
        m.set("flag", False)
        assert step.should_run(m) is False

    def test_step_condition_error_defaults_to_run(self):
        mock_agent = MagicMock()
        mock_agent.name = "MockAgent"
        step = AgentStep(
            step_id="test",
            agent=mock_agent,
            memory_key="result",
            condition=lambda m: 1 / 0,  # Always errors
        )
        m = WorkflowMemory()
        assert step.should_run(m) is True  # Default to running

    def test_no_condition_always_runs(self):
        mock_agent = MagicMock()
        mock_agent.name = "MockAgent"
        step = AgentStep(step_id="test", agent=mock_agent, memory_key="result")
        assert step.should_run(WorkflowMemory()) is True


# ── WorkflowEngine Tests (Mock LLM) ─────────────────────────────────────────

def make_mock_agent(name: str, output: dict, confidence: float = 0.9) -> MagicMock:
    """Create a mock agent that returns a pre-defined StepResult."""
    agent = MagicMock(spec=BaseAgent)
    agent.name = name

    def execute(step_id, memory):
        return StepResult(
            step_id=step_id,
            status=StepStatus.COMPLETED,
            output=output,
            confidence=confidence,
            duration_ms=10.0,
            llm_calls=1,
            tokens_used=100,
            agent_name=name,
        )

    agent.execute = execute
    return agent


class TestWorkflowEngine:
    def setup_method(self):
        self.engine = WorkflowEngine()

    def test_register_and_run(self):
        classify_output = {
            "area": "trabalhista", "subarea": "rescisao", "urgencia": "media",
            "complexidade": "medio", "procedimento": "rito_ordinario",
            "partes": {}, "fatos_principais": [], "confidence": 0.9,
        }
        mock_classifier = make_mock_agent("ClassifierAgent", classify_output)

        workflow = WorkflowDefinition(
            name="test_workflow",
            steps=[
                AgentStep(
                    step_id="classify",
                    agent=mock_classifier,
                    memory_key="classification",
                    required=True,
                )
            ]
        )
        self.engine.register(workflow)

        result = self.engine.run("test_workflow", {"caso": "teste"})
        assert result.status == WorkflowStatus.COMPLETED
        assert result.memory.get("classification") == classify_output

    def test_failed_required_step_fails_workflow(self):
        failing_agent = MagicMock(spec=BaseAgent)
        failing_agent.name = "FailAgent"
        failing_agent.execute.return_value = StepResult(
            step_id="fail_step",
            status=StepStatus.FAILED,
            error="Simulated failure",
            agent_name="FailAgent",
        )

        workflow = WorkflowDefinition(
            name="fail_workflow",
            steps=[
                AgentStep(
                    step_id="fail_step",
                    agent=failing_agent,
                    memory_key="result",
                    required=True,
                    max_retries=1,
                )
            ]
        )
        self.engine.register(workflow)

        result = self.engine.run("fail_workflow", {})
        assert result.status == WorkflowStatus.FAILED

    def test_failed_optional_step_continues(self):
        failing_agent = MagicMock(spec=BaseAgent)
        failing_agent.name = "FailAgent"
        failing_agent.execute.return_value = StepResult(
            step_id="opt_step",
            status=StepStatus.FAILED,
            error="Optional failure",
            agent_name="FailAgent",
        )

        success_agent = make_mock_agent("SuccessAgent", {"ok": True})

        workflow = WorkflowDefinition(
            name="optional_fail_workflow",
            steps=[
                AgentStep(
                    step_id="opt_step",
                    agent=failing_agent,
                    memory_key="optional_result",
                    required=False,
                    max_retries=1,
                ),
                AgentStep(
                    step_id="required_step",
                    agent=success_agent,
                    memory_key="final_result",
                    required=True,
                ),
            ]
        )
        self.engine.register(workflow)

        result = self.engine.run("optional_fail_workflow", {})
        assert result.status == WorkflowStatus.COMPLETED
        assert result.memory.get("final_result") == {"ok": True}

    def test_skipped_step_not_in_failure(self):
        workflow = WorkflowDefinition(
            name="skip_workflow",
            steps=[
                AgentStep(
                    step_id="always_skip",
                    agent=make_mock_agent("A", {}),
                    memory_key="skipped",
                    condition=lambda m: False,
                    required=True,  # Even required can be skipped
                ),
                AgentStep(
                    step_id="always_run",
                    agent=make_mock_agent("B", {"ran": True}),
                    memory_key="ran",
                ),
            ]
        )
        self.engine.register(workflow)
        result = self.engine.run("skip_workflow", {})
        assert result.status == WorkflowStatus.COMPLETED
        assert result.memory.get("ran") == {"ran": True}
        assert result.memory.get("skipped") is None

    def test_human_gate_auto_approved_non_interactive(self):
        human_gate = HumanGateStep(
            step_id="gate",
            prompt_fn=lambda m: "Review this",
            require_approval=True,
            description="Test gate",
        )
        success_agent = make_mock_agent("A", {"done": True})

        workflow = WorkflowDefinition(
            name="gate_workflow",
            steps=[human_gate, AgentStep(step_id="s", agent=success_agent, memory_key="done")],
        )
        self.engine.register(workflow)
        result = self.engine.run("gate_workflow", {}, interactive=False)
        assert result.status == WorkflowStatus.COMPLETED

    def test_unregistered_workflow_raises(self):
        with pytest.raises(ValueError, match="not registered"):
            self.engine.run("nonexistent", {})

    def test_trace_captures_all_steps(self):
        agents = [make_mock_agent(f"Agent{i}", {"i": i}) for i in range(3)]
        workflow = WorkflowDefinition(
            name="trace_workflow",
            steps=[
                AgentStep(step_id=f"step{i}", agent=a, memory_key=f"k{i}")
                for i, a in enumerate(agents)
            ]
        )
        self.engine.register(workflow)
        result = self.engine.run("trace_workflow", {})
        assert len(result.trace.steps) == 3
        assert result.trace.total_llm_calls == 3
        assert result.trace.total_tokens_used == 300


# ── Workflow Definitions Tests ────────────────────────────────────────────────

class TestWorkflowDefinitions:
    def test_triagem_rapida_has_2_steps(self):
        w = triagem_rapida_workflow()
        assert len(w.steps) == 2
        assert w.steps[0].step_id == "classify"
        assert w.steps[1].step_id == "analyze"

    def test_peticao_inicial_has_6_steps(self):
        w = peticao_inicial_workflow()
        assert len(w.steps) == 6
        step_ids = [s.step_id for s in w.steps]
        assert "classify" in step_ids
        assert "research" in step_ids
        assert "draft" in step_ids
        assert "review" in step_ids
        assert "human_approval" in step_ids

    def test_recurso_ordinario_has_4_steps(self):
        w = recurso_ordinario_workflow()
        assert len(w.steps) == 4

    def test_draft_condition_skips_unviable_case(self):
        """Draft step should be skipped when probability is too low."""
        w = recurso_ordinario_workflow()
        draft_step = w.get_step("draft")
        assert draft_step is not None

        m = WorkflowMemory()
        m.set("analise", {"probabilidade_exito": 0.1})  # Below threshold
        assert draft_step.should_run(m) is False

        m.set("analise", {"probabilidade_exito": 0.5})  # Above threshold
        assert draft_step.should_run(m) is True


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
