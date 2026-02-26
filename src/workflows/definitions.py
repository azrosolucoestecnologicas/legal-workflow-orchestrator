"""
Workflow Definitions — Pre-built Legal Workflow DAGs.

Three workflows of increasing complexity:

1. triagem_rapida (2 agents, ~5s without LLM)
   Classify → Analyze viability → Done
   Use case: quick viability check before committing to full research

2. recurso_ordinario (4 agents)
   Classify → Research → Analyze → Draft → Done
   Use case: standard labor law appeal drafting

3. peticao_inicial (5 agents + human gate)
   Classify → Research → Analyze → Draft → Review
   → Human gate → Done
   Use case: full initial petition workflow with quality control

Conditional routing examples:
  - Research step only runs if classification confidence > 0.6
  - Drafter runs only if analysis indicates case is viable (prob_exito > 0.3)
  - Human gate activates when case is urgent OR complexity is high
  - Reviewer can trigger re-draft (via condition on retry logic)

Human gate prompt functions:
  Summarize classification + analysis for the reviewing lawyer.
  Include: area, urgência, probabilidade de êxito, pedidos, alertas de prazo.
"""
from __future__ import annotations

from src.agents.classifier_agent import ClassifierAgent
from src.agents.researcher_agent import ResearcherAgent
from src.agents.analyst_agent import AnalystAgent
from src.agents.drafter_agent import DrafterAgent
from src.agents.reviewer_agent import ReviewerAgent
from src.utils.workflow_models import WorkflowMemory
from src.workflows.workflow_steps import (
    AgentStep, HumanGateStep, WorkflowDefinition
)


def _human_gate_prompt(memory: WorkflowMemory) -> str:
    """Build human-readable summary for gate review."""
    classification = memory.get("classification", {})
    analise = memory.get("analise", {})
    revisao = memory.get("revisao", {})
    minuta = memory.get("minuta", {})

    lines = []

    # Header
    area = classification.get("area", "?").upper()
    subarea = classification.get("subarea", "?")
    urgencia = classification.get("urgencia", "?").upper()
    lines.append(f"ÁREA: {area} / {subarea}")
    lines.append(f"URGÊNCIA: {urgencia}")

    # Parties
    partes = classification.get("partes", {})
    if partes:
        lines.append(f"REQUERENTE: {partes.get('requerente', '?')}")
        lines.append(f"REQUERIDO:  {partes.get('requerido', '?')}")

    lines.append("")

    # Analysis
    prob = analise.get("probabilidade_exito", 0)
    cat = analise.get("categoria_exito", "?")
    lines.append(f"PROBABILIDADE DE ÊXITO: {prob:.0%} ({cat})")

    estrategia = analise.get("estrategia", "")
    if estrategia:
        lines.append(f"ESTRATÉGIA: {estrategia}")

    # Alerts
    alertas = analise.get("alertas_prazo", [])
    if alertas:
        lines.append("\n⚠️  ALERTAS DE PRAZO:")
        for a in alertas:
            lines.append(f"  • {a}")

    # Risks
    riscos = analise.get("riscos", [])
    altos = [r for r in riscos if r.get("severidade") == "alta"]
    if altos:
        lines.append("\n🔴 RISCOS ALTOS:")
        for r in altos:
            lines.append(f"  • {r.get('descricao', '')}")

    # Review result
    if revisao:
        score = revisao.get("score_qualidade", 0)
        rec = revisao.get("recomendacao", "?")
        lines.append(f"\nREVISÃO DA MINUTA: score={score:.0%}, recomendação={rec.upper()}")
        issues = [i for i in revisao.get("issues", []) if i.get("severidade") == "alta"]
        if issues:
            lines.append("Issues altos:")
            for i in issues:
                lines.append(f"  • {i.get('descricao', '')}")

    # Draft summary
    tipo_peca = minuta.get("tipo_peca", "")
    valor = minuta.get("valor_causa", "")
    if tipo_peca:
        lines.append(f"\nPEÇA: {tipo_peca.upper()}")
    if valor:
        lines.append(f"VALOR DA CAUSA: {valor}")

    return "\n".join(lines)


def triagem_rapida_workflow() -> WorkflowDefinition:
    """
    Triagem rápida: classifica o caso e avalia viabilidade.
    Use quando precisa de uma resposta rápida ao cliente antes de pesquisa completa.
    """
    classifier = ClassifierAgent()
    analyst = AnalystAgent()

    return WorkflowDefinition(
        name="triagem_rapida",
        description="Triagem: classificação + análise de viabilidade sem pesquisa aprofundada",
        steps=[
            AgentStep(
                step_id="classify",
                agent=classifier,
                memory_key="classification",
                description="Classificar tipo, urgência e complexidade do caso",
                max_retries=2,
                required=True,
            ),
            AgentStep(
                step_id="analyze",
                agent=analyst,
                memory_key="analise",
                description="Avaliar viabilidade e riscos do caso",
                max_retries=1,
                required=True,
                # Only analyze if classification was confident enough
                condition=lambda m: (m.get_nested("classification", "confidence") or 0) >= 0.5,
            ),
        ],
    )


def recurso_ordinario_workflow() -> WorkflowDefinition:
    """
    Workflow de recurso ordinário: classificação → pesquisa → análise → minuta.
    Sem revisão automática e sem human gate — para uso em batch.
    """
    classifier = ClassifierAgent()
    researcher = ResearcherAgent()
    analyst = AnalystAgent()
    drafter = DrafterAgent()

    return WorkflowDefinition(
        name="recurso_ordinario",
        description="Recurso ordinário: classificação → pesquisa → análise → minuta",
        steps=[
            AgentStep(
                step_id="classify",
                agent=classifier,
                memory_key="classification",
                description="Classificar o caso",
                max_retries=2,
                required=True,
            ),
            AgentStep(
                step_id="research",
                agent=researcher,
                memory_key="pesquisa",
                description="Pesquisar legislação e jurisprudência",
                max_retries=2,
                required=True,
                condition=lambda m: (m.get_nested("classification", "confidence") or 0) >= 0.5,
            ),
            AgentStep(
                step_id="analyze",
                agent=analyst,
                memory_key="analise",
                description="Analisar mérito e estratégia",
                max_retries=1,
                required=True,
            ),
            AgentStep(
                step_id="draft",
                agent=drafter,
                memory_key="minuta",
                description="Redigir a minuta do recurso",
                max_retries=2,
                required=True,
                # Only draft if case has some viability
                condition=lambda m: (m.get_nested("analise", "probabilidade_exito") or 0) >= 0.2,
            ),
        ],
    )


def peticao_inicial_workflow() -> WorkflowDefinition:
    """
    Workflow completo de petição inicial com revisão e human gate.
    Use quando o caso requer máxima qualidade e aprovação de advogado.
    """
    classifier = ClassifierAgent()
    researcher = ResearcherAgent()
    analyst = AnalystAgent()
    drafter = DrafterAgent()
    reviewer = ReviewerAgent()

    def is_case_viable(m: WorkflowMemory) -> bool:
        return (m.get_nested("analise", "probabilidade_exito") or 0) >= 0.2

    def needs_human_gate(m: WorkflowMemory) -> bool:
        """
        Trigger human gate when:
        - Case is urgent
        - Complexity is high
        - Review recommends revision or rejection
        - Any high-severity issues found
        """
        urgencia = m.get_nested("classification", "urgencia") or ""
        complexidade = m.get_nested("classification", "complexidade") or ""
        recomendacao = m.get_nested("revisao", "recomendacao") or "aprovar"
        score = m.get_nested("revisao", "score_qualidade") or 1.0

        return (
            urgencia == "urgente"
            or complexidade == "complexo"
            or recomendacao in ("revisar", "rejeitar")
            or score < 0.8
        )

    return WorkflowDefinition(
        name="peticao_inicial",
        description="Petição inicial completa: classificação → pesquisa → análise → minuta → revisão → aprovação",
        steps=[
            AgentStep(
                step_id="classify",
                agent=classifier,
                memory_key="classification",
                description="Classificar o caso jurídico",
                max_retries=2,
                required=True,
            ),
            AgentStep(
                step_id="research",
                agent=researcher,
                memory_key="pesquisa",
                description="Pesquisar fundamentos jurídicos",
                max_retries=2,
                required=True,
            ),
            AgentStep(
                step_id="analyze",
                agent=analyst,
                memory_key="analise",
                description="Analisar mérito e definir estratégia",
                max_retries=1,
                required=True,
            ),
            AgentStep(
                step_id="draft",
                agent=drafter,
                memory_key="minuta",
                description="Redigir a petição inicial",
                max_retries=2,
                required=True,
                condition=is_case_viable,
            ),
            AgentStep(
                step_id="review",
                agent=reviewer,
                memory_key="revisao",
                description="Revisar a minuta produzida",
                max_retries=1,
                required=False,  # Workflow continues even if review fails
                condition=lambda m: "minuta" in m,
            ),
            HumanGateStep(
                step_id="human_approval",
                prompt_fn=_human_gate_prompt,
                require_approval=True,
                description="Revisão e aprovação pelo advogado responsável",
                condition=needs_human_gate,
            ),
        ],
    )
