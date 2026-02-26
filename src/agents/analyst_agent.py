"""
Analyst Agent — Merit Assessment and Risk Analysis.

Evaluates the legal case merits based on:
  - Classification output (area, complexity, urgency)
  - Research output (applicable law, jurisprudence)
  - Case facts

Produces a structured analysis with:
  1. Probabilidade de êxito (score 0-1 + categoria)
  2. Principais riscos processuais e materiais
  3. Estratégia recomendada
  4. Pedidos sugeridos (para o drafter)
  5. Alertas (prescrição, decadência, prazo processual)

Why risk analysis matters:
  The drafter needs to know what to argue and what to avoid.
  A strong case has different drafting strategy than a risky one.
  The analyst bridges research (what the law says) and drafting (what to write).

Output contract (written to memory as 'analise'):
  {
    "probabilidade_exito": 0.72,
    "categoria_exito": "favoravel|incerto|desfavoravel",
    "pedidos_sugeridos": ["pedido principal", "pedido subsidiário"],
    "estrategia": "descrição da estratégia processual recomendada",
    "riscos": [
      {"tipo": "prescricao", "descricao": "...", "severidade": "alta"}
    ],
    "alertas_prazo": ["atenção: prazo de X dias para Y"],
    "valor_causa_estimado": "R$ X.XXX,00 ou null",
    "confidence": 0.82
  }
"""
from __future__ import annotations

from src.agents.base_agent import BaseAgent
from src.utils.workflow_models import WorkflowMemory


class AnalystAgent(BaseAgent):
    """Analyzes case merits, risks, and recommends litigation strategy."""

    @property
    def name(self) -> str:
        return "AnalystAgent"

    @property
    def description(self) -> str:
        return "Avalia mérito, riscos e estratégia processual do caso."

    def _build_prompt(self, memory: WorkflowMemory) -> tuple[str, str]:
        classification = memory.get("classification", {})
        pesquisa = memory.get("pesquisa", {})
        caso = memory.get("caso", "")
        cliente = memory.get("cliente", "")

        area = classification.get("area", "")
        subarea = classification.get("subarea", "")
        complexidade = classification.get("complexidade", "")
        procedimento = classification.get("procedimento", "")
        fatos = classification.get("fatos_principais", [])

        leg_principal = pesquisa.get("legislacao_principal", [])
        sumulas = pesquisa.get("sumulas", [])
        fund_favor = pesquisa.get("fundamentos_favor", [])
        fund_contra = pesquisa.get("fundamentos_contra", [])
        jurisp = pesquisa.get("jurisprudencia_dominante", "controvertida")

        system = """\
Você é um advogado sênior brasileiro especializado em análise de mérito processual. 
Analise o caso e produza uma avaliação jurídica estruturada.

Retorne JSON com este schema:
{
  "probabilidade_exito": 0.0,
  "categoria_exito": "favoravel|incerto|desfavoravel",
  "pedidos_sugeridos": ["pedido principal", ...],
  "estrategia": "descrição da estratégia processual em 2-4 frases",
  "riscos": [
    {"tipo": "string", "descricao": "string", "severidade": "alta|media|baixa"}
  ],
  "alertas_prazo": ["alerta sobre prazo ou preclusão"],
  "valor_causa_estimado": "R$ X.XXX,00 ou null se não estimável",
  "recomendacao_cliente": "o que comunicar ao cliente em 1-2 frases",
  "confidence": 0.0
}

CRITÉRIOS DE PROBABILIDADE:
  0.8-1.0 → altamente favorável (jurisprudência consolidada, fatos claramente provados)
  0.6-0.8 → favorável (maioria da jurisprudência apoia, fatos razoavelmente comprovados)
  0.4-0.6 → incerto (jurisprudência controvertida ou fatos insuficientes)
  0.2-0.4 → desfavorável (jurisprudência contrária ou fatos desfavoráveis)
  0.0-0.2 → altamente desfavorável

TIPOS DE RISCO: prescricao, decadencia, prova, tecnico, economico, reputacional
"""
        user = f"""ÁREA: {area} / {subarea}
COMPLEXIDADE: {complexidade}
PROCEDIMENTO: {procedimento}
JURISPRUDÊNCIA DOMINANTE: {jurisp}

CASO:
{caso[:1500]}

CLIENTE: {cliente or 'não informado'}
FATOS PRINCIPAIS: {', '.join(fatos)}

FUNDAMENTOS FAVORÁVEIS:
{chr(10).join('- ' + f for f in fund_favor[:5])}

FUNDAMENTOS CONTRÁRIOS:
{chr(10).join('- ' + f for f in fund_contra[:3])}

LEGISLAÇÃO APLICÁVEL: {', '.join(leg_principal[:5])}
SÚMULAS: {', '.join(sumulas[:5])}

Produza a análise de mérito."""

        return system, user

    def _parse_output(self, raw: dict) -> tuple[dict, float]:
        confidence = float(raw.get("confidence", 0.7))
        confidence = max(0.0, min(1.0, confidence))

        prob = float(raw.get("probabilidade_exito", 0.5))
        prob = max(0.0, min(1.0, prob))

        # Auto-compute category from probability
        if prob >= 0.6:
            categoria = "favoravel"
        elif prob >= 0.4:
            categoria = "incerto"
        else:
            categoria = "desfavoravel"

        # Override with explicit if valid
        explicit_cat = raw.get("categoria_exito", "")
        if explicit_cat in ("favoravel", "incerto", "desfavoravel"):
            categoria = explicit_cat

        # Validate risks
        riscos = []
        for risk in raw.get("riscos", []):
            if isinstance(risk, dict) and "tipo" in risk:
                sev = risk.get("severidade", "media")
                if sev not in ("alta", "media", "baixa"):
                    sev = "media"
                riscos.append({
                    "tipo": risk["tipo"],
                    "descricao": risk.get("descricao", ""),
                    "severidade": sev,
                })

        output = {
            "probabilidade_exito": prob,
            "categoria_exito": categoria,
            "pedidos_sugeridos": raw.get("pedidos_sugeridos", []),
            "estrategia": raw.get("estrategia", ""),
            "riscos": riscos,
            "alertas_prazo": raw.get("alertas_prazo", []),
            "valor_causa_estimado": raw.get("valor_causa_estimado"),
            "recomendacao_cliente": raw.get("recomendacao_cliente", ""),
        }

        return output, confidence
