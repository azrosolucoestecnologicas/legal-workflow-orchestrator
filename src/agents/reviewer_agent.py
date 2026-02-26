"""
Reviewer Agent — Quality Control for Drafted Legal Briefs.

Reviews the minuta produced by DrafterAgent and checks:
  1. Consistência interna: pedidos batem com a fundamentação?
  2. Citações: as citações existem no direito brasileiro?
  3. Completude: todas as seções obrigatórias presentes?
  4. Coerência com a análise: pedidos batem com o que o AnalystAgent recomendou?
  5. Formalidade: linguagem forense adequada?

The reviewer does NOT rewrite — it produces a structured review report
with specific issues and recommendations. The engine can then decide
to either approve, reject (and re-run the drafter), or route to
human review.

Why a separate reviewer agent?
  The drafter cannot reliably self-review. The reviewer has a different
  "role" in its system prompt — it's adversarial (looking for problems)
  rather than constructive. This exploits the fact that LLMs perform
  better when their role is clearly defined and focused.

Output contract (written to memory as 'revisao'):
  {
    "aprovado": true|false,
    "score_qualidade": 0.0-1.0,
    "issues": [
      {"tipo": "citacao_inexistente", "descricao": "...", "severidade": "alta|media|baixa"}
    ],
    "sugestoes": ["sugestão de melhoria 1", ...],
    "secoes_ok": ["dos_fatos", "do_direito", "dos_pedidos"],
    "secoes_problematicas": ["qualificacao"],
    "recomendacao": "aprovar|revisar|rejeitar",
    "confidence": 0.88
  }

Routing rules (used by WorkflowEngine):
  aprovado=True + score >= 0.80 → proceed to human gate or finalize
  aprovado=False + severidade alta → re-run drafter (max 2 retries)
  aprovado=False + severidade media → proceed with warnings
"""
from __future__ import annotations

from src.agents.base_agent import BaseAgent
from src.utils.workflow_models import WorkflowMemory


class ReviewerAgent(BaseAgent):
    """Reviews drafted legal briefs for quality, consistency, and completeness."""

    @property
    def name(self) -> str:
        return "ReviewerAgent"

    @property
    def description(self) -> str:
        return "Revisa a minuta processual verificando consistência, citações e completude."

    def _build_prompt(self, memory: WorkflowMemory) -> tuple[str, str]:
        minuta = memory.get("minuta", {})
        analise = memory.get("analise", {})
        pesquisa = memory.get("pesquisa", {})

        system = """\
Você é um revisor jurídico experiente. Revise criticamente a peça processual a seguir.

Retorne JSON com este schema:
{
  "aprovado": true,
  "score_qualidade": 0.0,
  "issues": [
    {"tipo": "string", "descricao": "string", "severidade": "alta|media|baixa"}
  ],
  "sugestoes": ["sugestão concreta de melhoria"],
  "secoes_ok": ["nomes das seções sem problemas"],
  "secoes_problematicas": ["nomes das seções com problemas"],
  "recomendacao": "aprovar|revisar|rejeitar",
  "confidence": 0.0
}

CRITÉRIOS DE AVALIAÇÃO (verifique cada um):

1. CONSISTÊNCIA (alta severidade se falhar):
   - Os pedidos batem com a fundamentação jurídica?
   - Os fatos narrados suportam os pedidos?

2. CITAÇÕES (alta severidade se falhar):
   - As citações de artigos de lei existem?
   - As súmulas citadas existem e são do tribunal correto?
   - Os números são plausíveis?

3. COMPLETUDE (média severidade se falhar):
   - Todas as seções principais presentes? (fatos, direito, pedidos)
   - Pedido de honorários incluído?
   - Requerimento de provas incluído?

4. ADEQUAÇÃO COM A ANÁLISE (média severidade):
   - Os pedidos batem com os pedidos sugeridos na análise?
   - A estratégia da análise está refletida na peça?

5. FORMALIDADE (baixa severidade):
   - Linguagem forense adequada?
   - Ausência de gírias ou informalidades?

SCORE DE QUALIDADE:
  1.0 = peça excelente, pronta para protocolo
  0.8-0.9 = boa, com ajustes menores
  0.6-0.8 = precisa de revisão
  < 0.6 = rejeitar e redigir novamente

RECOMENDAÇÃO:
  aprovado true + score >= 0.75 → "aprovar"
  aprovado true + score < 0.75 → "revisar"
  aprovado false + issues alta → "rejeitar"
  aprovado false + apenas issues media/baixa → "revisar"
"""
        pedidos_analise = analise.get("pedidos_sugeridos", [])
        legislacao = pesquisa.get("legislacao_principal", [])

        user = f"""PEÇA A REVISAR:

QUALIFICAÇÃO:
{minuta.get('qualificacao', '(não informado)')}

DOS FATOS:
{minuta.get('dos_fatos', '(não informado)')}

DO DIREITO:
{minuta.get('do_direito', '(não informado)')}

DOS PEDIDOS:
{minuta.get('dos_pedidos', '(não informado)')}

VALOR DA CAUSA: {minuta.get('valor_causa', '(não informado)')}

---
CONTEXTO PARA REVISÃO:
Pedidos sugeridos pela análise: {', '.join(pedidos_analise) if pedidos_analise else 'não disponível'}
Legislação identificada na pesquisa: {', '.join(legislacao[:6]) if legislacao else 'não disponível'}

Revise a peça e produza o relatório de revisão."""

        return system, user

    def _parse_output(self, raw: dict) -> tuple[dict, float]:
        confidence = float(raw.get("confidence", 0.8))
        confidence = max(0.0, min(1.0, confidence))

        score = float(raw.get("score_qualidade", 0.5))
        score = max(0.0, min(1.0, score))

        aprovado = bool(raw.get("aprovado", False))

        # Validate recomendacao
        recomendacao = raw.get("recomendacao", "revisar")
        if recomendacao not in ("aprovar", "revisar", "rejeitar"):
            recomendacao = "revisar"

        # Normalize issues
        issues = []
        for issue in raw.get("issues", []):
            if isinstance(issue, dict) and "descricao" in issue:
                sev = issue.get("severidade", "baixa")
                if sev not in ("alta", "media", "baixa"):
                    sev = "baixa"
                issues.append({
                    "tipo": issue.get("tipo", "geral"),
                    "descricao": issue["descricao"],
                    "severidade": sev,
                })

        # Consistency: if high-severity issues, force não aprovado
        has_alta = any(i["severidade"] == "alta" for i in issues)
        if has_alta and aprovado:
            aprovado = False
            recomendacao = "rejeitar" if score < 0.6 else "revisar"

        output = {
            "aprovado": aprovado,
            "score_qualidade": score,
            "issues": issues,
            "sugestoes": raw.get("sugestoes", []),
            "secoes_ok": raw.get("secoes_ok", []),
            "secoes_problematicas": raw.get("secoes_problematicas", []),
            "recomendacao": recomendacao,
        }

        return output, confidence
