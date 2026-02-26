"""
Drafter Agent — Legal Brief Drafting.

Drafts the legal brief (peça processual) based on:
  - Classification (procedimento, área, partes)
  - Research (fundamentos, legislação, súmulas)
  - Analysis (estratégia, pedidos, valor da causa)

The drafter produces structured output with separate sections,
not a monolithic text blob. This allows the reviewer to check
each section independently and the system to assemble the final
document from validated parts.

Output structure:
  1. Qualificação das partes
  2. Dos fatos (narrative of events)
  3. Do direito (legal arguments with citations)
  4. Dos pedidos (prayer for relief)
  5. Valor da causa

Design choice: two-pass drafting.
  Pass 1: Draft the 'Dos Fatos' and 'Do Direito' sections
  Pass 2: Draft 'Dos Pedidos' using outputs of pass 1

Two passes avoid the common problem of pedidos that don't match
the direito section (the model forgets what it argued).
In this implementation we do a single-pass for brevity,
but the two-pass approach is documented for production use.

Output contract (written to memory as 'minuta'):
  {
    "qualificacao": "texto da qualificação",
    "dos_fatos": "texto narrativo dos fatos",
    "do_direito": "argumentação jurídica com citações",
    "dos_pedidos": "lista numerada de pedidos",
    "valor_causa": "R$ X.XXX,00",
    "tipo_peca": "peticao_inicial|recurso|contestacao|etc.",
    "observacoes_redacao": "notas para o revisor",
    "confidence": 0.80
  }
"""
from __future__ import annotations

from src.agents.base_agent import BaseAgent
from src.utils.workflow_models import WorkflowMemory


class DrafterAgent(BaseAgent):
    """Drafts legal briefs (petições, recursos) based on research and analysis."""

    @property
    def name(self) -> str:
        return "DrafterAgent"

    @property
    def description(self) -> str:
        return "Redige a minuta da peça processual com base na pesquisa e análise."

    def _build_prompt(self, memory: WorkflowMemory) -> tuple[str, str]:
        classification = memory.get("classification", {})
        pesquisa = memory.get("pesquisa", {})
        analise = memory.get("analise", {})
        caso = memory.get("caso", "")
        cliente = memory.get("cliente", "")
        parte_contraria = memory.get("parte_contraria", "")

        area = classification.get("area", "")
        procedimento = classification.get("procedimento", "rito_ordinario")
        partes = classification.get("partes", {})
        fatos = classification.get("fatos_principais", [])

        pedidos = analise.get("pedidos_sugeridos", [])
        estrategia = analise.get("estrategia", "")
        valor_causa = analise.get("valor_causa_estimado", "a ser arbitrado pelo juízo")

        legislacao = pesquisa.get("legislacao_principal", [])
        sumulas = pesquisa.get("sumulas", [])
        fund_favor = pesquisa.get("fundamentos_favor", [])
        observacoes_pesquisa = pesquisa.get("observacoes", "")

        requerente = partes.get("requerente") or cliente or "REQUERENTE"
        requerido = partes.get("requerido") or parte_contraria or "REQUERIDO"

        system = """\
Você é um advogado experiente especializado em redação processual brasileira.
Redija a peça processual seguindo o padrão forense brasileiro.

LINGUAGEM: formal, técnica, em português do Brasil
FORMATO: cada seção separada, citações em negrito implícito
TOM: objetivo, fundamentado, sem exageros

Retorne JSON com este schema:
{
  "qualificacao": "texto de qualificação das partes",
  "dos_fatos": "narrativa dos fatos (mínimo 3 parágrafos)",
  "do_direito": "argumentação jurídica com citações (mínimo 4 parágrafos)",
  "dos_pedidos": "pedidos numerados em formato processual",
  "valor_causa": "R$ X.XXX,00",
  "tipo_peca": "peticao_inicial|recurso_ordinario|contestacao|recurso_de_revista|etc.",
  "juizo_competente": "Juízo ou Tribunal competente",
  "observacoes_redacao": "notas sobre pontos que merecem atenção do revisor",
  "confidence": 0.0
}

INSTRUÇÕES PARA DOS FATOS:
- Narrativa cronológica dos fatos
- Linguagem objetiva, sem adjetivos desnecessários
- Destaque fatos que fundamentam os pedidos

INSTRUÇÕES PARA DO DIREITO:
- Cite artigos de lei com número e legislação
- Cite súmulas com número e tribunal
- Conecte cada fundamento a um pedido específico
- Refute previsíveis argumentos contrários

INSTRUÇÕES PARA DOS PEDIDOS:
- Pedido principal em primeiro
- Pedidos subsidiários numerados
- Inclua condenação em honorários e custas
- Requeira produção de provas
"""
        user = f"""ÁREA: {area}
PROCEDIMENTO: {procedimento}
REQUERENTE: {requerente}
REQUERIDO: {requerido}

FATOS DO CASO:
{caso[:2000]}

FATOS PRINCIPAIS IDENTIFICADOS: {', '.join(fatos)}

ESTRATÉGIA PROCESSUAL: {estrategia}

LEGISLAÇÃO APLICÁVEL: {', '.join(legislacao[:8])}
SÚMULAS: {', '.join(sumulas[:5])}

FUNDAMENTOS FAVORÁVEIS:
{chr(10).join('- ' + f for f in fund_favor[:5])}

PEDIDOS SUGERIDOS:
{chr(10).join(f'{i+1}. {p}' for i, p in enumerate(pedidos))}

VALOR DA CAUSA: {valor_causa}
{('OBSERVAÇÕES: ' + observacoes_pesquisa) if observacoes_pesquisa else ''}

Redija a peça processual."""

        return system, user

    def _parse_output(self, raw: dict) -> tuple[dict, float]:
        confidence = float(raw.get("confidence", 0.75))
        confidence = max(0.0, min(1.0, confidence))

        # Validate that major sections are present and non-empty
        required_sections = ["dos_fatos", "do_direito", "dos_pedidos"]
        for section in required_sections:
            if not raw.get(section, "").strip():
                confidence *= 0.5  # Penalize missing sections

        output = {
            "qualificacao": raw.get("qualificacao", ""),
            "dos_fatos": raw.get("dos_fatos", ""),
            "do_direito": raw.get("do_direito", ""),
            "dos_pedidos": raw.get("dos_pedidos", ""),
            "valor_causa": raw.get("valor_causa", ""),
            "tipo_peca": raw.get("tipo_peca", "peticao_inicial"),
            "juizo_competente": raw.get("juizo_competente", ""),
            "observacoes_redacao": raw.get("observacoes_redacao", ""),
        }

        return output, confidence
