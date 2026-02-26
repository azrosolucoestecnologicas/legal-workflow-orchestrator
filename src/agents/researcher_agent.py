"""
Researcher Agent — Legal Research and Jurisprudence Identification.

Identifies the most relevant legal basis for a classified case:
  1. Legislation (artigos de lei, códigos)
  2. Súmulas (TST, STF, STJ)
  3. Orientações Jurisprudenciais (OJ SDI-1, SDI-2, SDC)
  4. Leading cases / precedentes vinculantes
  5. Teses firmadas em recursos repetitivos

The researcher works from the classification output and the case facts.
It does NOT search an external database — it uses the LLM's parametric
knowledge of Brazilian law, supplemented by a curated reference document
if provided.

For production, extend this agent to call a real retrieval system:
  - Inject a `retrieval_fn` parameter
  - Call it with the case subarea and key terms
  - Append results to the context before the LLM call

Output contract (written to memory as 'pesquisa'):
  {
    "legislacao_principal": ["art. 483 CLT", "art. 7° CF/88"],
    "sumulas": ["Súmula 331 TST", "Súmula 85 TST"],
    "orientacoes": ["OJ 394 SDI-1"],
    "teses_repetitivos": ["IRR-xxx/TST: ..."],
    "fundamentos_favor": ["descrição do fundamento 1..."],
    "fundamentos_contra": ["possível tese defensiva..."],
    "jurisprudencia_dominante": "favoravel|desfavoravel|controvertida",
    "confidence": 0.85
  }
"""
from __future__ import annotations

from typing import Optional, Callable
from src.agents.base_agent import BaseAgent
from src.utils.workflow_models import WorkflowMemory


class ResearcherAgent(BaseAgent):
    """Researches applicable law, súmulas, and jurisprudence for the case."""

    def __init__(self, retrieval_fn: Optional[Callable] = None, **kwargs):
        """
        Args:
            retrieval_fn: Optional callable(query: str) -> list[str].
                          If provided, called before LLM to add real retrieved docs.
        """
        super().__init__(**kwargs)
        self._retrieval_fn = retrieval_fn

    @property
    def name(self) -> str:
        return "ResearcherAgent"

    @property
    def description(self) -> str:
        return "Pesquisa legislação, súmulas e jurisprudência aplicável ao caso."

    def _build_prompt(self, memory: WorkflowMemory) -> tuple[str, str]:
        classification = memory.get("classification", {})
        area = classification.get("area", "desconhecida")
        subarea = classification.get("subarea", "")
        fatos = classification.get("fatos_principais", [])
        caso = memory.get("caso", "")

        # Optionally enrich with retrieved documents
        retrieved_context = ""
        if self._retrieval_fn:
            query = f"{area} {subarea} {' '.join(fatos[:3])}"
            try:
                docs = self._retrieval_fn(query)
                if docs:
                    retrieved_context = "\n\nDOCUMENTOS RECUPERADOS:\n" + "\n---\n".join(docs[:5])
            except Exception as e:
                self.logger.warning(f"Retrieval failed: {e}")

        system = """\
Você é um especialista em pesquisa jurídica brasileira. Identifique os fundamentos legais aplicáveis.

Retorne JSON com este schema:
{
  "legislacao_principal": ["art. X da Lei Y", ...],
  "sumulas": ["Súmula N do TST/STF/STJ", ...],
  "orientacoes": ["OJ N da SDI-1/SDI-2/SDC", ...],
  "precedentes_vinculantes": ["IRR ou tese de repercussão geral", ...],
  "fundamentos_favor": ["descrição do fundamento favorável ao cliente", ...],
  "fundamentos_contra": ["possível argumento contrário/tese defensiva", ...],
  "jurisprudencia_dominante": "favoravel|desfavoravel|controvertida",
  "observacoes": "pontos de atenção ou peculiaridades desta área",
  "confidence": 0.0
}

REGRAS:
- Cite apenas fontes existentes no direito brasileiro
- Identifique teses que favorecem E que podem ser usadas contra o cliente
- jurisprudencia_dominante: avalie a tendência predominante dos tribunais
- confidence: sua confiança na pesquisa (considere se a subárea é bem definida)
"""
        user = f"""ÁREA: {area}
SUBÁREA: {subarea}
FATOS PRINCIPAIS: {', '.join(fatos) if fatos else caso[:500]}
CASO COMPLETO: {caso[:1000]}
{retrieved_context}

Pesquise os fundamentos legais aplicáveis."""

        return system, user

    def _parse_output(self, raw: dict) -> tuple[dict, float]:
        confidence = float(raw.get("confidence", 0.7))
        confidence = max(0.0, min(1.0, confidence))

        jurisprudencia = raw.get("jurisprudencia_dominante", "controvertida")
        if jurisprudencia not in ("favoravel", "desfavoravel", "controvertida"):
            jurisprudencia = "controvertida"

        output = {
            "legislacao_principal": raw.get("legislacao_principal", []),
            "sumulas": raw.get("sumulas", []),
            "orientacoes": raw.get("orientacoes", []),
            "precedentes_vinculantes": raw.get("precedentes_vinculantes", []),
            "fundamentos_favor": raw.get("fundamentos_favor", []),
            "fundamentos_contra": raw.get("fundamentos_contra", []),
            "jurisprudencia_dominante": jurisprudencia,
            "observacoes": raw.get("observacoes", ""),
        }

        return output, confidence
