"""
Classifier Agent — Case Type and Urgency Classification.

Classifies an incoming legal case by:
  1. Area of law (trabalhista, civil, penal, tributário, etc.)
  2. Sub-area (rescisão, dano moral, cobrança, etc.)
  3. Urgency (urgente, média, baixa)
  4. Procedimento (rito sumaríssimo, ordinário, especial)
  5. Estimated complexity (simples, médio, complexo)

Why have a dedicated classifier agent?
  Downstream agents (researcher, analyst, drafter) need this classification
  to know which prompts, which knowledge bases, and which templates to use.
  Mixing classification logic into every agent creates coupling.

Output contract (written to memory as 'classification'):
  {
    "area": "trabalhista",
    "subarea": "rescisao_contratual",
    "urgencia": "media",
    "procedimento": "rito_ordinario",
    "complexidade": "medio",
    "partes": {
      "requerente": "João da Silva",
      "requerido": "Empresa ABC Ltda."
    },
    "fatos_principais": ["demissão sem justa causa", "8 anos de empresa"],
    "confidence": 0.92
  }

Urgency rules encoded in the prompt:
  URGENTE: prazo processual em < 5 dias, tutela de urgência, liminar
  MÉDIA: prazo em 6-30 dias, ou caso sem prazo mas com impacto econômico alto
  BAIXA: planejamento, consulta, casos sem prazo imediato
"""
from __future__ import annotations

from src.agents.base_agent import BaseAgent
from src.utils.workflow_models import WorkflowMemory

VALID_AREAS = {
    "trabalhista", "civil", "penal", "tributario",
    "administrativo", "consumidor", "previdenciario", "familiar", "outros"
}
VALID_URGENCIAS = {"urgente", "media", "baixa"}
VALID_COMPLEXIDADES = {"simples", "medio", "complexo"}
VALID_PROCEDIMENTOS = {
    "rito_sumario", "rito_sumarissimo", "rito_ordinario",
    "especial", "habeas_corpus", "mandado_seguranca", "a_definir"
}


class ClassifierAgent(BaseAgent):
    """Classifies legal case type, urgency, and procedural route."""

    @property
    def name(self) -> str:
        return "ClassifierAgent"

    @property
    def description(self) -> str:
        return "Classifica área, subárea, urgência e complexidade do caso jurídico."

    def _build_prompt(self, memory: WorkflowMemory) -> tuple[str, str]:
        system = """\
Você é um especialista em triagem jurídica brasileira. Analise o caso e classifique-o.

Retorne JSON com exatamente este schema:
{
  "area": "trabalhista|civil|penal|tributario|administrativo|consumidor|previdenciario|familiar|outros",
  "subarea": "string (ex: rescisao_contratual, dano_moral, horas_extras, cobranca, etc.)",
  "urgencia": "urgente|media|baixa",
  "procedimento": "rito_sumario|rito_sumarissimo|rito_ordinario|especial|habeas_corpus|mandado_seguranca|a_definir",
  "complexidade": "simples|medio|complexo",
  "partes": {
    "requerente": "nome da parte requerente",
    "requerido": "nome da parte requerida"
  },
  "fatos_principais": ["fato 1", "fato 2"],
  "prazo_dias": null,
  "observacoes": "string opcional com pontos de atenção",
  "confidence": 0.0
}

Regras de urgência:
- URGENTE: prazo < 5 dias, tutela/liminar, prisão, perigo de dano irreparável
- MÉDIA: prazo 6-30 dias, ou impacto econômico alto sem prazo imediato
- BAIXA: planejamento, consulta, caso sem urgência processual

Regras de complexidade:
- SIMPLES: um pedido, fatos simples, jurisprudência pacífica
- MÉDIO: múltiplos pedidos OU fatos com divergência jurisprudencial
- COMPLEXO: fatos muito específicos, pluralidade de partes, tese inédita
"""
        caso = memory.get("caso", "")
        cliente = memory.get("cliente", "")
        parte_contraria = memory.get("parte_contraria", "")
        informacoes_adicionais = memory.get("informacoes_adicionais", "")

        user = f"""CASO:
{caso}

CLIENTE: {cliente or 'não informado'}
PARTE CONTRÁRIA: {parte_contraria or 'não informada'}
INFORMAÇÕES ADICIONAIS: {informacoes_adicionais or 'nenhuma'}

Classifique este caso."""

        return system, user

    def _parse_output(self, raw: dict) -> tuple[dict, float]:
        confidence = float(raw.get("confidence", 0.7))
        confidence = max(0.0, min(1.0, confidence))

        # Validate and normalize
        area = raw.get("area", "outros")
        if area not in VALID_AREAS:
            area = "outros"
            confidence *= 0.7

        urgencia = raw.get("urgencia", "media")
        if urgencia not in VALID_URGENCIAS:
            urgencia = "media"

        complexidade = raw.get("complexidade", "medio")
        if complexidade not in VALID_COMPLEXIDADES:
            complexidade = "medio"

        procedimento = raw.get("procedimento", "a_definir")
        if procedimento not in VALID_PROCEDIMENTOS:
            procedimento = "a_definir"

        output = {
            "area": area,
            "subarea": raw.get("subarea", ""),
            "urgencia": urgencia,
            "procedimento": procedimento,
            "complexidade": complexidade,
            "partes": raw.get("partes", {}),
            "fatos_principais": raw.get("fatos_principais", []),
            "prazo_dias": raw.get("prazo_dias"),
            "observacoes": raw.get("observacoes", ""),
        }

        return output, confidence
