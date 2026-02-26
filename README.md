# Legal Workflow Orchestrator

> **Automação de Fluxos Jurídicos com Orquestração de LLM**

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/licenses/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## Problema

Fluxos jurídicos típicos envolvem etapas sequenciais e condicionais:

1. **Triagem**: classificar tipo de demanda e urgência
2. **Pesquisa**: buscar jurisprudência e legislação relevante  
3. **Análise**: avaliar mérito e chances de êxito
4. **Minutagem**: redigir peça processual fundamentada
5. **Revisão**: verificar consistência, citações, completude
6. **Aprovação**: gate humano antes de protocolo

Fazer isso manualmente com LLMs é frágil: um prompt gigante não tem memória de etapas anteriores, não pode bifurcar baseado em condições, não tem controle de qualidade por etapa, e não tem auditoria do que cada LLM decidiu.

Este sistema implementa **orquestração multi-agente** com:
- Agentes especializados por função (classificador, pesquisador, analista, redator, revisor)
- Memória compartilhada entre agentes dentro de um workflow
- Execução condicional baseada em outputs de agentes anteriores
- Registro completo de trace (quem decidiu o quê, com qual confiança)
- Human-in-the-loop gates onde necessário
- Retentativas com prompts diferentes em caso de falha

## Arquitetura

```
                    ┌────────────────────────────────┐
                    │         Workflow Engine        │
                    │                                │
                    │  ┌──────────────────────────┐  │
                    │  │    WorkflowDefinition    │  │
                    │  │  (DAG de steps/condições)│  │
                    │  └──────────────┬───────────┘  │
                    │                 │               │
                    │  ┌──────────────▼───────────┐  │
                    │  │    WorkflowExecutor      │  │
                    │  │  Executa steps em ordem  │  │
                    │  │  Avalia condições        │  │
                    │  │  Registra trace          │  │
                    │  └──────────────┬───────────┘  │
                    │                 │               │
                    │  ┌──────────────▼───────────┐  │
                    │  │    WorkflowMemory        │  │
                    │  │  Estado compartilhado    │  │
                    │  │  entre todos os steps    │  │
                    └──└──────────────────────────┘──┘
                                     │
              ┌──────────────────────┼──────────────────────┐
              │                      │                      │
  ┌───────────▼──────┐  ┌────────────▼───────┐  ┌──────────▼────────┐
  │  ClassifierAgent │  │  ResearcherAgent   │  │   DrafterAgent    │
  │  Classifica tipo │  │  Busca precedentes │  │   Redige a peça   │
  │  e urgência      │  │  e legislação      │  │   processual      │
  └──────────────────┘  └────────────────────┘  └───────────────────┘
              │                      │                      │
  ┌───────────▼──────┐  ┌────────────▼───────┐  ┌──────────▼────────┐
  │  AnalystAgent    │  │   ReviewerAgent    │  │   HumanGateStep   │
  │  Avalia mérito   │  │   Verifica a peça  │  │   Aprovação human │
  │  e riscos        │  │   minutada         │  │   in the loop     │
  └──────────────────┘  └────────────────────┘  └───────────────────┘
```

## Workflows Implementados

| Workflow | Descrição | Agentes |
|---|---|---|
| `peticao_inicial` | Triagem → Pesquisa → Análise → Minutagem → Revisão | 5 agentes |
| `recurso_ordinario` | Classificação → Análise de sentença → Minutagem recurso | 4 agentes |
| `triagem_rapida` | Classificação → Análise de viabilidade → Resposta | 2 agentes |

## Trace e Auditoria

Cada execução gera um trace completo:

```json
{
  "workflow_id": "abc123",
  "workflow_name": "peticao_inicial",
  "started_at": "2024-11-15T10:30:00",
  "completed_at": "2024-11-15T10:35:22",
  "status": "completed",
  "steps": [
    {
      "step_id": "classify",
      "agent": "ClassifierAgent",
      "input_snapshot": {"caso": "..."},
      "output": {"tipo": "trabalhista", "urgencia": "media"},
      "confidence": 0.92,
      "duration_ms": 1230,
      "llm_calls": 1,
      "tokens_used": 450
    }
  ],
  "human_gates_encountered": 1,
  "human_gates_approved": 1
}
```

## Setup

```bash
git clone https://github.com/thiagoazeredorodrigues/legal-workflow-orchestrator
cd legal-workflow-orchestrator

python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

cp .env.example .env  # adicionar chaves de API

# Executar workflow de triagem rápida
python scripts/run_workflow.py \
    --workflow triagem_rapida \
    --input data/inputs/caso_exemplo.json

# Executar com modo interativo (human gates ativados)
python scripts/run_workflow.py \
    --workflow peticao_inicial \
    --input data/inputs/caso_exemplo.json \
    --interactive
```

## Uso Rápido

```python
from src.workflows.workflow_engine import WorkflowEngine
from src.workflows.definitions import peticao_inicial_workflow

engine = WorkflowEngine()
engine.register(peticao_inicial_workflow())

result = engine.run(
    workflow_name="peticao_inicial",
    initial_input={
        "caso": "Empregado demitido sem justa causa após 8 anos...",
        "cliente": "João da Silva",
        "parte_contraria": "Empresa ABC Ltda.",
    },
    interactive=False,  # True para human-in-the-loop
)

print(f"Status: {result.status}")
print(f"Peça minutada: {result.memory.get('minuta_petica')}")
print(f"Trace: {result.trace.to_json()}")
```

## Referências

- Yao et al. (2022) — ReAct: Synergizing Reasoning and Acting in Language Models. ICLR.
- OpenAI (2024) — Agentic AI Systems design patterns.

## Autor

**Thiago Azeredo Rodrigues**  
Especialista em IA para o Direito | LLM Orchestration | Legal Tech  
[LinkedIn](https://linkedin.com/in/thiagoazeredorodrigues) · [GitHub](https://github.com/thiagoazeredorodrigues)

## Licença

MIT License
# legal-workflow-orchestrator
