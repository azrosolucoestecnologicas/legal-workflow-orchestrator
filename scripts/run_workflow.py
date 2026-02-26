"""
Script: Run a Legal Workflow

Usage:
    python scripts/run_workflow.py --workflow triagem_rapida --input data/inputs/caso.json
    python scripts/run_workflow.py --workflow peticao_inicial --input caso.json --interactive
    python scripts/run_workflow.py --workflow triagem_rapida --caso "Empregado demitido após 5 anos..."
"""
import argparse
import json
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.workflows.workflow_engine import WorkflowEngine
from src.workflows.definitions import (
    triagem_rapida_workflow,
    recurso_ordinario_workflow,
    peticao_inicial_workflow,
)
from src.utils.workflow_models import WorkflowStatus

logging.basicConfig(level=logging.INFO, format="%(levelname)s — %(message)s")


WORKFLOW_MAP = {
    "triagem_rapida": triagem_rapida_workflow,
    "recurso_ordinario": recurso_ordinario_workflow,
    "peticao_inicial": peticao_inicial_workflow,
}


def main():
    parser = argparse.ArgumentParser(description="Run a legal workflow")
    parser.add_argument("--workflow", required=True, choices=list(WORKFLOW_MAP.keys()))

    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument("--input", help="JSON file with workflow input")
    input_group.add_argument("--caso", help="Case description text (quick mode)")

    parser.add_argument("--interactive", action="store_true", help="Enable human gates")
    parser.add_argument("--output", help="Save result JSON to file")
    parser.add_argument("--trace-only", action="store_true", help="Print only execution trace")
    args = parser.parse_args()

    # Load input
    if args.input:
        with open(args.input) as f:
            initial_input = json.load(f)
    else:
        initial_input = {"caso": args.caso}

    # Build and register workflow
    engine = WorkflowEngine()
    workflow_factory = WORKFLOW_MAP[args.workflow]
    engine.register(workflow_factory())

    # Run
    print(f"\n🚀 Iniciando workflow: {args.workflow}")
    print(f"   Interativo: {args.interactive}")
    print()

    result = engine.run(args.workflow, initial_input, interactive=args.interactive)

    # Output
    print(f"\n{'='*60}")
    icon = "✅" if result.succeeded else "❌"
    print(f"{icon} Workflow {result.status.value.upper()}")
    print(f"   ID: {result.workflow_id}")
    print(f"   Duração total: {result.trace.total_duration_ms:.0f}ms")
    print(f"   LLM calls: {result.trace.total_llm_calls}")
    print(f"   Tokens: {result.trace.total_tokens_used}")
    print()

    if args.trace_only:
        print(result.trace.to_json())
        return

    # Print step summary
    print("STEPS:")
    for step in result.trace.steps:
        status_icon = {
            "completed": "✓", "failed": "✗", "skipped": "⊘",
            "waiting": "⏸", "cancelled": "✕"
        }.get(step.status.value, "?")
        print(f"  {status_icon} {step.step_id} ({step.agent_name}) "
              f"— conf={step.confidence:.0%}, {step.duration_ms:.0f}ms")
        if step.error:
            print(f"    ⚠ {step.error}")

    print()

    # Print key outputs
    if "classification" in result.final_output:
        c = result.final_output["classification"]
        print(f"CLASSIFICAÇÃO: {c.get('area','')} / {c.get('subarea','')} "
              f"(urgência: {c.get('urgencia','')})")

    if "analise" in result.final_output:
        a = result.final_output["analise"]
        prob = a.get("probabilidade_exito", 0)
        cat = a.get("categoria_exito", "")
        print(f"ANÁLISE: prob. êxito = {prob:.0%} ({cat})")

    if "revisao" in result.final_output:
        r = result.final_output["revisao"]
        print(f"REVISÃO: score={r.get('score_qualidade', 0):.0%}, "
              f"recomendação={r.get('recomendacao', '?')}")

    # Save output
    if args.output:
        output_data = {
            "workflow_id": result.workflow_id,
            "workflow_name": result.workflow_name,
            "status": result.status.value,
            "final_output": result.final_output,
            "trace": result.trace.to_dict(),
        }
        with open(args.output, "w", encoding="utf-8") as f:
            json.dump(output_data, f, ensure_ascii=False, indent=2)
        print(f"\n💾 Resultado salvo em: {args.output}")

    sys.exit(0 if result.succeeded else 1)


if __name__ == "__main__":
    main()
