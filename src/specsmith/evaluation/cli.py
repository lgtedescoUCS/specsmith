"""Command line for the evaluation: `uv run python eval/run_eval.py`.

A composition root, like main.py: the only evaluation module that picks a
concrete provider. Real runs spend tokens; `--rescore` does not.
"""

import argparse
import json
import sys
from collections.abc import Callable, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from specsmith.config import Settings, get_settings
from specsmith.domain.errors import LLMNotConfiguredError
from specsmith.domain.pricing import estimate_cost
from specsmith.evaluation.dataset import DatasetError, EvalCase, load_dataset
from specsmith.evaluation.runner import (
    TABLE_HEADER,
    CaseRun,
    ReportMismatchError,
    UnannotatedCasesError,
    build_report,
    require_annotated,
    rescore,
    run_cases,
    table_row,
)
from specsmith.llm.anthropic import AnthropicProvider
from specsmith.llm.base import LLMProvider
from specsmith.llm.prompts import (
    InvalidPromptVersionError,
    PromptNotFoundError,
    load_prompt,
)
from specsmith.services.extraction import ExtractionService

# src/specsmith/evaluation/cli.py -> repository root -> eval/
EVAL_DIR = Path(__file__).resolve().parents[3] / "eval"

ProviderFactory = Callable[[Settings, str], LLMProvider]


def anthropic_provider(settings: Settings, model: str) -> LLMProvider:
    if settings.anthropic_api_key is None:
        raise LLMNotConfiguredError()
    return AnthropicProvider.create(
        api_key=settings.anthropic_api_key.get_secret_value(),
        model=model,
        timeout_seconds=settings.llm_timeout_seconds,
    )


def parse_args(argv: Sequence[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="run_eval.py",
        description="Run the extraction over the evaluation dataset and score it.",
    )
    parser.add_argument("--dataset", type=Path, default=EVAL_DIR / "dataset.jsonl")
    parser.add_argument("--output-dir", type=Path, default=EVAL_DIR / "results")
    parser.add_argument(
        "--prompt-version", help="default: PROMPT_VERSION from the settings"
    )
    parser.add_argument("--model", help="default: LLM_MODEL from the settings")
    parser.add_argument(
        "--cases", help="comma-separated case ids, e.g. 001,007 (default: all)"
    )
    parser.add_argument(
        "--rescore",
        type=Path,
        metavar="REPORT",
        help="score a stored report again against the current answer keys, "
        "without calling the model",
    )
    args = parser.parse_args(argv)
    if args.rescore and (args.prompt_version or args.model or args.cases):
        parser.error("--rescore takes the version, model and cases from the report")
    return args


def select_cases(cases: Sequence[EvalCase], ids: str | None) -> list[EvalCase]:
    if ids is None:
        return list(cases)
    wanted = {part.strip() for part in ids.split(",") if part.strip()}
    unknown = sorted(wanted - {case.id for case in cases})
    if unknown:
        raise DatasetError(f"unknown case ids: {', '.join(unknown)}")
    return [case for case in cases if case.id in wanted]


def write_report(report: dict[str, Any], output_dir: Path, name: str) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    for attempt in range(1, 100):
        suffix = "" if attempt == 1 else f"-{attempt}"
        path = output_dir / f"{name}{suffix}.json"
        try:
            # "x": never overwrite an earlier report.
            with path.open("x", encoding="utf-8", newline="\n") as file:
                json.dump(report, file, ensure_ascii=False, indent=2)
                file.write("\n")
            return path
        except FileExistsError:
            continue
    raise FileExistsError(f"could not find a free report name for {name}")


def _report_name(created_at: datetime, prompt_version: str, tag: str = "") -> str:
    return f"{created_at:%Y%m%dT%H%M%SZ}_{prompt_version}{tag}"


def _print_case(run: CaseRun) -> None:
    status = "ok" if run.spec is not None else f"FAILED {run.error_type}"
    cost = estimate_cost(run.meta)
    cost_text = "n/a" if cost is None else f"${cost:.4f}"
    print(
        f"{run.case_id}  {status}  attempts={run.meta.attempts}  "
        f"latency={run.meta.latency_ms}ms  cost={cost_text}",
        flush=True,
    )
    for rejected in run.rejected:
        lines = rejected.errors.splitlines() or [""]
        extra = f" (+{len(lines) - 1} more)" if len(lines) > 1 else ""
        print(f"     attempt {rejected.attempt} rejected: {lines[0]}{extra}", flush=True)


def main(
    argv: Sequence[str] | None = None,
    *,
    settings: Settings | None = None,
    make_provider: ProviderFactory = anthropic_provider,
) -> int:
    args = parse_args(argv)
    created_at = datetime.now(UTC)
    try:
        cases = load_dataset(args.dataset)
        if args.rescore:
            stored = json.loads(args.rescore.read_text(encoding="utf-8"))
            report = rescore(
                stored, cases, dataset_path=args.dataset, created_at=created_at
            )
            name = _report_name(created_at, report["prompt_version"], "_rescored")
        else:
            selected = select_cases(cases, args.cases)
            # Checked before any settings or provider, so an incomplete
            # answer key fails fast and without a key in the environment.
            require_annotated(selected)
            settings = settings or get_settings()
            model = args.model or settings.llm_model
            prompt = load_prompt(args.prompt_version or settings.prompt_version)
            service = ExtractionService(
                make_provider(settings, model),
                prompt,
                max_retries=settings.max_extraction_retries,
            )
            print(
                f"Running {len(selected)} case(s) with {model}, "
                f"prompt {prompt.version}",
                flush=True,
            )
            runs = run_cases(service, selected, on_case=_print_case)
            report = build_report(
                runs,
                cases,
                prompt_version=prompt.version,
                model=model,
                dataset_path=args.dataset,
                created_at=created_at,
            )
            name = _report_name(created_at, prompt.version)
    except (
        DatasetError,
        UnannotatedCasesError,
        ReportMismatchError,
        LLMNotConfiguredError,
        InvalidPromptVersionError,
        PromptNotFoundError,
        # A missing dataset or report file, or a report that is not JSON.
        OSError,
        json.JSONDecodeError,
    ) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    path = write_report(report, args.output_dir, name)
    print()
    print(TABLE_HEADER)
    print(table_row(report))
    print()
    print(f"Report: {path}")
    return 0
