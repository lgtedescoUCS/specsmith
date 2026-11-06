"""Runs the extraction over evaluation cases and builds the report.

Calls the ExtractionService directly, without the database: an eval run
must not leave specs in the dev database, and the service needs none.
Depends on the LLMProvider protocol only; the CLI picks the provider.

A report keeps each case's raw output next to its score, so that
`rescore` can score it again after the answer key changes, without
calling the model.
"""

import hashlib
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

from specsmith.domain.errors import ExtractionError
from specsmith.domain.models import ExtractionMeta, Spec
from specsmith.evaluation.dataset import EvalCase
from specsmith.evaluation.scoring import CaseScore, Summary, score_case, summarize
from specsmith.services.extraction import ExtractionService

REPORT_FORMAT = 1

# USD per million tokens (input, output), standard Messages API requests.
# Checked on 2026-11-06 at
# https://platform.claude.com/docs/en/about-claude/pricing
# A model missing here gets no cost estimate rather than a wrong one.
PRICES_PER_MTOK: dict[str, tuple[float, float]] = {
    "claude-sonnet-5-5": (2.0, 10.0),
    "claude-opus-5-5": (4.0, 20.0),
}


class UnannotatedCasesError(ValueError):
    """Real runs need every selected case annotated.

    The answer key is written before any model output is seen, so that it
    is not anchored on what the baseline happened to produce.
    """

    def __init__(self, case_ids: Sequence[str]) -> None:
        self.case_ids = list(case_ids)
        super().__init__(
            "cases without an answer key (expected is {}): "
            + ", ".join(self.case_ids)
        )


class ReportMismatchError(ValueError):
    """A stored report cannot be rescored against the current dataset."""


def require_annotated(cases: Sequence[EvalCase]) -> None:
    missing = [case.id for case in cases if not case.annotated]
    if missing:
        raise UnannotatedCasesError(missing)


def estimate_cost(meta: ExtractionMeta) -> float | None:
    prices = PRICES_PER_MTOK.get(meta.model)
    if prices is None:
        return None
    input_price, output_price = prices
    return (meta.input_tokens * input_price + meta.output_tokens * output_price) / 1e6


@dataclass(frozen=True)
class CaseRun:
    case_id: str
    raw_text: str
    spec: Spec | None
    meta: ExtractionMeta
    error_type: str | None = None
    error: str | None = None


def run_cases(
    service: ExtractionService,
    cases: Sequence[EvalCase],
    *,
    on_case: Callable[[CaseRun], None] | None = None,
) -> list[CaseRun]:
    """One extraction per case. A failed extraction is recorded and the run
    goes on; any other exception is a bug and stops the run."""
    require_annotated(cases)
    runs: list[CaseRun] = []
    for case in cases:
        try:
            result = service.extract(case.raw_text)
            run = CaseRun(case.id, case.raw_text, result.spec, result.meta)
        except ExtractionError as exc:
            run = CaseRun(
                case.id,
                case.raw_text,
                None,
                exc.meta,
                error_type=type(exc).__name__,
                error=exc.reason,
            )
        runs.append(run)
        if on_case is not None:
            on_case(run)
    return runs


def score_to_dict(score: CaseScore) -> dict[str, Any]:
    return {
        "extracted": score.extracted,
        "objective_coverage": score.objective_coverage,
        "objective_matched": list(score.objective_matched),
        "objective_missing": list(score.objective_missing),
        "constraints_expected": sorted(score.constraints_expected),
        "constraints_predicted": sorted(score.constraints_predicted),
        "constraint_tp": score.constraint_tp,
        "constraint_fp": score.constraint_fp,
        "constraint_fn": score.constraint_fn,
        "open_questions": score.open_questions,
        "min_open_questions": score.min_open_questions,
        "enough_open_questions": score.enough_open_questions,
        "confidence_expected": score.confidence_expected,
        "confidence_predicted": score.confidence_predicted,
        "confidence_correct": score.confidence_correct,
        "must_ask_covered": score.must_ask_covered,
        "must_ask_missed": list(score.must_ask_missed),
        "must_not_asked": score.must_not_asked,
        "must_not_total": score.must_not_total,
        "redundant_questions": score.redundant_questions,
        "forbidden_hits": score.forbidden_hits,
        "domain_clean": score.domain_clean,
    }


def _summary_to_dict(summary: Summary) -> dict[str, Any]:
    return {name: getattr(summary, name) for name in Summary.__dataclass_fields__}


def _mean(values: Sequence[float]) -> float | None:
    return sum(values) / len(values) if values else None


def dataset_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def build_report(
    runs: Sequence[CaseRun],
    cases: Sequence[EvalCase],
    *,
    prompt_version: str,
    model: str,
    dataset_path: Path,
    created_at: datetime,
) -> dict[str, Any]:
    by_id = {case.id: case for case in cases}
    entries: list[dict[str, Any]] = []
    scores: list[CaseScore] = []
    costs: list[float] = []
    for run in runs:
        expected = by_id[run.case_id].expected
        if expected is None:
            raise UnannotatedCasesError([run.case_id])
        score = score_case(expected, run.spec)
        cost = estimate_cost(run.meta)
        scores.append(score)
        if cost is not None:
            costs.append(cost)
        entries.append(
            {
                "id": run.case_id,
                "raw_text": run.raw_text,
                "spec": None if run.spec is None else run.spec.model_dump(mode="json"),
                "error_type": run.error_type,
                "error": run.error,
                "meta": run.meta.model_dump(mode="json"),
                "cost_usd": cost,
                "score": score_to_dict(score),
            }
        )
    metas = [run.meta for run in runs]
    return {
        "format": REPORT_FORMAT,
        "created_at": created_at.isoformat(),
        "prompt_version": prompt_version,
        "model": model,
        "dataset": {
            "path": dataset_path.as_posix(),
            "sha256": dataset_sha256(dataset_path),
        },
        "summary": _summary_to_dict(summarize(scores)),
        "usage": {
            "input_tokens": sum(m.input_tokens for m in metas),
            "output_tokens": sum(m.output_tokens for m in metas),
            "attempts": sum(m.attempts for m in metas),
            "avg_latency_ms": _mean([m.latency_ms for m in metas]),
            # Over the cases whose model has a known price.
            "avg_cost_usd": _mean(costs),
            "priced_cases": len(costs),
        },
        "cases": entries,
    }


def rescore(
    report: dict[str, Any],
    cases: Sequence[EvalCase],
    *,
    dataset_path: Path,
    created_at: datetime,
) -> dict[str, Any]:
    """Score a stored report again against the current answer keys.

    Refuses a case whose text changed since the run: its output answers a
    different requirement.
    """
    if report.get("format") != REPORT_FORMAT:
        raise ReportMismatchError(
            f"report format {report.get('format')!r}, expected {REPORT_FORMAT}"
        )
    by_id = {case.id: case for case in cases}
    runs: list[CaseRun] = []
    for entry in report["cases"]:
        case = by_id.get(entry["id"])
        if case is None:
            raise ReportMismatchError(f"case {entry['id']} is not in the dataset")
        if case.raw_text != entry["raw_text"]:
            raise ReportMismatchError(
                f"case {case.id}: raw_text changed since the run; run it again"
            )
        spec = entry["spec"]
        runs.append(
            CaseRun(
                case_id=case.id,
                raw_text=case.raw_text,
                spec=None if spec is None else Spec.model_validate(spec),
                meta=ExtractionMeta.model_validate(entry["meta"]),
                error_type=entry["error_type"],
                error=entry["error"],
            )
        )
    require_annotated([by_id[run.case_id] for run in runs])
    rescored = build_report(
        runs,
        cases,
        prompt_version=report["prompt_version"],
        model=report["model"],
        dataset_path=dataset_path,
        created_at=created_at,
    )
    rescored["rescored_from"] = report["created_at"]
    return rescored


TABLE_HEADER = (
    "| Prompt | Objective | Constraints (F1) | Confidence | Must-ask recall "
    "| Redundant questions | Domain clean | Success | Avg cost |\n"
    "|---|---|---|---|---|---|---|---|---|"
)


def _percent(value: float | None) -> str:
    return "n/a" if value is None else f"{value:.0%}"


def table_row(report: dict[str, Any]) -> str:
    """One markdown row for the README table, straight from a report."""
    s = report["summary"]
    f1 = s["constraints_f1"]
    cost = report["usage"]["avg_cost_usd"]
    cells = [
        report["prompt_version"],
        _percent(s["objective"]),
        "n/a" if f1 is None else f"{f1:.2f}",
        _percent(s["confidence_accuracy"]),
        _percent(s["must_ask_recall"]),
        _percent(s["redundant_question_rate"]),
        _percent(s["domain_clean_rate"]),
        _percent(s["success_rate"]),
        "n/a" if cost is None else f"${cost:.4f}",
    ]
    return "| " + " | ".join(cells) + " |"
