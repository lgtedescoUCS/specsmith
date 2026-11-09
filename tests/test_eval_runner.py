"""Evaluation runner and CLI, driven by a FakeProvider. No database, no
network, no tokens.

The cases and answer keys below are synthetic; they are not annotations of
eval/dataset.jsonl.
"""

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pytest

from specsmith.config import Settings
from specsmith.evaluation.cli import main, write_report
from specsmith.evaluation.dataset import EvalCase, load_dataset
from specsmith.evaluation.runner import (
    TABLE_HEADER,
    ReportMismatchError,
    UnannotatedCasesError,
    build_report,
    rescore,
    run_cases,
    table_row,
)
from specsmith.llm.base import LLMProvider, LLMResponse, LLMTimeoutError
from specsmith.llm.fake import FakeProvider
from specsmith.llm.prompts import load_prompt
from specsmith.services.extraction import ExtractionService

NOW = datetime(2026, 11, 6, 12, 0, tzinfo=UTC)


def expected(**overrides: Any) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "objective_keywords": ["widget"],
        "constraint_kinds": ["budget"],
        "min_open_questions": 1,
        "confidence": "medium",
        "must_ask_topics": {"colour": ["colour"]},
        "must_not_ask_topics": {"budget": ["what is the budget"]},
        "forbidden_terms": ["gizmo"],
    }
    payload.update(overrides)
    return payload


def case_line(case_id: str, text: str, answer: dict[str, Any]) -> str:
    return json.dumps(
        {
            "id": case_id,
            "lang": "en",
            "tags": ["synthetic"],
            "raw_text": text,
            "expected": answer,
        }
    )


def draft(**overrides: Any) -> str:
    payload: dict[str, Any] = {
        "objective": "Make widgets faster",
        "current_state": [],
        "target_state": [],
        "constraints": [{"kind": "budget", "value": "USD 10"}],
        "open_questions": ["Which colour?"],
        "confidence": "medium",
    }
    payload.update(overrides)
    return json.dumps(payload)


@pytest.fixture
def dataset(tmp_path: Path) -> Path:
    path = tmp_path / "dataset.jsonl"
    lines = [
        case_line("001", "Make widgets faster. Budget USD 10.", expected()),
        case_line("002", "Widgets again.", expected(confidence="low")),
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8", newline="\n")
    return path


def service(provider: LLMProvider, max_retries: int = 0) -> ExtractionService:
    return ExtractionService(provider, load_prompt("v1"), max_retries=max_retries)


def test_run_records_failures_and_goes_on(dataset: Path) -> None:
    cases = load_dataset(dataset)
    fake = FakeProvider(["not json", LLMTimeoutError("slow")])

    runs = run_cases(service(fake), cases)

    assert [run.error_type for run in runs] == [
        "ExtractionFailedError",
        "ExtractionTimeoutError",
    ]
    assert all(run.spec is None for run in runs)
    assert runs[0].meta.attempts == 1
    assert "validation" in (runs[0].error or "")


def test_run_refuses_unannotated_cases_before_any_call(tmp_path: Path) -> None:
    path = tmp_path / "dataset.jsonl"
    path.write_text(
        case_line("001", "a", expected()) + "\n" + case_line("002", "b", {}) + "\n",
        encoding="utf-8",
    )
    fake = FakeProvider([])

    with pytest.raises(UnannotatedCasesError, match="002"):
        run_cases(service(fake), load_dataset(path))
    assert fake.calls == []


def test_report_scores_each_case_and_sums_usage(dataset: Path) -> None:
    cases = load_dataset(dataset)
    fake = FakeProvider(
        [
            LLMResponse(draft(), "claude-sonnet-5-5", 1000, 100),
            draft(open_questions=["What is the budget?"], confidence="medium"),
        ]
    )
    runs = run_cases(service(fake), cases)

    report = build_report(
        runs,
        cases,
        prompt_version="v1",
        model="claude-sonnet-5-5",
        dataset_path=dataset,
        created_at=NOW,
    )

    first, second = report["cases"]
    assert first["score"]["must_ask_covered"] == {"colour": "Which colour?"}
    assert first["cost_usd"] == pytest.approx((1000 * 2 + 100 * 10) / 1e6)
    assert second["score"]["must_not_asked"] == {"budget": "What is the budget?"}
    # The fake's own model has no price, so it gets no estimate.
    assert second["cost_usd"] is None
    assert report["summary"]["confidence_accuracy"] == 0.5
    assert report["summary"]["redundant_question_rate"] == 0.5
    assert report["usage"]["input_tokens"] == 1100
    assert report["usage"]["priced_cases"] == 1
    assert report["dataset"]["sha256"]
    # Everything in the report must survive a JSON round trip.
    assert json.loads(json.dumps(report)) == report


def test_report_keeps_rejected_attempts_and_rescore_carries_them(
    dataset: Path,
) -> None:
    cases = load_dataset(dataset)
    too_many = draft(open_questions=[f"Q{i}?" for i in range(11)])
    fake = FakeProvider([too_many, draft(), "a", "b"])
    runs = run_cases(service(fake, max_retries=1), cases)
    report = build_report(
        runs,
        cases,
        prompt_version="v1",
        model="fake-model",
        dataset_path=dataset,
        created_at=NOW,
    )
    stored = json.loads(json.dumps(report))

    first, second = stored["cases"]
    assert [r["attempt"] for r in first["rejected_attempts"]] == [1]
    assert first["rejected_attempts"][0]["content"] == too_many
    assert "open_questions" in first["rejected_attempts"][0]["errors"]
    # A case that failed keeps every rejected answer, the last one included.
    assert [r["content"] for r in second["rejected_attempts"]] == ["a", "b"]

    again = rescore(stored, cases, dataset_path=dataset, created_at=NOW)
    assert again["cases"][0]["rejected_attempts"] == first["rejected_attempts"]


def test_rescore_reads_format_1_reports_without_rejected_attempts(
    dataset: Path,
) -> None:
    report = make_report(dataset)
    report["format"] = 1
    for entry in report["cases"]:
        del entry["rejected_attempts"]

    again = rescore(report, load_dataset(dataset), dataset_path=dataset, created_at=NOW)

    # Format 1 never stored them, so the rescored report has none either.
    assert [c["rejected_attempts"] for c in again["cases"]] == [[], []]


def make_report(dataset: Path) -> dict[str, Any]:
    cases = load_dataset(dataset)
    runs = run_cases(service(FakeProvider([draft(), "not json"])), cases)
    report = build_report(
        runs,
        cases,
        prompt_version="v1",
        model="fake-model",
        dataset_path=dataset,
        created_at=NOW,
    )
    return json.loads(json.dumps(report))


def rewrite(dataset: Path, cases: list[EvalCase]) -> None:
    lines = [case.model_dump_json() for case in cases]
    dataset.write_text("\n".join(lines) + "\n", encoding="utf-8", newline="\n")


def test_rescore_without_changes_reproduces_the_scores(dataset: Path) -> None:
    report = make_report(dataset)

    again = rescore(report, load_dataset(dataset), dataset_path=dataset, created_at=NOW)

    assert again["summary"] == report["summary"]
    assert [c["score"] for c in again["cases"]] == [
        c["score"] for c in report["cases"]
    ]
    assert again["rescored_from"] == report["created_at"]


def test_rescore_uses_the_current_answer_keys(dataset: Path) -> None:
    report = make_report(dataset)
    cases = load_dataset(dataset)
    assert cases[0].expected is not None
    cases[0].expected.must_ask_topics["size"] = ["size"]
    rewrite(dataset, cases)

    again = rescore(report, load_dataset(dataset), dataset_path=dataset, created_at=NOW)

    assert again["cases"][0]["score"]["must_ask_missed"] == ["size"]
    # The failed case is still failed: rescoring never calls the model.
    assert again["cases"][1]["spec"] is None


def test_rescore_refuses_a_case_whose_text_changed(dataset: Path) -> None:
    report = make_report(dataset)
    cases = load_dataset(dataset)
    cases[1].raw_text = "A different requirement."
    rewrite(dataset, cases)

    with pytest.raises(ReportMismatchError, match="002.*raw_text changed"):
        rescore(report, load_dataset(dataset), dataset_path=dataset, created_at=NOW)


def test_rescore_refuses_a_case_missing_from_the_dataset(dataset: Path) -> None:
    report = make_report(dataset)
    rewrite(dataset, load_dataset(dataset)[:1])

    with pytest.raises(ReportMismatchError, match="002 is not in the dataset"):
        rescore(report, load_dataset(dataset), dataset_path=dataset, created_at=NOW)


def test_rescore_refuses_another_report_format(dataset: Path) -> None:
    report = make_report(dataset) | {"format": 99}

    with pytest.raises(ReportMismatchError, match="format"):
        rescore(report, load_dataset(dataset), dataset_path=dataset, created_at=NOW)


def test_table_row_formats_the_summary(dataset: Path) -> None:
    report = make_report(dataset)
    report["usage"]["avg_cost_usd"] = 0.00912

    row = table_row(report)

    assert row.startswith("| v1 | ")
    assert row.endswith(" | 50% | $0.0091 |")
    assert row.count("|") == TABLE_HEADER.splitlines()[0].count("|")


def test_write_report_never_overwrites(tmp_path: Path) -> None:
    first = write_report({"a": 1}, tmp_path, "run")
    second = write_report({"a": 2}, tmp_path, "run")

    assert (first.name, second.name) == ("run.json", "run-2.json")
    assert json.loads(first.read_text(encoding="utf-8")) == {"a": 1}


# --- CLI -----------------------------------------------------------------------


def settings() -> Settings:
    return Settings(  # type: ignore[call-arg]
        _env_file=None,
        database_url="postgresql+psycopg://unused@127.0.0.1:1/unused",
        max_extraction_retries=0,
    )


def run_cli(
    args: list[str], fake: FakeProvider | None = None
) -> tuple[int, list[str]]:
    requested: list[str] = []

    def make_provider(_settings: Settings, model: str) -> LLMProvider:
        requested.append(model)
        assert fake is not None, "the CLI asked for a provider it should not need"
        return fake

    code = main(args, settings=settings(), make_provider=make_provider)
    return code, requested


def test_cli_runs_scores_and_writes_a_report(
    dataset: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    out_dir = tmp_path / "results"
    fake = FakeProvider([draft()])

    code, requested = run_cli(
        [
            "--dataset", str(dataset),
            "--output-dir", str(out_dir),
            "--prompt-version", "v1",
            "--model", "claude-sonnet-5-5",
            "--cases", "001",
        ],
        fake,
    )

    assert code == 0
    assert requested == ["claude-sonnet-5-5"]
    (path,) = out_dir.glob("*_v1.json")
    report = json.loads(path.read_text(encoding="utf-8"))
    assert [c["id"] for c in report["cases"]] == ["001"]
    output = capsys.readouterr().out
    assert TABLE_HEADER in output
    assert table_row(report) in output


def test_cli_refuses_unannotated_cases_without_a_provider(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    path = tmp_path / "dataset.jsonl"
    path.write_text(case_line("001", "a", {}) + "\n", encoding="utf-8")

    code, requested = run_cli(
        ["--dataset", str(path), "--output-dir", str(tmp_path / "r")]
    )

    assert code == 2
    assert requested == []
    assert "001" in capsys.readouterr().err
    assert not (tmp_path / "r").exists()


def test_cli_rejects_unknown_case_ids(
    dataset: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    code, requested = run_cli(
        ["--dataset", str(dataset), "--output-dir", str(tmp_path), "--cases", "999"]
    )

    assert code == 2
    assert requested == []
    assert "999" in capsys.readouterr().err


def test_cli_rejects_an_invalid_prompt_version(
    dataset: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    code, _ = run_cli(
        [
            "--dataset", str(dataset),
            "--output-dir", str(tmp_path),
            "--prompt-version", "v99",
            "--model", "m",
        ],
        FakeProvider([]),
    )

    assert code == 2
    assert "v99" in capsys.readouterr().err


def test_cli_reports_a_missing_dataset_without_a_traceback(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    code, requested = run_cli(
        ["--dataset", str(tmp_path / "absent.jsonl"), "--output-dir", str(tmp_path)]
    )

    assert code == 2
    assert requested == []
    assert "absent.jsonl" in capsys.readouterr().err


def test_cli_rescore_needs_no_provider(dataset: Path, tmp_path: Path) -> None:
    stored = write_report(make_report(dataset), tmp_path, "stored")

    code, requested = run_cli(
        [
            "--dataset", str(dataset),
            "--output-dir", str(tmp_path),
            "--rescore", str(stored),
        ]
    )

    assert code == 0
    assert requested == []
    (path,) = tmp_path.glob("*_v1_rescored.json")
    assert json.loads(path.read_text(encoding="utf-8"))["rescored_from"]


def test_cli_rescore_rejects_run_options(dataset: Path, tmp_path: Path) -> None:
    with pytest.raises(SystemExit) as exc_info:
        run_cli(["--rescore", str(tmp_path / "r.json"), "--cases", "001"])

    assert exc_info.value.code == 2
