"""Field-level scoring of the evaluation. No database, no network.

The specs and answer keys below are synthetic and exist only to exercise
the scoring; they are not annotations of eval/dataset.jsonl.
"""

from typing import Any

import pytest

from specsmith.domain.models import ConstraintKind, Spec
from specsmith.evaluation.dataset import Expected
from specsmith.evaluation.scoring import (
    CaseScore,
    mentions,
    score_case,
    summarize,
)


def make_expected(**overrides: Any) -> Expected:
    payload: dict[str, Any] = {
        "objective_keywords": [],
        "constraint_kinds": [],
        "min_open_questions": 0,
        "confidence": "medium",
        "must_ask_topics": {},
        "must_not_ask_topics": {},
        "forbidden_terms": [],
    }
    payload.update(overrides)
    return Expected.model_validate(payload)


def make_spec(**overrides: Any) -> Spec:
    payload: dict[str, Any] = {
        "objective": "Increase tempering line throughput",
        "confidence": "medium",
    }
    payload.update(overrides)
    return Spec.model_validate(payload)


# --- matching ---------------------------------------------------------------


@pytest.mark.parametrize(
    ("text", "alias", "found"),
    [
        ("Increase Tempering throughput", "temper", True),
        ("Preheat the glass", "heat", False),
        ("Glass thickness range?", "THICKNESS", True),
        ("What is the\n  budget?", "what is the budget", True),
        ("What is the total budget?", "what is the budget", False),
        ("Tolerance of +0.5 mm", "+0.5", True),
        ("Spacer (warm-edge)?", "warm-edge", True),
        ("Low-e coating?", "low-e", True),
    ],
)
def test_mentions_matches_at_word_start(text: str, alias: str, found: bool) -> None:
    assert mentions(text, [alias]) is found


def test_mentions_is_true_when_any_alias_matches() -> None:
    assert mentions("How many shifts?", ["crew", "shift"])
    assert not mentions("How many shifts?", ["crew", "staff"])


# --- per-case fields ---------------------------------------------------------


def test_objective_coverage_counts_alternatives_once() -> None:
    expected = make_expected(
        objective_keywords=["temper", ["throughput", "capacity"], "laminat"]
    )

    score = score_case(expected, make_spec())

    assert score.objective_matched == ("temper", "throughput|capacity")
    assert score.objective_missing == ("laminat",)
    assert score.objective_coverage == pytest.approx(2 / 3)


def test_objective_coverage_is_none_without_keywords() -> None:
    assert score_case(make_expected(), make_spec()).objective_coverage is None


def test_constraint_kinds_are_compared_as_sets() -> None:
    expected = make_expected(constraint_kinds=["budget", "space"])
    spec = make_spec(
        constraints=[
            {"kind": "budget", "value": "USD 1"},
            {"kind": "technical", "value": "4 mm"},
            {"kind": "technical", "value": "6 mm"},
        ]
    )

    score = score_case(expected, spec)

    assert score.constraints_predicted == {
        ConstraintKind.BUDGET,
        ConstraintKind.TECHNICAL,
    }
    assert (score.constraint_tp, score.constraint_fp, score.constraint_fn) == (1, 1, 1)


@pytest.mark.parametrize(("count", "ok"), [(1, False), (2, True), (3, True)])
def test_open_question_minimum(count: int, ok: bool) -> None:
    spec = make_spec(open_questions=[f"Question {i}?" for i in range(count)])

    score = score_case(make_expected(min_open_questions=2), spec)

    assert score.enough_open_questions is ok


def test_confidence_must_match_exactly() -> None:
    expected = make_expected(confidence="low")

    assert score_case(expected, make_spec(confidence="low")).confidence_correct
    assert not score_case(expected, make_spec(confidence="medium")).confidence_correct


def test_must_ask_records_the_question_that_covers_each_topic() -> None:
    expected = make_expected(
        must_ask_topics={
            "thickness": ["thickness"],
            "shifts": ["shift"],
            "glass_type": ["low-e", "coated"],
        }
    )
    spec = make_spec(
        open_questions=[
            "How many shifts per day?",
            "What thickness range?",
            "Is the furnace kept?",
        ]
    )

    score = score_case(expected, spec)

    assert score.must_ask_covered == {
        "thickness": "What thickness range?",
        "shifts": "How many shifts per day?",
    }
    assert score.must_ask_missed == ("glass_type",)


def test_must_not_ask_flags_questions_the_text_already_answers() -> None:
    expected = make_expected(
        must_not_ask_topics={"budget": ["what is the budget", "budget range"]}
    )

    asking = score_case(
        expected, make_spec(open_questions=["What is the budget for this?"])
    )
    # A question that only mentions the stated budget is not redundant; a
    # phrase alias keeps it from matching, where a bare "budget" would not.
    mentioning = score_case(
        expected,
        make_spec(open_questions=["Does the USD 1.2M budget include installation?"]),
    )

    assert asking.must_not_asked == {"budget": "What is the budget for this?"}
    assert asking.redundant_questions == 1
    assert mentioning.must_not_asked == {}
    assert mentioning.redundant_questions == 0


def test_forbidden_terms_are_searched_in_every_text_field() -> None:
    expected = make_expected(
        forbidden_terms=["steel", "hardening", "metallurg", "quench", "absent"]
    )
    spec = make_spec(
        target_state={"hardening depth": "2 mm"},
        constraints=[{"kind": "technical", "value": "Metallurgical grade"}],
        open_questions=["Which steel grade?", "Keep the quench section?"],
    )

    score = score_case(expected, spec)

    assert score.forbidden_hits == {
        "steel": "open_questions[0]",
        "hardening": "target_state key 'hardening depth'",
        "metallurg": "constraints[0]",
        "quench": "open_questions[1]",
    }
    assert not score.domain_clean


def test_a_term_is_only_forbidden_where_the_case_lists_it() -> None:
    # "quench" is the cooling section of a glass tempering line; a case
    # about glass does not list it, so using it is not a domain error.
    spec = make_spec(open_questions=["Does the quench handle 4 mm?"])

    score = score_case(make_expected(forbidden_terms=["steel"]), spec)

    assert score.forbidden_hits == {}
    assert score.domain_clean


def test_failed_extraction_scores_worst_on_every_field() -> None:
    expected = make_expected(
        objective_keywords=["temper"],
        constraint_kinds=["budget"],
        min_open_questions=0,
        confidence="low",
        must_ask_topics={"shifts": ["shift"]},
        must_not_ask_topics={"budget": ["budget"], "space": ["space"]},
        forbidden_terms=["steel"],
    )

    score = score_case(expected, None)

    assert not score.extracted
    assert score.objective_coverage == 0
    assert score.constraint_fn == 1
    # Even a minimum of zero is not met by an extraction that failed.
    assert not score.enough_open_questions
    assert not score.confidence_correct
    assert score.must_ask_missed == ("shifts",)
    assert score.redundant_questions == 2
    assert not score.domain_clean


# --- summary -----------------------------------------------------------------


def test_summary_micro_averages_counts_across_cases() -> None:
    perfect = score_case(
        make_expected(
            objective_keywords=["temper"],
            constraint_kinds=["budget", "space", "deadline"],
            confidence="medium",
            must_ask_topics={"shifts": ["shift"]},
            must_not_ask_topics={"budget": ["budget"]},
        ),
        make_spec(
            constraints=[
                {"kind": "budget", "value": "a"},
                {"kind": "space", "value": "b"},
                {"kind": "deadline", "value": "c"},
            ],
            open_questions=["How many shifts?"],
        ),
    )
    poor = score_case(
        make_expected(
            objective_keywords=["temper", "laminat"],
            constraint_kinds=["budget"],
            min_open_questions=2,
            confidence="low",
            must_ask_topics={"thickness": ["thickness"], "size": ["size"]},
            must_not_ask_topics={"budget": ["budget"]},
            forbidden_terms=["steel"],
        ),
        make_spec(
            constraints=[{"kind": "technical", "value": "steel"}],
            open_questions=["What is the budget?"],
        ),
    )

    summary = summarize([perfect, poor])

    assert summary.cases == 2
    assert summary.success_rate == 1
    assert summary.objective == pytest.approx((1 + 0.5) / 2)
    # tp 3, fp 1, fn 1 summed over both cases.
    assert summary.constraints_precision == pytest.approx(3 / 4)
    assert summary.constraints_recall == pytest.approx(3 / 4)
    assert summary.constraints_f1 == pytest.approx(6 / 8)
    assert summary.open_questions_ok == 0.5
    assert summary.confidence_accuracy == 0.5
    assert summary.must_ask_recall == pytest.approx(1 / 3)
    assert summary.redundant_question_rate == 0.5
    assert summary.domain_clean_rate == 0.5


def test_summary_counts_failures_against_every_rate() -> None:
    ok = score_case(make_expected(), make_spec())
    failed = score_case(make_expected(), None)

    summary = summarize([ok, failed])

    assert summary.success_rate == 0.5
    assert summary.confidence_accuracy == 0.5
    assert summary.domain_clean_rate == 0.5


def test_summary_metrics_without_anything_to_measure_are_none() -> None:
    summary = summarize([score_case(make_expected(), make_spec())])

    assert summary.objective is None
    assert summary.constraints_precision is None
    assert summary.constraints_recall is None
    assert summary.constraints_f1 is None
    assert summary.must_ask_recall is None
    assert summary.redundant_question_rate is None


def test_summary_of_no_cases_is_an_error() -> None:
    scores: list[CaseScore] = []

    with pytest.raises(ValueError):
        summarize(scores)
