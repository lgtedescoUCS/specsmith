"""Field-level scoring of one extracted Spec against its Expected answer.

Exact string comparison does not work on LLM output, so every check is
either structural (constraint kinds, confidence, question count) or a
keyword match. Keyword matching is case-insensitive and anchored at the
start of a word, with any run of whitespace in an alias matching any run
in the text. It is deliberately simple and auditable: every hit records
the question or field that triggered it, so a false positive can be seen
in the report and fixed in the alias.

A failed extraction (no spec) gets the worst value on every metric,
including those where lower is better. Otherwise a prompt that fails more
often would look better on redundant questions and domain errors.
"""

import re
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from functools import lru_cache

from specsmith.domain.models import Confidence, ConstraintKind, Spec
from specsmith.evaluation.dataset import Expected, Keyword


@lru_cache(maxsize=1024)
def _pattern(alias: str) -> re.Pattern[str]:
    words = alias.split()
    if not words:
        raise ValueError("alias must not be blank")
    # (?<!\w) rather than \b: \b before a non-word first character such as
    # "+" or "±" would require a word character in front of it.
    body = r"\s+".join(re.escape(word) for word in words)
    return re.compile(rf"(?<!\w){body}", re.IGNORECASE)


def mentions(text: str, aliases: Iterable[str]) -> bool:
    return any(_pattern(alias).search(text) for alias in aliases)


def _alternatives(keyword: Keyword) -> list[str]:
    return [keyword] if isinstance(keyword, str) else list(keyword)


def keyword_label(keyword: Keyword) -> str:
    return "|".join(_alternatives(keyword))


def _first_question(questions: Sequence[str], aliases: Iterable[str]) -> str | None:
    aliases = list(aliases)
    return next((q for q in questions if mentions(q, aliases)), None)


def _text_fields(spec: Spec) -> list[tuple[str, str]]:
    """Every free-text field of a spec, with where it came from."""
    fields = [("objective", spec.objective)]
    for state_name in ("current_state", "target_state"):
        for key, value in getattr(spec, state_name).items():
            fields.append((f"{state_name} key {key!r}", key))
            fields.append((f"{state_name}.{key}", value))
    for index, constraint in enumerate(spec.constraints):
        fields.append((f"constraints[{index}]", constraint.value))
    for index, question in enumerate(spec.open_questions):
        fields.append((f"open_questions[{index}]", question))
    return fields


@dataclass(frozen=True)
class CaseScore:
    extracted: bool
    objective_matched: tuple[str, ...]
    objective_missing: tuple[str, ...]
    constraints_expected: frozenset[ConstraintKind]
    constraints_predicted: frozenset[ConstraintKind]
    open_questions: int
    min_open_questions: int
    confidence_expected: Confidence
    confidence_predicted: Confidence | None
    # topic -> the first open question that mentions it
    must_ask_covered: dict[str, str] = field(default_factory=dict)
    must_ask_missed: tuple[str, ...] = ()
    must_not_asked: dict[str, str] = field(default_factory=dict)
    must_not_total: int = 0
    # term -> where it was found, e.g. "open_questions[2]"
    forbidden_hits: dict[str, str] = field(default_factory=dict)

    @property
    def objective_coverage(self) -> float | None:
        total = len(self.objective_matched) + len(self.objective_missing)
        return len(self.objective_matched) / total if total else None

    @property
    def constraint_tp(self) -> int:
        return len(self.constraints_expected & self.constraints_predicted)

    @property
    def constraint_fp(self) -> int:
        return len(self.constraints_predicted - self.constraints_expected)

    @property
    def constraint_fn(self) -> int:
        return len(self.constraints_expected - self.constraints_predicted)

    @property
    def enough_open_questions(self) -> bool:
        return self.extracted and self.open_questions >= self.min_open_questions

    @property
    def confidence_correct(self) -> bool:
        return self.confidence_predicted == self.confidence_expected

    @property
    def redundant_questions(self) -> int:
        # Worst case on failure: every topic the text already answers counts
        # as asked.
        return len(self.must_not_asked) if self.extracted else self.must_not_total

    @property
    def domain_clean(self) -> bool:
        return self.extracted and not self.forbidden_hits


def score_case(expected: Expected, spec: Spec | None) -> CaseScore:
    """Score one case. `spec` is None when the extraction failed."""
    must_ask = list(expected.must_ask_topics)
    if spec is None:
        return CaseScore(
            extracted=False,
            objective_matched=(),
            objective_missing=tuple(
                keyword_label(k) for k in expected.objective_keywords
            ),
            constraints_expected=frozenset(expected.constraint_kinds),
            constraints_predicted=frozenset(),
            open_questions=0,
            min_open_questions=expected.min_open_questions,
            confidence_expected=expected.confidence,
            confidence_predicted=None,
            must_ask_missed=tuple(must_ask),
            must_not_total=len(expected.must_not_ask_topics),
        )

    matched, missing = [], []
    for keyword in expected.objective_keywords:
        hit = mentions(spec.objective, _alternatives(keyword))
        (matched if hit else missing).append(keyword_label(keyword))

    covered: dict[str, str] = {}
    for topic, aliases in expected.must_ask_topics.items():
        question = _first_question(spec.open_questions, aliases)
        if question is not None:
            covered[topic] = question

    asked: dict[str, str] = {}
    for topic, aliases in expected.must_not_ask_topics.items():
        question = _first_question(spec.open_questions, aliases)
        if question is not None:
            asked[topic] = question

    hits: dict[str, str] = {}
    text_fields = _text_fields(spec)
    for term in expected.forbidden_terms:
        where = next((loc for loc, text in text_fields if mentions(text, [term])), None)
        if where is not None:
            hits[term] = where

    return CaseScore(
        extracted=True,
        objective_matched=tuple(matched),
        objective_missing=tuple(missing),
        constraints_expected=frozenset(expected.constraint_kinds),
        constraints_predicted=frozenset(c.kind for c in spec.constraints),
        open_questions=len(spec.open_questions),
        min_open_questions=expected.min_open_questions,
        confidence_expected=expected.confidence,
        confidence_predicted=spec.confidence,
        must_ask_covered=covered,
        must_ask_missed=tuple(t for t in must_ask if t not in covered),
        must_not_asked=asked,
        must_not_total=len(expected.must_not_ask_topics),
        forbidden_hits=hits,
    )


def _ratio(numerator: int, denominator: int) -> float | None:
    return numerator / denominator if denominator else None


@dataclass(frozen=True)
class Summary:
    """Dataset-level metrics. None means the metric had nothing to measure
    (for instance no case lists any must_ask topic)."""

    cases: int
    success_rate: float
    # Mean per-case keyword coverage, over cases that list keywords.
    objective: float | None
    # Micro-averaged over all cases: counts are summed before dividing.
    constraints_precision: float | None
    constraints_recall: float | None
    constraints_f1: float | None
    open_questions_ok: float
    confidence_accuracy: float
    must_ask_recall: float | None
    # Lower is better.
    redundant_question_rate: float | None
    domain_clean_rate: float


def summarize(scores: Sequence[CaseScore]) -> Summary:
    if not scores:
        raise ValueError("cannot summarize zero cases")
    n = len(scores)
    coverages = [c for s in scores if (c := s.objective_coverage) is not None]
    tp = sum(s.constraint_tp for s in scores)
    fp = sum(s.constraint_fp for s in scores)
    fn = sum(s.constraint_fn for s in scores)
    ask_total = sum(len(s.must_ask_covered) + len(s.must_ask_missed) for s in scores)
    return Summary(
        cases=n,
        success_rate=sum(s.extracted for s in scores) / n,
        objective=sum(coverages) / len(coverages) if coverages else None,
        constraints_precision=_ratio(tp, tp + fp),
        constraints_recall=_ratio(tp, tp + fn),
        constraints_f1=_ratio(2 * tp, 2 * tp + fp + fn),
        open_questions_ok=sum(s.enough_open_questions for s in scores) / n,
        confidence_accuracy=sum(s.confidence_correct for s in scores) / n,
        must_ask_recall=_ratio(sum(len(s.must_ask_covered) for s in scores), ask_total),
        redundant_question_rate=_ratio(
            sum(s.redundant_questions for s in scores),
            sum(s.must_not_total for s in scores),
        ),
        domain_clean_rate=sum(s.domain_clean for s in scores) / n,
    )
