"""Read-only endpoints: a spec's run, the spec listing and /stats.

Against a real PostgreSQL (skipped when it is not running). Specs are
created through POST /specs with the FakeProvider, or stored ready-made
through the repository for specs that have no run.
"""

import json
from datetime import timedelta
from typing import Any
from uuid import UUID, uuid4

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import Connection, func, update
from sqlalchemy.orm import Session

from specsmith.db import repository
from specsmith.db.models import ExtractionRunRow, SpecRow
from specsmith.domain.models import Spec
from specsmith.llm.fake import FakeProvider
from specsmith.main import app, get_daily_extraction_limit

RAW_TEXT = "Client wants 650 plates/day on the tempering line, from 400 today."


def answer(confidence: str = "medium", kinds: tuple[str, ...] = ("space",)) -> str:
    return json.dumps(
        {
            "objective": "Increase tempering line throughput",
            "current_state": [{"name": "throughput", "value": "400 plates/day"}],
            "target_state": [{"name": "throughput", "value": "650 plates/day"}],
            "constraints": [{"kind": k, "value": f"a {k} limit"} for k in kinds],
            "open_questions": ["How many shifts per day?"],
            "confidence": confidence,
        }
    )


def create(client: TestClient, fake_llm: FakeProvider, *answers: str) -> Any:
    fake_llm.queue(*(answers or (answer(),)))
    response = client.post("/specs", json={"raw_text": RAW_TEXT})
    assert response.status_code == 201, response.text
    return response.json()


def store_ready_spec(connection: Connection) -> UUID:
    """A spec with no extraction run, as Phase 1 stored them."""
    with Session(bind=connection, join_transaction_mode="create_savepoint") as s:
        spec = Spec.model_validate({"objective": "Ready-made", "confidence": "high"})
        stored = repository.save_spec(s, raw_text="x", source=None, spec=spec)
        s.commit()
    return stored.id


# --- GET /specs/{id}/run ------------------------------------------------------


def test_run_endpoint_returns_how_the_spec_was_extracted(
    client: TestClient, fake_llm: FakeProvider
) -> None:
    spec = create(client, fake_llm, "not json", answer())

    response = client.get(f"/specs/{spec['id']}/run")

    assert response.status_code == 200
    run = response.json()
    UUID(run["id"])
    assert run["spec_id"] == spec["id"]
    assert run["requirement_id"] == spec["requirement_id"]
    assert (run["model"], run["prompt_version"]) == ("fake-model", "v1")
    # Summed over both attempts (the first answer failed validation).
    assert (run["attempts"], run["input_tokens"], run["output_tokens"]) == (2, 200, 100)
    assert run["succeeded"] is True
    assert run["latency_ms"] >= 0
    assert run["created_at"]
    # The fake model has no price.
    assert run["estimated_cost_usd"] is None


def test_run_cost_is_estimated_from_list_prices(
    client: TestClient, fake_llm: FakeProvider, connection: Connection
) -> None:
    spec = create(client, fake_llm)
    connection.execute(
        update(ExtractionRunRow).values(
            model="claude-sonnet-5-5", input_tokens=1378, output_tokens=653
        )
    )

    run = client.get(f"/specs/{spec['id']}/run").json()

    assert run["estimated_cost_usd"] == pytest.approx((1378 * 2 + 653 * 10) / 1e6)


def test_run_of_an_unknown_spec_is_404(client: TestClient) -> None:
    missing = uuid4()

    response = client.get(f"/specs/{missing}/run")

    assert response.status_code == 404
    assert response.json() == {"detail": f"Spec {missing} not found"}


def test_spec_stored_without_a_run_is_404_with_its_own_message(
    client: TestClient, connection: Connection
) -> None:
    spec_id = store_ready_spec(connection)

    response = client.get(f"/specs/{spec_id}/run")

    assert response.status_code == 404
    assert response.json() == {"detail": f"Spec {spec_id} has no extraction run"}


# --- GET /specs ---------------------------------------------------------------


def ids(response: Any) -> list[str]:
    return [item["id"] for item in response.json()["items"]]


def oldest_first(connection: Connection, *specs: Any) -> list[str]:
    """Spreads created_at so `specs` are in age order, oldest first.

    Every test runs inside one outer transaction, where now() never moves,
    so all rows would otherwise share created_at."""
    for minutes_ago, spec in zip(range(len(specs), 0, -1), specs, strict=True):
        connection.execute(
            update(SpecRow)
            .where(SpecRow.id == UUID(spec["id"]))
            .values(created_at=func.now() - timedelta(minutes=minutes_ago))
        )
    return [spec["id"] for spec in specs]


def without_created_at(spec: dict[str, Any]) -> dict[str, Any]:
    return {k: v for k, v in spec.items() if k != "created_at"}


def test_list_is_empty_without_specs(client: TestClient) -> None:
    response = client.get("/specs")

    assert response.status_code == 200
    assert response.json() == {"items": [], "total": 0, "limit": 20, "offset": 0}


def test_list_returns_full_specs_newest_first(
    client: TestClient, fake_llm: FakeProvider, connection: Connection
) -> None:
    created = [create(client, fake_llm) for _ in range(3)]
    order = oldest_first(connection, *created)

    response = client.get("/specs")

    assert ids(response) == order[::-1]
    newest = response.json()["items"][0]
    assert without_created_at(newest) == without_created_at(created[-1])
    assert response.json()["total"] == 3


def test_pages_cover_every_spec_once(
    client: TestClient, fake_llm: FakeProvider
) -> None:
    created = {create(client, fake_llm)["id"] for _ in range(5)}

    pages = [client.get(f"/specs?limit=2&offset={o}") for o in (0, 2, 4)]

    seen = [spec_id for page in pages for spec_id in ids(page)]
    assert sorted(seen) == sorted(created)
    assert [len(ids(page)) for page in pages] == [2, 2, 1]
    assert {page.json()["total"] for page in pages} == {5}
    assert (pages[1].json()["limit"], pages[1].json()["offset"]) == (2, 2)


def test_ties_in_created_at_are_ordered_by_id(
    client: TestClient, connection: Connection
) -> None:
    # Specs stored in one transaction share created_at, as now() is the
    # transaction's start time.
    with Session(bind=connection, join_transaction_mode="create_savepoint") as s:
        spec = Spec.model_validate({"objective": "Same time", "confidence": "low"})
        stored = [
            repository.save_spec(s, raw_text="x", source=None, spec=spec).id
            for _ in range(3)
        ]
        s.commit()

    first = ids(client.get("/specs?limit=2"))
    second = ids(client.get("/specs?limit=2&offset=2"))

    assert first + second == sorted((str(i) for i in stored), reverse=True)


def test_list_filters_by_confidence(
    client: TestClient, fake_llm: FakeProvider, connection: Connection
) -> None:
    low = create(client, fake_llm, answer("low"))
    high = create(client, fake_llm, answer("high"))
    other_low = create(client, fake_llm, answer("low"))
    oldest_first(connection, low, high, other_low)

    response = client.get("/specs?confidence=low")

    assert ids(response) == [other_low["id"], low["id"]]
    assert response.json()["total"] == 2


@pytest.mark.parametrize(
    "query",
    ["confidence=certain", "limit=0", "limit=101", "offset=-1", "limit=abc"],
)
def test_invalid_list_parameters_are_422(client: TestClient, query: str) -> None:
    assert client.get(f"/specs?{query}").status_code == 422


def test_offset_past_the_end_is_an_empty_page(
    client: TestClient, fake_llm: FakeProvider
) -> None:
    create(client, fake_llm)

    body = client.get("/specs?offset=10").json()

    assert (body["items"], body["total"]) == ([], 1)


# --- GET /stats ---------------------------------------------------------------

ZERO_CONFIDENCE = {"high": 0, "medium": 0, "low": 0}
ZERO_KINDS = {
    "budget": 0,
    "deadline": 0,
    "space": 0,
    "throughput": 0,
    "regulatory": 0,
    "technical": 0,
    "other": 0,
}


def test_stats_on_an_empty_database(client: TestClient) -> None:
    response = client.get("/stats")

    assert response.status_code == 200
    assert response.json() == {
        "extractions": {
            "total": 0,
            "succeeded": 0,
            "failed": 0,
            "success_rate": None,
            "avg_attempts": None,
            "avg_latency_ms": None,
        },
        "tokens": {"input": 0, "output": 0},
        "estimated_cost": {
            "total_usd": 0.0,
            "avg_per_extraction_usd": None,
            "unpriced_extractions": 0,
        },
        "specs": {"total": 0, "by_confidence": ZERO_CONFIDENCE},
        "constraints_by_kind": ZERO_KINDS,
        # TEST_DAILY_LIMIT from conftest.
        "daily_limit": {"limit": 1000, "used_last_24h": 0, "remaining": 1000},
    }


def test_stats_aggregate_runs_specs_and_constraints(
    client: TestClient, fake_llm: FakeProvider
) -> None:
    create(client, fake_llm, answer("low", ("budget", "space")))
    create(client, fake_llm, "not json", answer("high", ("budget",)))
    fake_llm.queue("bad", "bad", "bad")  # 3 attempts, all rejected: a 422
    assert client.post("/specs", json={"raw_text": RAW_TEXT}).status_code == 422

    body = client.get("/stats").json()

    assert body["extractions"] == {
        "total": 3,
        "succeeded": 2,
        "failed": 1,
        "success_rate": 0.6667,
        "avg_attempts": 2.0,  # (1 + 2 + 3) / 3
        "avg_latency_ms": body["extractions"]["avg_latency_ms"],
    }
    assert body["extractions"]["avg_latency_ms"] >= 0
    # The fake answers 100 in / 50 out per attempt, 6 attempts in all.
    assert body["tokens"] == {"input": 600, "output": 300}
    assert body["specs"] == {
        "total": 2,
        "by_confidence": ZERO_CONFIDENCE | {"low": 1, "high": 1},
    }
    assert body["constraints_by_kind"] == ZERO_KINDS | {"budget": 2, "space": 1}
    assert body["daily_limit"] == {
        "limit": 1000,
        "used_last_24h": 3,
        "remaining": 997,
    }


def test_stats_estimate_cost_per_model_and_count_unpriced_runs(
    client: TestClient, fake_llm: FakeProvider, connection: Connection
) -> None:
    sonnet, opus, _fake = (create(client, fake_llm) for _ in range(3))
    for spec, model in ((sonnet, "claude-sonnet-5-5"), (opus, "claude-opus-5-5")):
        connection.execute(
            update(ExtractionRunRow)
            .where(ExtractionRunRow.spec_id == UUID(spec["id"]))
            .values(model=model, input_tokens=1000, output_tokens=1000)
        )

    cost = client.get("/stats").json()["estimated_cost"]

    # 0.012 for Sonnet and 0.024 for Opus; the fake-model run has no price.
    assert cost["total_usd"] == pytest.approx(0.036)
    assert cost["avg_per_extraction_usd"] == pytest.approx(0.018)
    assert cost["unpriced_extractions"] == 1


def test_remaining_never_goes_below_zero(
    client: TestClient, fake_llm: FakeProvider
) -> None:
    create(client, fake_llm)
    create(client, fake_llm)
    app.dependency_overrides[get_daily_extraction_limit] = lambda: 1

    limit = client.get("/stats").json()["daily_limit"]

    assert limit == {"limit": 1, "used_last_24h": 2, "remaining": 0}


def test_stats_never_call_the_llm(client: TestClient, fake_llm: FakeProvider) -> None:
    client.get("/stats")

    assert fake_llm.calls == []
