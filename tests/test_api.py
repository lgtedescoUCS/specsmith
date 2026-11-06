"""HTTP API against a real PostgreSQL. Skipped when it is not running."""

from typing import Any
from uuid import UUID, uuid4

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import Connection, event, func, select
from sqlalchemy.orm import Session

from specsmith.db.models import RequirementRow
from specsmith.main import app


def payload() -> dict[str, Any]:
    return {
        "raw_text": "Client wants 650 plates/day on the tempering line.",
        "source": "api-test",
        "spec": {
            "objective": "Increase tempering line throughput",
            "current_state": {"throughput": "400 plates/day"},
            "target_state": {"throughput": "650 plates/day"},
            "constraints": [
                {"kind": "space", "value": "12 m of available floor space"},
                {"kind": "deadline", "value": "operational by March"},
            ],
            "open_questions": ["How many shifts per day?"],
            "confidence": "medium",
        },
    }


def requirement_count(connection: Connection) -> int:
    return connection.scalar(select(func.count()).select_from(RequirementRow)) or 0


def test_post_spec_returns_201_with_stored_spec(client: TestClient) -> None:
    response = client.post("/specs", json=payload())

    assert response.status_code == 201
    body = response.json()
    UUID(body["id"])
    UUID(body["requirement_id"])
    assert body["created_at"]
    stored_fields = {k: v for k, v in body.items() if k in payload()["spec"]}
    assert stored_fields == payload()["spec"]


def test_get_spec_returns_what_post_stored(client: TestClient) -> None:
    created = client.post("/specs", json=payload()).json()

    response = client.get(f"/specs/{created['id']}")

    assert response.status_code == 200
    assert response.json() == created


def test_post_without_source_is_accepted(client: TestClient) -> None:
    body = payload()
    del body["source"]

    assert client.post("/specs", json=body).status_code == 201


def test_get_unknown_spec_returns_404(client: TestClient) -> None:
    missing = uuid4()

    response = client.get(f"/specs/{missing}")

    assert response.status_code == 404
    assert response.json() == {"detail": f"Spec {missing} not found"}


def test_get_with_malformed_id_returns_422(client: TestClient) -> None:
    assert client.get("/specs/not-a-uuid").status_code == 422


def _without_confidence(body: dict[str, Any]) -> None:
    del body["spec"]["confidence"]


@pytest.mark.parametrize(
    "mutate",
    [
        pytest.param(lambda b: b.update(raw_text="   "), id="blank-raw-text"),
        pytest.param(lambda b: b.update(source="x" * 81), id="source-too-long"),
        pytest.param(lambda b: b.update(priority="high"), id="unknown-top-field"),
        pytest.param(lambda b: b["spec"].update(owner="me"), id="unknown-spec-field"),
        pytest.param(_without_confidence, id="missing-confidence"),
        pytest.param(
            lambda b: b["spec"].update(open_questions=["Q?"] * 11),
            id="too-many-questions",
        ),
        pytest.param(
            lambda b: b["spec"]["constraints"].append({"kind": "colour", "value": "red"}),
            id="unknown-constraint-kind",
        ),
    ],
)
def test_invalid_payload_returns_422_and_stores_nothing(
    client: TestClient, connection: Connection, mutate: Any
) -> None:
    body = payload()
    mutate(body)

    response = client.post("/specs", json=body)

    assert response.status_code == 422
    assert requirement_count(connection) == 0


def test_failed_commit_is_an_error_not_a_201(connection: Connection) -> None:
    # Guards the scope="function" choice on the session dependency: the
    # commit must run before the response is sent, so its failure is visible.
    # before_commit fires only for the real commit, not for the internal
    # subtransaction of each flush, so the endpoint itself still succeeds.
    def failing_commit(session: Session) -> None:
        raise RuntimeError("simulated commit failure")

    event.listen(Session, "before_commit", failing_commit)
    try:
        client = TestClient(app, raise_server_exceptions=False)
        response = client.post("/specs", json=payload())
    finally:
        event.remove(Session, "before_commit", failing_commit)

    assert response.status_code == 500
    assert requirement_count(connection) == 0
