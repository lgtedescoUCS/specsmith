"""Read-only endpoints: a spec's run, the spec listing and /stats.

Against a real PostgreSQL (skipped when it is not running). Specs are
created through POST /specs with the FakeProvider, or stored ready-made
through the repository for specs that have no run.
"""

import json
from typing import Any
from uuid import UUID, uuid4

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import Connection, update
from sqlalchemy.orm import Session

from specsmith.db import repository
from specsmith.db.models import ExtractionRunRow
from specsmith.domain.models import Spec
from specsmith.llm.fake import FakeProvider

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
