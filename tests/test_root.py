"""The root URL redirects to the interactive docs. Offline."""

from fastapi.testclient import TestClient

from specsmith.main import app


def test_root_redirects_to_docs() -> None:
    response = TestClient(app).get("/", follow_redirects=False)

    assert response.status_code == 307
    assert response.headers["location"] == "/docs"


def test_root_is_not_part_of_the_openapi_schema() -> None:
    assert "/" not in app.openapi()["paths"]
