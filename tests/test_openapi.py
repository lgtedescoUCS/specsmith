"""What the interactive docs show: tags and the request example. Offline."""

from typing import Any

from specsmith.main import EXAMPLE_RAW_TEXT, CreateSpecRequest, app


def schema() -> dict[str, Any]:
    return app.openapi()


def test_every_route_has_a_tag_and_none_is_default() -> None:
    tags = {
        (path, method): operation.get("tags")
        for path, item in schema()["paths"].items()
        for method, operation in item.items()
    }

    assert tags == {
        ("/specs", "post"): ["specs"],
        ("/specs", "get"): ["specs"],
        ("/specs/{spec_id}", "get"): ["specs"],
        ("/specs/{spec_id}/run", "get"): ["specs"],
        ("/health", "get"): ["system"],
        ("/stats", "get"): ["system"],
    }


def test_tags_are_described_in_display_order() -> None:
    assert [t["name"] for t in schema()["tags"]] == ["specs", "system"]
    assert all(t["description"] for t in schema()["tags"])


def test_post_body_example_is_the_guide_sample_and_valid() -> None:
    request_schema = schema()["components"]["schemas"]["CreateSpecRequest"]

    [example] = request_schema["examples"]
    assert example["raw_text"] == EXAMPLE_RAW_TEXT
    assert "têmpera" in example["raw_text"]
    # The prefilled body must pass the endpoint's own validation.
    CreateSpecRequest.model_validate(example)
