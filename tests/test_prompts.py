"""Versioned prompt loading. Offline; uses tmp_path for synthetic prompts."""

import hashlib
from pathlib import Path

import pytest

from specsmith.llm.prompts import (
    DEFAULT_PROMPTS_DIR,
    InvalidPromptVersionError,
    MalformedPromptError,
    PromptNotFoundError,
    load_prompt,
)

# Published prompts are immutable: a change must be a new vN file, so that
# runs recorded with an old version stay reproducible. Add a line here for
# each new version; never update an existing hash.
PUBLISHED_PROMPT_SHA256 = {
    "v1": "02e1c68dba6f0c2382e92539e4cec8cdd201afb2e21078f87dec22db93a97caa",
}


def write_prompt(root: Path, version: str, text: str) -> Path:
    path = root / "extraction" / f"{version}.md"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return root


def test_loads_the_real_v1_prompt() -> None:
    prompt = load_prompt("v1")

    assert (prompt.task, prompt.version) == ("extraction", "v1")
    assert "requirements analyst" in prompt.template


@pytest.mark.parametrize("version", sorted(PUBLISHED_PROMPT_SHA256))
def test_published_prompts_are_unchanged(version: str) -> None:
    path = DEFAULT_PROMPTS_DIR / "extraction" / f"{version}.md"
    # Normalise line endings so a CRLF checkout does not count as an edit.
    content = path.read_bytes().replace(b"\r\n", b"\n")

    assert hashlib.sha256(content).hexdigest() == PUBLISHED_PROMPT_SHA256[version], (
        f"prompts/extraction/{version}.md was edited. "
        "Create a new version instead."
    )


def test_render_inserts_the_raw_text() -> None:
    rendered = load_prompt("v1").render("Need 650 plates/day.")

    assert "---\nNeed 650 plates/day.\n---" in rendered
    assert "{raw_text}" not in rendered


def test_render_keeps_braces_in_client_text(tmp_path: Path) -> None:
    root = write_prompt(tmp_path, "v1", "Before\n{raw_text}\nAfter\n")
    text = 'Spec says {"temp": 620} and {} and {raw_text} literally'

    rendered = load_prompt("v1", prompts_dir=root).render(text)

    assert rendered == f"Before\n{text}\nAfter\n"


def test_loads_from_a_custom_directory(tmp_path: Path) -> None:
    root = write_prompt(tmp_path, "v7", "Custom: {raw_text}")

    assert load_prompt("v7", prompts_dir=root).render("x") == "Custom: x"


@pytest.mark.parametrize(
    "version", ["1", "V1", "v0", "v01", "v1.md", "v1 ", "", "../../.env", "v1/../v2"]
)
def test_rejects_malformed_versions(version: str) -> None:
    with pytest.raises(InvalidPromptVersionError):
        load_prompt(version)


def test_missing_version_is_a_clear_error(tmp_path: Path) -> None:
    with pytest.raises(PromptNotFoundError, match="extraction/v9"):
        load_prompt("v9", prompts_dir=tmp_path)


@pytest.mark.parametrize(
    "template, count",
    [("No placeholder here", 0), ("{raw_text} and again {raw_text}", 2)],
)
def test_template_needs_exactly_one_placeholder(
    tmp_path: Path, template: str, count: int
) -> None:
    root = write_prompt(tmp_path, "v1", template)

    with pytest.raises(MalformedPromptError, match=f"found {count}"):
        load_prompt("v1", prompts_dir=root)
