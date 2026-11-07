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

# Published prompts are immutable: a change must be a new vN, so that runs
# recorded with an old version stay reproducible. Add lines here for each
# new version (main and retry); never update an existing hash.
PUBLISHED_PROMPT_SHA256 = {
    "v1.md": "02e1c68dba6f0c2382e92539e4cec8cdd201afb2e21078f87dec22db93a97caa",
    "v1_retry.md": "8aa062857050b75a6174ec8744fd4244165cafbd07e395aa57ddb421a082eaa8",
    "v2.md": "d8dc9b1eb11f7c0af52cb55d7087519b67b5e6dabc8d9da2946414dd090bc8af",
    "v2_retry.md": "7464204eb91323c4ce2afd9e6cc1d6ad6fe2d8d631124064eb6f8648ea31aab6",
    "v3.md": "e3b6feebc71b4d1de1540514371df26f111b1714bf67fd49c1434abf4c134043",
    "v3_retry.md": "7464204eb91323c4ce2afd9e6cc1d6ad6fe2d8d631124064eb6f8648ea31aab6",
}

MAIN = "Main\n{raw_text}\n"
RETRY = "Was:\n{previous_answer}\nErrors:\n{errors}\n"


def write_prompt(
    root: Path, version: str, main: str | None = MAIN, retry: str | None = RETRY
) -> Path:
    folder = root / "extraction"
    folder.mkdir(parents=True, exist_ok=True)
    if main is not None:
        (folder / f"{version}.md").write_text(main, encoding="utf-8")
    if retry is not None:
        (folder / f"{version}_retry.md").write_text(retry, encoding="utf-8")
    return root


def test_loads_the_real_v1_prompt() -> None:
    prompt = load_prompt("v1")

    assert (prompt.task, prompt.version) == ("extraction", "v1")
    assert "requirements analyst" in prompt.template
    assert "{previous_answer}" in prompt.retry_template


def test_every_published_prompt_file_is_pinned() -> None:
    # A new vN that nobody added to the hash table would not be protected.
    on_disk = {p.name for p in (DEFAULT_PROMPTS_DIR / "extraction").glob("*.md")}

    assert on_disk == set(PUBLISHED_PROMPT_SHA256)


@pytest.mark.parametrize("filename", sorted(PUBLISHED_PROMPT_SHA256))
def test_published_prompts_are_unchanged(filename: str) -> None:
    path = DEFAULT_PROMPTS_DIR / "extraction" / filename
    # Normalise line endings so a CRLF checkout does not count as an edit.
    content = path.read_bytes().replace(b"\r\n", b"\n")

    assert hashlib.sha256(content).hexdigest() == PUBLISHED_PROMPT_SHA256[filename], (
        f"prompts/extraction/{filename} was edited. Create a new version instead."
    )


def test_render_inserts_the_raw_text() -> None:
    rendered = load_prompt("v1").render("Need 650 plates/day.")

    assert "---\nNeed 650 plates/day.\n---" in rendered
    assert "{raw_text}" not in rendered


def test_render_retry_is_main_prompt_followed_by_feedback(tmp_path: Path) -> None:
    prompt = load_prompt("v1", prompts_dir=write_prompt(tmp_path, "v1"))

    rendered = prompt.render_retry("text", previous_answer="{}", errors="- bad")

    assert rendered == "Main\ntext\n\nWas:\n{}\nErrors:\n- bad\n"


def test_real_v1_retry_renders_every_placeholder() -> None:
    rendered = load_prompt("v1").render_retry(
        "client text", previous_answer='{"objective": ""}', errors="- objective: x"
    )

    assert rendered.startswith(load_prompt("v1").render("client text"))
    assert '<previous_answer>\n{"objective": ""}\n</previous_answer>' in rendered
    assert "- objective: x" in rendered
    for placeholder in ("{raw_text}", "{previous_answer}", "{errors}"):
        assert placeholder not in rendered


def test_inserted_text_is_never_substituted_again(tmp_path: Path) -> None:
    prompt = load_prompt("v1", prompts_dir=write_prompt(tmp_path, "v1"))
    # Braces and placeholder names inside inputs must come out verbatim.
    raw_text = 'Spec says {"temp": 620} and {} and {raw_text}'
    previous = 'I wrote {errors} and {previous_answer} here'

    rendered = prompt.render_retry(raw_text, previous_answer=previous, errors="E")

    assert rendered == f"Main\n{raw_text}\n\nWas:\n{previous}\nErrors:\nE\n"


def test_loads_from_a_custom_directory(tmp_path: Path) -> None:
    root = write_prompt(tmp_path, "v7", main="Custom: {raw_text}")

    assert load_prompt("v7", prompts_dir=root).render("x") == "Custom: x"


@pytest.mark.parametrize(
    "version",
    ["1", "V1", "v0", "v01", "v1.md", "v1 ", "", "../../.env", "v1/../v2", "v1_retry"],
)
def test_rejects_malformed_versions(version: str) -> None:
    with pytest.raises(InvalidPromptVersionError):
        load_prompt(version)


def test_missing_version_is_a_clear_error(tmp_path: Path) -> None:
    with pytest.raises(PromptNotFoundError, match="extraction/v9"):
        load_prompt("v9", prompts_dir=tmp_path)


def test_missing_retry_file_is_a_clear_error(tmp_path: Path) -> None:
    # A version is both files; a main prompt alone is incomplete.
    root = write_prompt(tmp_path, "v1", retry=None)

    with pytest.raises(PromptNotFoundError, match="extraction/v1_retry"):
        load_prompt("v1", prompts_dir=root)


@pytest.mark.parametrize(
    "main, retry, message",
    [
        ("No placeholder", RETRY, r"\{raw_text\} exactly once; found 0"),
        ("{raw_text} {raw_text}", RETRY, r"\{raw_text\} exactly once; found 2"),
        (MAIN, "{errors}", r"\{previous_answer\} exactly once; found 0"),
        (MAIN, "{previous_answer}", r"\{errors\} exactly once; found 0"),
        (
            MAIN,
            "{previous_answer} {errors} {errors}",
            r"\{errors\} exactly once; found 2",
        ),
    ],
)
def test_each_placeholder_must_appear_exactly_once(
    tmp_path: Path, main: str, retry: str, message: str
) -> None:
    root = write_prompt(tmp_path, "v1", main=main, retry=retry)

    with pytest.raises(MalformedPromptError, match=message):
        load_prompt("v1", prompts_dir=root)
