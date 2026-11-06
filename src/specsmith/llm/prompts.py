"""Loads versioned prompt templates from prompts/<task>/.

A version is a pair of files that change together:
  vN.md        the main prompt, with {raw_text}
  vN_retry.md  appended on a retry, with {previous_answer} and {errors}

Prompts are files, not strings in code, so each version is a reviewable
artifact in git and every extraction can record which one produced it.
One prompt_version covers both texts, so a recorded run identifies every
word the model saw. Old versions are never edited; a change is a new vN.
"""

import re
from dataclasses import dataclass
from pathlib import Path

# src/specsmith/llm/prompts.py -> repository root -> prompts/
DEFAULT_PROMPTS_DIR = Path(__file__).resolve().parents[3] / "prompts"

MAIN_PLACEHOLDERS = ("raw_text",)
RETRY_PLACEHOLDERS = ("previous_answer", "errors")

# Only v1, v2, ... are valid. This also keeps a configured version from
# being a path such as "../../.env".
_VERSION_PATTERN = re.compile(r"v[1-9][0-9]*")


class InvalidPromptVersionError(ValueError):
    pass


class PromptNotFoundError(LookupError):
    pass


class MalformedPromptError(ValueError):
    pass


@dataclass(frozen=True)
class Prompt:
    task: str
    version: str
    template: str
    retry_template: str

    def render(self, raw_text: str) -> str:
        return _fill(self.template, {"raw_text": raw_text})

    def render_retry(self, raw_text: str, previous_answer: str, errors: str) -> str:
        """The main prompt followed by the retry feedback."""
        retry = _fill(
            self.retry_template,
            {"previous_answer": previous_answer, "errors": errors},
        )
        return f"{self.render(raw_text)}\n{retry}"


def load_prompt(
    version: str,
    *,
    task: str = "extraction",
    prompts_dir: Path = DEFAULT_PROMPTS_DIR,
) -> Prompt:
    if not _VERSION_PATTERN.fullmatch(version):
        raise InvalidPromptVersionError(
            f"Prompt version must look like v1, v2, ...; got {version!r}"
        )
    return Prompt(
        task=task,
        version=version,
        template=_read(prompts_dir, task, version, MAIN_PLACEHOLDERS),
        retry_template=_read(
            prompts_dir, task, f"{version}_retry", RETRY_PLACEHOLDERS
        ),
    )


def _read(
    prompts_dir: Path, task: str, name: str, placeholders: tuple[str, ...]
) -> str:
    path = prompts_dir / task / f"{name}.md"
    try:
        template = path.read_text(encoding="utf-8")
    except FileNotFoundError:
        raise PromptNotFoundError(
            f"No prompt {task}/{name} (looked for {path})"
        ) from None

    # Each placeholder exactly once: a missing one means the model never
    # sees that input, a repeated one means it sees it twice. Both are
    # template bugs and should fail at load, not produce a plausible but
    # wrong extraction.
    for placeholder in placeholders:
        count = template.count(f"{{{placeholder}}}")
        if count != 1:
            raise MalformedPromptError(
                f"Prompt {task}/{name} must contain {{{placeholder}}} exactly "
                f"once; found {count}"
            )
    return template


def _fill(template: str, values: dict[str, str]) -> str:
    # One regex pass instead of str.format or chained str.replace. Client
    # text and model answers may contain "{" or "}", which str.format would
    # treat as fields; and with chained replaces, a previous answer that
    # happens to contain "{errors}" would be substituted by the second pass.
    # Here inserted text is never scanned again.
    pattern = re.compile("|".join(re.escape(f"{{{name}}}") for name in values))
    return pattern.sub(lambda match: values[match.group(0)[1:-1]], template)
