"""Loads versioned prompt templates from prompts/<task>/vN.md.

Prompts are files, not strings in code, so each version is a reviewable
artifact in git and every extraction can record which one produced it.
Old versions are never edited; a change is a new vN file.
"""

import re
from dataclasses import dataclass
from pathlib import Path

# src/specsmith/llm/prompts.py -> repository root -> prompts/
DEFAULT_PROMPTS_DIR = Path(__file__).resolve().parents[3] / "prompts"

PLACEHOLDER = "{raw_text}"

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

    def render(self, raw_text: str) -> str:
        # str.replace, not str.format: client text may contain "{" or "}",
        # which str.format would treat as fields and fail on. The inserted
        # text is not scanned again, so a client who writes "{raw_text}"
        # changes nothing.
        return self.template.replace(PLACEHOLDER, raw_text)


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
    path = prompts_dir / task / f"{version}.md"
    try:
        template = path.read_text(encoding="utf-8")
    except FileNotFoundError:
        raise PromptNotFoundError(
            f"No prompt {task}/{version} (looked for {path})"
        ) from None

    # Exactly one placeholder: with none the model would never see the
    # client text, with two it would see it twice. Both are template bugs
    # and should fail at load, not produce a plausible but wrong extraction.
    count = template.count(PLACEHOLDER)
    if count != 1:
        raise MalformedPromptError(
            f"Prompt {task}/{version} must contain {PLACEHOLDER} exactly once; "
            f"found {count}"
        )
    return Prompt(task=task, version=version, template=template)
