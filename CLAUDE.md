# Specsmith — working rules for Claude Code

Specsmith turns free-form client requirements into validated, structured
technical specifications using an LLM. FastAPI, PostgreSQL, SQLAlchemy 2.0,
Alembic, Pydantic v2, pytest, uv. Python 3.12.

The full build guide lives in `notes/guide.md` (git-ignored). Read it before
starting any phase.

## How we work
- One phase at a time. Before writing code for a phase, show a short plan and
  wait for my approval.
- Stop after each coherent unit of work (one Conventional Commit) so I can
  review the diff. Do not commit until I say so.
- Explain non-obvious decisions in one or two sentences.
- Conventional Commits only: feat, fix, test, docs, refactor, chore.

## Architecture rules
- Layers: `domain` knows nothing about the database; `db` knows nothing about
  the LLM. Business logic never imports a vendor SDK.
- `LLMProvider` is a Protocol. `FakeProvider` is written and tested before any
  real provider.
- Structured output is enforced via the provider's tool use / structured
  output, then validated with Pydantic, with retry that feeds the validation
  error back (max retries from config).
- `extraction_run` is written on success and on failure.
- Prompts are versioned files in `prompts/extraction/vN.md`. Never edit an old
  version; create a new one.
- Domain errors map to explicit HTTP status codes (422 extraction failed,
  504 LLM timeout). No generic 500s.

## Hard rules
- Never read, print or commit `.env` or any API key.
- The test suite must run offline with `FakeProvider` and spend no tokens.
- In `eval/dataset.jsonl`, you may draft the requirement texts, but never write
  or alter the `expected` answers. I annotate them myself. You may build the
  runner and the scoring.
- No real client names, companies or confidential figures anywhere in the repo.
- Never put invented evaluation numbers in the README. Only numbers produced
  by `eval/run_eval.py`.
- For SDK calls and model names, check the provider's current official docs
  instead of relying on memory.
