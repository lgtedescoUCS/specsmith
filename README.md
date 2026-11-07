# Specsmith

Turns free-form client requirements into validated, structured technical
specifications using an LLM.

**[Live demo](https://specsmith.onrender.com/docs)** · interactive OpenAPI

The demo runs on free tiers: after 15 idle minutes the first request
takes about a minute while the service wakes up. Extractions are capped
at 20 per 24 hours across all users; past that, `POST /specs` answers
429 with a `Retry-After` header.

> Work in progress. `POST /specs` extracts a specification from raw text
> and `GET /specs/{id}` returns it; listing and stats endpoints come later.

## Evaluation

`eval/dataset.jsonl` holds 20 synthetic requirements for flat glass
processing equipment (tempering, laminating, cutting, insulating glass,
screen printing), 11 in Portuguese and 9 in English, including vague,
very short, ambiguous and conflicting ones. Each has a hand-written
answer key. Exact string matching does not work on LLM output, so
`eval/run_eval.py` scores each field separately:

- **Objective**: share of the expected keywords found in the objective.
- **Constraints (F1)**: precision and recall over constraint kinds,
  summed over all cases.
- **Confidence**: exact match on low / medium / high.
- **Must-ask recall**: share of the expected topics that the open
  questions mention.
- **Redundant questions**: share of the topics the text already answers
  that the open questions ask about anyway (lower is better).
- **Domain clean**: share of cases with no domain error, such as reading
  "tempering" as the heat treatment of steel.

Results with `claude-sonnet-5-5`, one run per prompt version, dataset as
of commit `646b8d5`:

| Prompt | Objective | Constraints (F1) | Confidence | Must-ask recall | Redundant questions | Domain clean | Success | Avg cost |
|---|---|---|---|---|---|---|---|---|
| v1 | 88% | 0.68 | 90% | 25% | 11% | 95% | 100% | $0.0158 |
| v2 | 93% | 0.79 | 80% | 28% | 14% | 100% | 100% | $0.0123 |
| v3 | 86% | 0.86 | 90% | 63% | 14% | 100% | 100% | $0.0120 |

What changed between versions (each version is a separate file in
`prompts/extraction/`, and its commit message has the details):

- **v2** states the domain (flat glass, so "têmpera" is glass tempering),
  defines every constraint kind, keeps derived values out of the target
  state and caps the open questions at 10 in priority order. In v1, 9 of
  20 cases needed a second call because the first answer had 11 to 13
  questions; v2 and v3 needed none, which is most of the cost drop.
- **v3** lists what a project engineer usually needs to quote flat glass
  equipment, so the open questions ask for it, and rewrites the
  confidence rule around whether the technical scope is explicit.

```bash
uv run python eval/run_eval.py --prompt-version v3   # calls the API
uv run python eval/run_eval.py --rescore eval/results/<report>.json   # free
```

Reports go to `eval/results/` (not committed), with each case's output,
score, token usage and every answer rejected by validation.

## Limitations

- **Small and narrow dataset.** 20 cases, one domain, one annotator. It
  can show that a prompt version is better on these cases, not that the
  system is accurate in general.
- **Tuned on the test set.** v2 and v3 were written after reading the
  errors on these same 20 cases, and there is no held-out set. The v3
  question checklist comes from the annotator's own practice, the same
  person who wrote the expected topics. The v3 numbers are optimistic.
- **Lexical scoring.** Topics and keywords are matched as words, not as
  meaning. Must-ask recall is a lower bound: a question that covers a
  topic in other words is not counted. Redundant questions also has false
  positives: in v2 and v3, some flagged questions only share a word with
  a stated fact (for example "interlayers" in a question about glass
  layer configuration). The objective keyword score misses hyphenated
  forms such as "triple-glazing".
- **One run per version.** Output varies between runs of the same
  prompt. Two runs of v1 scored against the same answer keys differed by
  11 points in redundant questions, 5 in confidence and 5 in domain
  clean, so differences of that size between versions are within noise.
- **No human review step.** The output is a first pass, not a
  specification anyone should quote from unreviewed.
- **Hardcoded prices.** Cost estimates use per-token prices checked on
  2026-11-06 and will drift when the provider changes them.

## Running locally

Requires Python 3.12+, [uv](https://docs.astral.sh/uv/) and Docker.

```bash
git clone https://github.com/lgtedescoUCS/specsmith.git
cd specsmith

cp .env.example .env        # add your API key
docker compose up -d        # postgres 16 on host port 5433

uv sync
uv run alembic upgrade head
uv run uvicorn specsmith.main:app --reload
```

Open http://localhost:8000/docs

The database is published on host port 5433 rather than 5432, so it does
not clash with a PostgreSQL server already installed on the machine.

```bash
uv run pytest               # offline, no API calls
```

Tests that touch the database need PostgreSQL running (`docker compose up -d`).
When it is not reachable they are skipped with a message saying so, and the
rest of the suite still runs. They use a separate `specsmith_test` database,
rebuilt from the migrations at the start of each run, and every test is
rolled back, so your development data is never touched. Override the
location with `TEST_DATABASE_URL` if needed.

## Deployment

The live demo is a Docker web service on [Render](https://render.com)
with a PostgreSQL 16 database on [Neon](https://neon.com), both on free
plans. Render auto-deploys every push to `main`.

The image is built from the multi-stage `Dockerfile`. On every start the
container runs `alembic upgrade head` and then uvicorn on `$PORT`, so
migrations need no separate step.

To deploy your own copy:

1. Create a Neon project with PostgreSQL 16 and copy its **direct**
   connection string, with connection pooling off (no `-pooler` in the
   host).
2. Create a Render web service from the repository, with the Docker
   language and the Free instance type.
3. Set the environment variables below in the Render dashboard. They
   live there only; nothing secret is in the repository.
4. Leave the health check path empty.

| Variable | Value |
|---|---|
| `DATABASE_URL` | The Neon direct connection string, as given (`postgresql://…?sslmode=require`) |
| `ANTHROPIC_API_KEY` | A key used only for this deployment |
| `MAX_EXTRACTIONS_PER_DAY` | Optional, default `20` |
| `PROMPT_VERSION`, `LLM_MODEL`, `MAX_EXTRACTION_RETRIES`, `LLM_TIMEOUT_SECONDS` | Optional, defaults as in `.env.example` |

Why it is set up this way:

- **The URL is used as the provider gives it.** Settings rewrite a
  driverless `postgresql://` or `postgres://` scheme to
  `postgresql+psycopg://` and keep the rest of the URL.
- **Direct connection, not the pooler.** Neon's pooler runs in
  transaction mode, which Neon advises against for schema migrations.
  One instance needs few connections, so the app uses the direct
  connection too.
- **No HTTP health check.** Render checks running services every few
  seconds. Pointed at `/health`, which queries the database, the checks
  would keep Neon's compute from scaling to zero and use up the free
  plan's monthly compute hours. Render falls back to a TCP check on the
  port; `/health` is still there for people.
- **Budget guard.** The daily cap counts runs in the database, failures
  included, so restarts do not reset it. It is soft when requests arrive
  at the same moment; the hard ceiling is a spend limit on the API key's
  workspace.

## License

MIT
