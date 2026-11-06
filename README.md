# Specsmith

Turns free-form client requirements into validated, structured technical
specifications using an LLM.

> Work in progress. The API currently exposes only a health check.

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

## License

MIT
