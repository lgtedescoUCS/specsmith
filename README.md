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
uv run uvicorn specsmith.main:app --reload
```

Open http://localhost:8000/docs

The database is published on host port 5433 rather than 5432, so it does
not clash with a PostgreSQL server already installed on the machine.

```bash
uv run pytest               # offline, no API calls
```

## License

MIT
