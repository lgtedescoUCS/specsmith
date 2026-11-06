from fastapi import FastAPI

app = FastAPI(
    title="Specsmith",
    description=(
        "Turns free-form client requirements into validated, "
        "structured technical specifications using an LLM."
    ),
    version="0.1.0",
)


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}
