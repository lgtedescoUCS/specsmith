from typing import Annotated
from uuid import UUID

from fastapi import Depends, FastAPI, Request, status
from fastapi.responses import JSONResponse
from pydantic import BaseModel, ConfigDict, Field
from sqlalchemy.orm import Session

from specsmith.db import repository
from specsmith.db.session import get_session
from specsmith.domain.errors import SpecNotFoundError
from specsmith.domain.models import Spec, StoredSpec

app = FastAPI(
    title="Specsmith",
    description=(
        "Turns free-form client requirements into validated, "
        "structured technical specifications using an LLM."
    ),
    version="0.1.0",
)

# scope="function" makes FastAPI run the code after get_session's yield
# (the commit) before sending the response. With the default "request"
# scope it runs after the response is sent, so a failed commit would still
# reach the client as a 201.
SessionDep = Annotated[Session, Depends(get_session, scope="function")]


class CreateSpecRequest(BaseModel):
    """Phase 1 input: a spec written by hand. From phase 2 the LLM fills it."""

    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    raw_text: str = Field(min_length=1)
    source: str | None = Field(default=None, max_length=80)
    spec: Spec


@app.exception_handler(SpecNotFoundError)
def spec_not_found(request: Request, exc: SpecNotFoundError) -> JSONResponse:
    return JSONResponse(
        status_code=status.HTTP_404_NOT_FOUND, content={"detail": str(exc)}
    )


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/specs", status_code=status.HTTP_201_CREATED)
def create_spec(body: CreateSpecRequest, session: SessionDep) -> StoredSpec:
    return repository.save_spec(
        session, raw_text=body.raw_text, source=body.source, spec=body.spec
    )


@app.get("/specs/{spec_id}")
def read_spec(spec_id: UUID, session: SessionDep) -> StoredSpec:
    return repository.get_spec(session, spec_id)
