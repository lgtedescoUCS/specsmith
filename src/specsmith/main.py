from functools import lru_cache
from typing import Annotated
from uuid import UUID

from fastapi import Depends, FastAPI, Request, status
from fastapi.responses import JSONResponse, RedirectResponse
from pydantic import BaseModel, ConfigDict, Field
from sqlalchemy.orm import Session

from specsmith.config import get_settings
from specsmith.db import repository
from specsmith.db.session import database_is_reachable, get_session
from specsmith.domain.errors import (
    DailyExtractionLimitError,
    ExtractionError,
    ExtractionFailedError,
    ExtractionTimeoutError,
    ExtractionUpstreamError,
    LLMNotConfiguredError,
    SpecNotFoundError,
)
from specsmith.domain.models import StoredSpec
from specsmith.llm.anthropic import AnthropicProvider
from specsmith.llm.base import LLMProvider
from specsmith.llm.prompts import load_prompt
from specsmith.services.extraction import ExtractionService
from specsmith.services.specs import create_spec_from_text

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


# This module is the composition root: the only place that picks a concrete
# provider. Tests override get_provider with a FakeProvider.
@lru_cache
def get_provider() -> LLMProvider:
    # Cached: one SDK client, and so one connection pool, per process.
    # A missing key raises every time; lru_cache does not cache exceptions.
    settings = get_settings()
    if settings.anthropic_api_key is None:
        raise LLMNotConfiguredError()
    return AnthropicProvider.create(
        api_key=settings.anthropic_api_key.get_secret_value(),
        model=settings.llm_model,
        timeout_seconds=settings.llm_timeout_seconds,
    )


def get_extractor(
    provider: Annotated[LLMProvider, Depends(get_provider)],
) -> ExtractionService:
    settings = get_settings()
    return ExtractionService(
        provider,
        load_prompt(settings.prompt_version),
        max_retries=settings.max_extraction_retries,
    )


ExtractorDep = Annotated[ExtractionService, Depends(get_extractor)]


def get_daily_extraction_limit() -> int:
    return get_settings().max_extractions_per_day


DailyLimitDep = Annotated[int, Depends(get_daily_extraction_limit)]

# Upper bound on the client text, so one request cannot run up an
# arbitrary token bill. Generous for a written requirement.
RAW_TEXT_MAX_LENGTH = 20_000


class CreateSpecRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    raw_text: str = Field(min_length=1, max_length=RAW_TEXT_MAX_LENGTH)
    source: str | None = Field(default=None, max_length=80)


@app.exception_handler(SpecNotFoundError)
def spec_not_found(request: Request, exc: SpecNotFoundError) -> JSONResponse:
    return JSONResponse(
        status_code=status.HTTP_404_NOT_FOUND, content={"detail": str(exc)}
    )


# Each extraction error has its own status; the failed run is already
# committed (keeps_transaction) when these handlers run.
_EXTRACTION_RESPONSES: dict[type[ExtractionError], tuple[int, str]] = {
    ExtractionFailedError: (
        status.HTTP_422_UNPROCESSABLE_CONTENT,
        "Could not extract a valid specification from the text",
    ),
    ExtractionTimeoutError: (
        status.HTTP_504_GATEWAY_TIMEOUT,
        "The LLM did not answer in time",
    ),
    ExtractionUpstreamError: (
        status.HTTP_502_BAD_GATEWAY,
        "The LLM provider is unavailable",
    ),
}


def extraction_error(request: Request, exc: Exception) -> JSONResponse:
    assert isinstance(exc, ExtractionError)
    status_code, detail = _EXTRACTION_RESPONSES[type(exc)]
    content: dict[str, object] = {"detail": detail, "attempts": exc.meta.attempts}
    # Validation errors and refusals describe the model's answer to this
    # text, which is useful to the caller. Timeout and upstream reasons
    # describe our infrastructure; they stay in extraction_run only.
    if isinstance(exc, ExtractionFailedError):
        content["reason"] = exc.reason
    return JSONResponse(status_code=status_code, content=content)


for _error_type in _EXTRACTION_RESPONSES:
    app.add_exception_handler(_error_type, extraction_error)


@app.exception_handler(LLMNotConfiguredError)
def llm_not_configured(request: Request, exc: LLMNotConfiguredError) -> JSONResponse:
    return JSONResponse(
        status_code=status.HTTP_503_SERVICE_UNAVAILABLE, content={"detail": str(exc)}
    )


@app.exception_handler(DailyExtractionLimitError)
def daily_limit_reached(
    request: Request, exc: DailyExtractionLimitError
) -> JSONResponse:
    return JSONResponse(
        status_code=status.HTTP_429_TOO_MANY_REQUESTS,
        content={"detail": str(exc)},
        headers={"Retry-After": str(exc.retry_after_seconds)},
    )


_ERROR_RESPONSES = {
    422: {"description": "The LLM output never passed validation"},
    429: {"description": "The daily extraction limit was reached; see Retry-After"},
    502: {"description": "The LLM provider is unreachable or rejected the call"},
    503: {"description": "No LLM provider is configured"},
    504: {"description": "The LLM did not answer in time"},
}


# The bare URL of the deployment opens the interactive docs. Kept out of the
# OpenAPI schema: it is a convenience for browsers, not part of the API.
# Temporary (307), so browsers do not cache it if / ever serves a page.
@app.get("/", include_in_schema=False)
def root() -> RedirectResponse:
    return RedirectResponse(url="/docs")


@app.get(
    "/health",
    responses={503: {"description": "The database is not reachable"}},
)
def health(session: SessionDep) -> JSONResponse:
    # 503, not 500: the app is running but cannot serve requests, which is
    # what load balancers and orchestrators expect from a failing check.
    # Error details stay out of the body so the check leaks nothing.
    if not database_is_reachable(session):
        return JSONResponse(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            content={"status": "unavailable", "database": "unreachable"},
        )
    return JSONResponse(content={"status": "ok", "database": "ok"})


@app.post(
    "/specs",
    status_code=status.HTTP_201_CREATED,
    responses=_ERROR_RESPONSES,
)
def create_spec(
    body: CreateSpecRequest,
    session: SessionDep,
    extractor: ExtractorDep,
    daily_limit: DailyLimitDep,
) -> StoredSpec:
    return create_spec_from_text(
        session,
        extractor,
        raw_text=body.raw_text,
        source=body.source,
        daily_limit=daily_limit,
    )


@app.get("/specs/{spec_id}")
def read_spec(spec_id: UUID, session: SessionDep) -> StoredSpec:
    return repository.get_spec(session, spec_id)
