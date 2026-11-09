"""Domain errors. The API layer maps each one to an explicit HTTP status."""

from uuid import UUID

from specsmith.domain.models import ExtractionMeta, RejectedAttempt


class DomainError(Exception):
    """Base class for errors raised by the domain and service layers."""

    # True for errors raised after the request wrote a record of the
    # failure itself (an audit row). The request's transaction then commits
    # before the error propagates, instead of rolling that record back.
    keeps_transaction: bool = False


class SpecNotFoundError(DomainError):
    def __init__(self, spec_id: UUID) -> None:
        self.spec_id = spec_id
        super().__init__(f"Spec {spec_id} not found")


class RunNotFoundError(DomainError):
    """The spec exists but no extraction run produced it (it was stored
    ready-made, before extraction from raw text existed)."""

    def __init__(self, spec_id: UUID) -> None:
        self.spec_id = spec_id
        super().__init__(f"Spec {spec_id} has no extraction run")


class ExtractionError(DomainError):
    """An extraction that did not produce a spec.

    Carries the metadata of the attempts made, so the failed run can be
    recorded: a failure is data, not the absence of data.
    """

    keeps_transaction = True

    def __init__(
        self,
        reason: str,
        meta: ExtractionMeta,
        rejected: tuple[RejectedAttempt, ...] = (),
    ) -> None:
        self.reason = reason
        self.meta = meta
        # Every answer that failed validation before the extraction gave up
        # or the provider failed, in order.
        self.rejected = rejected
        super().__init__(reason)


class ExtractionFailedError(ExtractionError):
    """The model's output never passed validation, or the model declined."""


class ExtractionTimeoutError(ExtractionError):
    """The model did not answer within the configured timeout."""


class ExtractionUpstreamError(ExtractionError):
    """The LLM provider was unreachable or rejected the call."""


class DailyExtractionLimitError(DomainError):
    """The server already ran its allowed number of extractions in the last
    24 hours. Raised before calling the LLM, so it spends no tokens."""

    def __init__(self, limit: int, retry_after_seconds: int) -> None:
        self.limit = limit
        self.retry_after_seconds = retry_after_seconds
        super().__init__(f"Daily limit of {limit} extractions reached")


class LLMNotConfiguredError(DomainError):
    """No LLM provider credentials are configured on this server."""

    def __init__(self) -> None:
        super().__init__("LLM provider is not configured")
