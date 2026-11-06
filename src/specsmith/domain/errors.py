"""Domain errors. The API layer maps each one to an explicit HTTP status."""

from uuid import UUID

from specsmith.domain.models import ExtractionMeta


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


class ExtractionError(DomainError):
    """An extraction that did not produce a spec.

    Carries the metadata of the attempts made, so the failed run can be
    recorded: a failure is data, not the absence of data.
    """

    keeps_transaction = True

    def __init__(self, reason: str, meta: ExtractionMeta) -> None:
        self.reason = reason
        self.meta = meta
        super().__init__(reason)


class ExtractionFailedError(ExtractionError):
    """The model's output never passed validation, or the model declined."""


class ExtractionTimeoutError(ExtractionError):
    """The model did not answer within the configured timeout."""
