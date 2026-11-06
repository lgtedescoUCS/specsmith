"""Domain errors. The API layer maps each one to an explicit HTTP status."""

from uuid import UUID


class DomainError(Exception):
    """Base class for errors raised by the domain and service layers."""


class SpecNotFoundError(DomainError):
    def __init__(self, spec_id: UUID) -> None:
        self.spec_id = spec_id
        super().__init__(f"Spec {spec_id} not found")
