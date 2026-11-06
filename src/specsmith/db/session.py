"""Engine and session factory, created lazily from settings.

Nothing connects at import time, so the app (and the offline tests) can be
imported without a database or a DATABASE_URL.
"""

from collections.abc import Iterator
from functools import lru_cache

from sqlalchemy import Engine, create_engine, text
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.orm import Session, sessionmaker

from specsmith.config import get_settings


@lru_cache
def get_engine() -> Engine:
    settings = get_settings()
    return create_engine(
        settings.database_url,
        pool_pre_ping=True,
        connect_args={"connect_timeout": settings.database_connect_timeout_seconds},
    )


@lru_cache
def get_sessionmaker() -> sessionmaker[Session]:
    return sessionmaker(bind=get_engine(), expire_on_commit=False)


def get_session() -> Iterator[Session]:
    """One session and one transaction per request.

    The transaction commits when the request handler returns and rolls back
    if it raises. Repositories only flush; they never commit, so the whole
    request is atomic and tests can wrap it in a transaction they roll back.
    """
    with get_sessionmaker().begin() as session:
        yield session


def database_is_reachable(session: Session) -> bool:
    try:
        session.execute(text("SELECT 1"))
    except SQLAlchemyError:
        return False
    return True
