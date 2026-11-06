"""Shared fixtures.

Tests that use `db_session` or `client` need PostgreSQL (docker compose up -d).
When it is not reachable they are skipped with a message saying so; the
other tests still run. Nothing here calls an LLM or any external service.

Each session builds a fresh `specsmith_test` database from the Alembic
migrations, so the migrations themselves are under test. Each test then
runs inside a transaction that is rolled back at the end: get_session's
commit only releases a savepoint, and no data survives between tests.

Every test runs with the LLM replaced by a FakeProvider (autouse), so no
test can reach the network or spend tokens, even by accident. Tests script
its answers through the `fake_llm` fixture.
"""

import os
from collections.abc import Iterator

import pytest
from alembic import command
from alembic.config import Config
from fastapi.testclient import TestClient
from sqlalchemy import Connection, Engine, create_engine, make_url, text
from sqlalchemy.exc import OperationalError
from sqlalchemy.orm import Session, sessionmaker

from specsmith.db import session as db_session_module
from specsmith.llm.fake import FakeProvider
from specsmith.llm.prompts import load_prompt
from specsmith.main import app, get_extractor, get_provider
from specsmith.services.extraction import ExtractionService

TEST_MAX_RETRIES = 2

TEST_DATABASE_URL = os.environ.get(
    "TEST_DATABASE_URL",
    "postgresql+psycopg://specsmith:specsmith@127.0.0.1:5433/specsmith_test",
)


def _recreate_database(url: str) -> None:
    target = make_url(url)
    # CREATE/DROP DATABASE cannot run inside a transaction, hence AUTOCOMMIT,
    # and cannot run while connected to the target, hence the admin database.
    admin = create_engine(
        target.set(database="postgres"),
        isolation_level="AUTOCOMMIT",
        connect_args={"connect_timeout": 3},
    )
    try:
        with admin.connect() as conn:
            name = target.database
            conn.execute(text(f'DROP DATABASE IF EXISTS "{name}" WITH (FORCE)'))
            conn.execute(text(f'CREATE DATABASE "{name}"'))
    finally:
        admin.dispose()


def _migrate(url: str) -> None:
    config = Config("alembic.ini")
    # ConfigParser treats % as interpolation; escape it in case of passwords.
    config.set_main_option("sqlalchemy.url", url.replace("%", "%%"))
    command.upgrade(config, "head")


@pytest.fixture(scope="session")
def engine() -> Iterator[Engine]:
    safe_url = make_url(TEST_DATABASE_URL).render_as_string(hide_password=True)
    try:
        _recreate_database(TEST_DATABASE_URL)
    except OperationalError as exc:
        pytest.skip(
            f"PostgreSQL not reachable at {safe_url}; "
            f"start it with `docker compose up -d`. ({exc.orig.__class__.__name__})"
        )
    _migrate(TEST_DATABASE_URL)

    test_engine = create_engine(TEST_DATABASE_URL)
    yield test_engine
    test_engine.dispose()


@pytest.fixture
def connection(
    engine: Engine, monkeypatch: pytest.MonkeyPatch
) -> Iterator[Connection]:
    with engine.connect() as conn:
        outer = conn.begin()
        factory = sessionmaker(
            bind=conn,
            expire_on_commit=False,
            join_transaction_mode="create_savepoint",
        )
        # The real get_session runs in API tests; only its factory is swapped.
        monkeypatch.setattr(db_session_module, "get_sessionmaker", lambda: factory)
        yield conn
        outer.rollback()


@pytest.fixture
def db_session(connection: Connection) -> Iterator[Session]:
    factory = db_session_module.get_sessionmaker()
    with factory.begin() as session:
        yield session


@pytest.fixture
def client(connection: Connection) -> TestClient:
    return TestClient(app)


@pytest.fixture(autouse=True)
def fake_llm() -> Iterator[FakeProvider]:
    """The LLM for this test. Starts with no scripted answers, so a test that
    triggers an extraction without scripting one fails loudly."""
    provider = FakeProvider([])
    # get_extractor is overridden too, not only get_provider: the real one
    # reads Settings, which needs a DATABASE_URL the offline suite may lack.
    app.dependency_overrides[get_provider] = lambda: provider
    app.dependency_overrides[get_extractor] = lambda: ExtractionService(
        provider, load_prompt("v1"), max_retries=TEST_MAX_RETRIES
    )
    yield provider
    app.dependency_overrides.pop(get_provider, None)
    app.dependency_overrides.pop(get_extractor, None)
