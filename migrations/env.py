from logging.config import fileConfig

from alembic import context
from sqlalchemy import create_engine, pool

from specsmith.config import get_settings
from specsmith.db.models import Base

config = context.config

if config.config_file_name is not None:
    # disable_existing_loggers=False: the test suite runs migrations in its
    # own process, and the default would silence every logger the app had
    # already created (so caplog would see nothing from them).
    fileConfig(config.config_file_name, disable_existing_loggers=False)

target_metadata = Base.metadata


def database_url() -> str:
    # An explicit sqlalchemy.url (set by the test suite) wins; otherwise the
    # application's own DATABASE_URL is used, so there is one source of truth.
    return config.get_main_option("sqlalchemy.url") or get_settings().database_url


def run_migrations_offline() -> None:
    """Emit SQL to stdout instead of running it (alembic upgrade --sql)."""
    context.configure(
        url=database_url(),
        target_metadata=target_metadata,
        literal_binds=True,
        dialect_opts={"paramstyle": "named"},
    )

    with context.begin_transaction():
        context.run_migrations()


def run_migrations_online() -> None:
    connectable = create_engine(database_url(), poolclass=pool.NullPool)

    with connectable.connect() as connection:
        context.configure(connection=connection, target_metadata=target_metadata)

        with context.begin_transaction():
            context.run_migrations()


if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()
