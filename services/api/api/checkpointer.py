"""Durable LangGraph checkpointer — the agent's short-term memory, in Postgres.

LangGraph persists a run's state (the message graph, pending tool calls, any open
interrupt) to a **checkpointer**, keyed by ``thread_id``. This app uses the
Postgres saver so a conversation resumes across requests and process restarts:
the ``thread_id`` is the :class:`~shared.models.conversation.Conversation` id, so
the durable agent state and the human-readable transcript share one key.

Two wrinkles this module handles:

* **Driver.** The saver speaks **psycopg v3**, not the ``asyncpg`` that SQLAlchemy
  uses elsewhere, so it needs a psycopg-form DSN (``postgresql://…`` without the
  ``+asyncpg`` suffix) and its own connection pool. The pool's connections must be
  ``autocommit=True``, ``prepare_threshold=0``, ``row_factory=dict_row`` — the
  exact settings the saver's own ``from_conn_string`` uses — or its statements
  fail at runtime.
* **Schema setup.** ``AsyncPostgresSaver.setup()`` creates the checkpoint tables.
  Like the relational schema, this runs **once as a one-shot** (via
  :func:`setup_checkpointer`), never per-replica on startup — concurrent
  ``CREATE TABLE`` across replicas is exactly the race the project forbids for
  Alembic. The long-lived app builds a *pool-backed* saver (:func:`build_pool` +
  :class:`AsyncPostgresSaver`) and assumes the tables already exist.
"""

from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
from psycopg.rows import dict_row
from psycopg_pool import AsyncConnectionPool

from shared.core.config import Settings

# Connection kwargs the Postgres saver requires (mirrors its own
# ``from_conn_string``); wrong values here surface as runtime statement errors.
_SAVER_CONN_KWARGS = {"autocommit": True, "prepare_threshold": 0, "row_factory": dict_row}


def to_psycopg_dsn(database_url: str) -> str:
    """Convert a SQLAlchemy URL to the psycopg-form DSN the saver needs.

    Strips a ``+driver`` suffix from the scheme (``postgresql+asyncpg://…`` →
    ``postgresql://…``) so the same configured ``DATABASE_URL`` drives both the
    async SQLAlchemy engine and the psycopg-based checkpointer.
    """
    scheme, separator, rest = database_url.partition("://")
    if not separator:
        raise ValueError(f"not a database URL: {database_url!r}")
    base_scheme = scheme.split("+", 1)[0]
    return f"{base_scheme}://{rest}"


def build_pool(settings: Settings) -> AsyncConnectionPool:
    """Build the checkpointer's psycopg connection pool (unopened).

    Returns the pool closed (``open=False``); the caller opens it in the app
    lifespan and closes it on shutdown, so the pool's lifetime matches the
    process rather than any single request.
    """
    return AsyncConnectionPool(
        conninfo=to_psycopg_dsn(settings.database_url),
        open=False,
        kwargs=_SAVER_CONN_KWARGS,
    )


async def setup_checkpointer(settings: Settings) -> None:
    """Create the checkpoint tables (one-shot; idempotent).

    Invoked by the dedicated one-shot migration step — **not** by the API on
    startup — so replicas never race on ``CREATE TABLE``. Safe to re-run: the
    saver only applies missing schema migrations.
    """
    async with AsyncPostgresSaver.from_conn_string(to_psycopg_dsn(settings.database_url)) as saver:
        await saver.setup()


def main() -> None:
    """Run the one-shot checkpoint-schema setup (``python -m api.checkpointer``).

    The entry point a dedicated one-shot deployment step invokes — the LangGraph
    analogue of ``alembic upgrade head`` — so the checkpoint tables are created
    exactly once, out of the request path and away from replica startup.
    """
    import asyncio

    from shared.core.config import get_settings

    asyncio.run(setup_checkpointer(get_settings()))


if __name__ == "__main__":
    main()
