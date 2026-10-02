"""Engine and session factory for the RESS results store."""

from __future__ import annotations

import os
from urllib.parse import quote

from sqlalchemy import Engine, create_engine
from sqlalchemy.orm import Session, sessionmaker

from ress.db.models import Base


def database_url_from_env() -> str:
    """Build the PostgreSQL URL from the POSTGRES_* environment (.env)."""
    user = os.environ.get("POSTGRES_USER", "ress")
    password = os.environ.get("POSTGRES_PASSWORD", "")
    host = os.environ.get("POSTGRES_HOST", "127.0.0.1")
    port = os.environ.get("POSTGRES_PORT", "5432")
    db = os.environ.get("POSTGRES_DB", "ress")
    auth = f"{quote(user)}:{quote(password)}" if password else quote(user)
    return f"postgresql+psycopg://{auth}@{host}:{port}/{db}"


def make_engine(url: str) -> Engine:
    # pre-ping keeps long-lived sessions resilient to Postgres restarts
    return create_engine(url, pool_pre_ping=True)


def make_session_factory(engine: Engine) -> sessionmaker[Session]:
    return sessionmaker(bind=engine, expire_on_commit=False)


def init_db(engine: Engine) -> None:
    """Create the schema if it does not exist (idempotent)."""
    Base.metadata.create_all(engine)
