"""SQLAlchemy declarative base shared by all ORM models."""

from __future__ import annotations

from sqlalchemy.orm import DeclarativeBase


class Base(DeclarativeBase):
    """Common base for all ORM models.

    Distinct from Pydantic domain models (app.domain.*) — never exposed
    directly to routes or services without conversion.
    """
