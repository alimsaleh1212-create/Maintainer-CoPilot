"""SQLAlchemy ORM base — re-exports from app.repositories.base to share one MetaData."""

from __future__ import annotations

from app.repositories.base import Base

__all__ = ["Base"]
