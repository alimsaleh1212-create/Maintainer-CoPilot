"""Add parent_id and parent_text to rag_chunks for small-to-big retrieval.

The pattern is "parent-document retrieval, denormalized":
- Children (chunks) are what we embed + match against.
- Parents (full docs) provide expansion context at LLM time.
- Parent text lives inline in the same row (TEXT) rather than in MinIO —
  one DB roundtrip total, no second store to keep consistent.

Revision ID: 002_a3b6c9
Revises: 001_c0ffee_baseline
Create Date: 2026-05-21
"""

from __future__ import annotations

import sqlalchemy as sa
from alembic import op

revision = "002_a3b6c9"
down_revision = "001_c0ffee"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column("rag_chunks", sa.Column("parent_id", sa.String(255), nullable=True))
    op.add_column("rag_chunks", sa.Column("parent_text", sa.Text(), nullable=True))
    op.create_index("ix_rag_chunks_parent_id", "rag_chunks", ["parent_id"])


def downgrade() -> None:
    op.drop_index("ix_rag_chunks_parent_id", table_name="rag_chunks")
    op.drop_column("rag_chunks", "parent_text")
    op.drop_column("rag_chunks", "parent_id")
