"""Health check endpoint — used by compose healthchecks and load balancers."""

from __future__ import annotations

from fastapi import APIRouter
from pydantic import BaseModel

router = APIRouter(tags=["ops"])


class HealthResponse(BaseModel):
    status: str
    environment: str


@router.get("/healthz", response_model=HealthResponse)
async def healthz(environment: str = "development") -> HealthResponse:
    """Return 200 when the service is ready to handle requests."""
    return HealthResponse(status="ok", environment=environment)
