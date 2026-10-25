import logging

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from app.core.errors import GraphUnavailableError, NotFoundError, SessionOwnershipError

logger = logging.getLogger(__name__)


def _error(status: int, code: str, message: str) -> JSONResponse:
    return JSONResponse(status_code=status, content={"error": {"code": code, "message": message}})


def register_error_handlers(app: FastAPI) -> None:
    @app.exception_handler(NotFoundError)
    async def not_found(_: Request, exc: NotFoundError):
        return _error(404, "not_found", str(exc))

    @app.exception_handler(SessionOwnershipError)
    async def session_conflict(_: Request, exc: SessionOwnershipError):
        return _error(409, "session_conflict", str(exc))

    @app.exception_handler(GraphUnavailableError)
    async def graph_down(_: Request, exc: GraphUnavailableError):
        # Only reached by endpoints that cannot degrade (profile/memory admin); /chat degrades.
        return _error(503, "graph_unavailable", "The memory store is temporarily unavailable.")

    @app.exception_handler(Exception)
    async def unhandled(_: Request, exc: Exception):
        logger.exception("unhandled error")
        return _error(500, "internal_error", "Something went wrong.")
