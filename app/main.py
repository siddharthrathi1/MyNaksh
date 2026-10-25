import logging
import time
import uuid
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request

from app.api.errors import register_error_handlers
from app.api.routes import router
from app.config import get_settings
from app.container import Container

logger = logging.getLogger("app")


def create_app(container: Container | None = None) -> FastAPI:
    """App factory. Tests pass a pre-built container (in-memory repo, fake LLM)."""
    settings = container.settings if container else get_settings()
    logging.basicConfig(
        level=settings.log_level, format="%(asctime)s %(levelname)s %(name)s: %(message)s"
    )
    # Idempotent schema statements emit "already exists" notices on every start.
    logging.getLogger("neo4j.notifications").setLevel(logging.WARNING)

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        c = container or Container.build(settings)
        app.state.container = c
        await c.startup()
        yield
        await c.shutdown()

    app = FastAPI(
        title="MyNaksh Shared Brain",
        version="1.0.0",
        description="Personalised astrology chat with short-term context and a graph-based long-term memory.",
        lifespan=lifespan,
    )
    register_error_handlers(app)
    app.include_router(router)

    @app.middleware("http")
    async def request_context(request: Request, call_next):
        # Request id for log correlation across the pipeline; echoed back to the client.
        request_id = request.headers.get("x-request-id") or uuid.uuid4().hex[:12]
        started = time.perf_counter()
        response = await call_next(request)
        response.headers["x-request-id"] = request_id
        logger.info(
            "http %s %s status=%s ms=%.0f rid=%s",
            request.method, request.url.path, response.status_code,
            (time.perf_counter() - started) * 1000, request_id,
        )
        return response

    return app


app = create_app()
