from typing import Annotated

from fastapi import APIRouter, BackgroundTasks, Depends, status
from fastapi.exceptions import RequestValidationError
from pydantic import ValidationError

from app.api.deps import get_container
from app.api.schemas import (
    ChatRequest,
    ChatResponse,
    CreateUserRequest,
    MemoryOut,
    ProfileFields,
    ProfileOut,
    QueryAnalysisOut,
)
from app.container import Container
from app.core.errors import NotFoundError
from app.profile.service import ProfileUpdate

router = APIRouter()
ContainerDep = Annotated[Container, Depends(get_container)]


def _to_update(fields: ProfileFields) -> ProfileUpdate:
    # Domain-level validation (e.g. DOB not in the future) surfaces as a normal 422.
    try:
        return ProfileUpdate(**fields.model_dump(exclude_none=True))
    except ValidationError as exc:
        raise RequestValidationError(exc.errors()) from exc


@router.get("/health", tags=["ops"])
async def health(c: ContainerDep):
    graph_ok = await c.repo.ping()
    return {
        "status": "ok" if graph_ok else "degraded",
        "brain_backend": c.settings.brain_backend,
        "graph": graph_ok,
        "llm_providers": getattr(c.llm, "provider_names", [c.llm.name]),
    }


@router.post("/chat", response_model=ChatResponse, tags=["chat"])
async def chat(body: ChatRequest, background: BackgroundTasks, c: ContainerDep):
    result = await c.orchestrator.handle(body.user_id, body.session_id, body.message)

    memories_written, memory_update = [], "skipped"
    if result.memory_job:
        if c.settings.memory_write_mode == "background":
            background.add_task(c.orchestrator.update_memory, result.memory_job)
            memory_update = "scheduled"
        else:
            memories_written = await c.orchestrator.update_memory(result.memory_job)
            memory_update = "applied"

    a = result.analysis
    return ChatResponse(
        response=result.response,
        user_id=result.user_id,
        session_id=result.session_id,
        context_used=result.context_used,
        memories_written=memories_written,
        memory_update=memory_update,
        degraded=result.degraded,
        degraded_reasons=result.degraded_reasons,
        query_analysis=QueryAnalysisOut(
            intent=a.intent.value,
            life_areas=[x.value for x in a.life_areas],
            is_follow_up=a.is_follow_up,
            is_memory_query=a.is_memory_query,
        ),
    )


@router.post("/users", response_model=ProfileOut, status_code=status.HTTP_201_CREATED, tags=["users"])
async def create_user(body: CreateUserRequest, c: ContainerDep):
    """Create a user with an optional profile. Idempotent: re-posting merges fields."""
    profile, _ = await c.profiles.apply(body.user_id, _to_update(body))
    return ProfileOut.of(profile)


@router.get("/users/{user_id}", response_model=ProfileOut, tags=["users"])
async def get_user(user_id: str, c: ContainerDep):
    profile = await c.profiles.get(user_id)
    if not profile:
        raise NotFoundError(f"user {user_id} not found")
    return ProfileOut.of(profile)


@router.patch("/users/{user_id}", response_model=ProfileOut, tags=["users"])
async def update_user(user_id: str, body: ProfileFields, c: ContainerDep):
    if not await c.profiles.get(user_id):
        raise NotFoundError(f"user {user_id} not found")
    profile, _ = await c.profiles.apply(user_id, _to_update(body))
    return ProfileOut.of(profile)


@router.get("/users/{user_id}/memories", response_model=list[MemoryOut], tags=["memory"])
async def list_memories(user_id: str, c: ContainerDep, include_inactive: bool = False):
    """Inspect the Shared Brain for a user (debugging, transparency, evaluation)."""
    items = await c.repo.list_memories(user_id, include_inactive=include_inactive)
    return [MemoryOut.of(m) for m in items]


@router.delete("/users/{user_id}/memories", tags=["memory"])
async def forget_memories(user_id: str, c: ContainerDep):
    """Right to be forgotten: hard-deletes all long-term memories for the user."""
    return {"deleted": await c.repo.forget_memories(user_id)}
