"""Long-term memory routes: view and delete stored memories (FR-4.5 privacy).

Handlers stay thin: reads/writes go through a ``LongTermMemoryRepository``
already scoped to the authenticated user, and :class:`MemoryService` owns the
join between Postgres (content, the source of truth) and the memory vector
store (the embedding point deleted alongside the row). A caller only ever
sees or deletes their own memories.
"""

import uuid
from typing import Annotated

from fastapi import APIRouter, Query, status

from api.deps import DbSession, MemoryRepositoryDep, MemoryServiceDep
from api.schemas import MemoryPage, MemoryPublic
from shared.core.enums import MemoryType

router = APIRouter(prefix="/memory", tags=["memory"])


@router.get("", response_model=MemoryPage, summary="List your stored memories")
async def list_memories(
    memories: MemoryRepositoryDep,
    memory_service: MemoryServiceDep,
    type: Annotated[
        MemoryType | None, Query(description="Restrict to one memory kind; omit for all kinds.")
    ] = None,
    page: int = Query(1, ge=1, description="1-based page number."),
    page_size: int = Query(10, ge=1, le=100, description="Items per page (max 100)."),
) -> MemoryPage:
    """Return a page of the caller's stored memories, newest first (FR-4.5).

    Covers saved preferences, facts, habits, FAQs, and page-range summaries.
    ``total`` is the caller's overall memory count, independent of ``type``
    (matching the other list routes' pagination envelope).
    """
    offset = (page - 1) * page_size
    items = await memory_service.list_memories(
        memories=memories, type=type, limit=page_size, offset=offset
    )
    total = await memories.count()
    return MemoryPage(
        items=[MemoryPublic.model_validate(item) for item in items],
        total=total,
        page=page,
        page_size=page_size,
    )


@router.delete(
    "/{memory_id}", status_code=status.HTTP_204_NO_CONTENT, summary="Delete a stored memory"
)
async def delete_memory(
    memory_id: uuid.UUID,
    memory_service: MemoryServiceDep,
    memories: MemoryRepositoryDep,
    session: DbSession,
) -> None:
    """Delete a memory and its vector point (FR-4.5). 404 if it isn't the caller's."""
    await memory_service.delete_memory(memories=memories, session=session, memory_id=memory_id)
