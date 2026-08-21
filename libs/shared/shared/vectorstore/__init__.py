"""Qdrant-backed vector storage for document chunks and long-term memory.

Wraps the Qdrant client behind small, purpose-built stores so callers never hand-
roll payloads or filters — and so the **per-user isolation invariant** (every
payload carries ``user_id`` and every search/delete is filtered by it) lives in
one place.
"""

from shared.vectorstore.chunks import (
    ChunkVectorStore,
    ScoredChunk,
    build_chunk_payload,
    chunk_point_id,
)
from shared.vectorstore.memory import (
    MemoryVectorStore,
    ScoredMemory,
    build_memory_payload,
    memory_point_id,
)

__all__ = [
    "ChunkVectorStore",
    "MemoryVectorStore",
    "ScoredChunk",
    "ScoredMemory",
    "build_chunk_payload",
    "build_memory_payload",
    "chunk_point_id",
    "memory_point_id",
]
