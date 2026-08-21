"""Outbox event type names, shared by producers (API) and consumers (ingestion).

Centralized so the string an event is written with is exactly the string a relay
dispatches on — the two services can't import each other, so a shared constant is
the contract between them.
"""

from typing import Final

# Emitted by the API when a document is uploaded; consumed by the ingestion relay
# to enqueue the parse→chunk→embed→index task.
DOCUMENT_UPLOADED: Final = "document.uploaded"

# Emitted by the ingestion service once a document is durably indexed; consumed
# by downstream memory-indexing (a later milestone).
DOCUMENT_INDEXED: Final = "document.indexed"
