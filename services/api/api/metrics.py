"""HTTP metrics instrumentation for the API service.

Adds request latency/throughput/in-flight/error metrics and exposes ``/metrics``
(the Prometheus scrape target) at the root — not under the ``/api/v1`` prefix, so
scrape config stays independent of the API version. Importing the shared metrics
module here registers the custom RAG metrics on the default registry, so they
appear on ``/metrics`` even before their first observation.
"""

from fastapi import FastAPI
from prometheus_fastapi_instrumentator import Instrumentator

import shared.observability.metrics  # noqa: F401 — registers custom metrics on import


def setup_metrics(app: FastAPI) -> None:
    """Instrument HTTP metrics and expose ``GET /metrics`` on ``app``.

    Always-on and dependency-free. Serves HTTP metrics plus the shared custom RAG
    metrics and process CPU/memory, all from the default Prometheus registry.

    Note: build one app per process (the production path). The instrumentator's
    metrics register on the default registry, so constructing many apps in a
    single process (e.g. a function-scoped test fixture) would double-register;
    tests use a module-scoped app.
    """
    Instrumentator().instrument(app).expose(app, endpoint="/metrics", include_in_schema=False)
