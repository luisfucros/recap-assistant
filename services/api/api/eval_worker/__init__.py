"""Eval Celery worker: background scoring of evaluation dataset runs (FR-12.5).

Lives in the API package because a run calls ``AgentService`` / LangGraph.
Ingestion workers must not consume this queue.
"""
