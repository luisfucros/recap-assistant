"""Evaluation service (FR-12): datasets, scorers, and the run orchestration.

``datasets/`` holds versioned, fixture-based question/answer/retrieval cases
(:mod:`api.evaluation.datasets.loader`); ``scorers/`` holds the pure retrieval
metrics and the LLM-as-judge answer-quality scorer. The orchestration itself —
seeding fixtures, running each case through retrieval + the agent, scoring, and
persisting a run — is ``api.services.evaluation_service.EvaluationService``,
alongside every other service.
"""
