"""Spoiler-safe effective-setting resolution (FR-18).

Spoiler-safe is a layered setting: a user has a global default
(``users.spoiler_safe``), each document may override it
(``reading_progress.spoiler_safe``, nullable), and a single request may override
both (a per-query flag). This module holds the one pure function that collapses
those three inputs into the effective boolean, so retrieval, memory recall, and
the output guardrail all decide spoiler-safety the same way.
"""


def resolve_spoiler_safe(
    *, per_query: bool | None, per_document: bool | None, user_default: bool
) -> bool:
    """Resolve whether spoiler-safe is in effect for this request.

    Precedence (most specific wins): an explicit per-query override, else the
    per-document override if set, else the user's global default.

    Args:
        per_query: A one-off override for this request (``None`` if not given).
        per_document: The document's stored override (``None`` = defer to user).
        user_default: The user's global spoiler-safe setting.

    Returns:
        ``True`` when spoiler-safe protection should apply.
    """
    if per_query is not None:
        return per_query
    if per_document is not None:
        return per_document
    return user_default
