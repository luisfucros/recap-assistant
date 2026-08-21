"""Shared password-hashing config.

A single factory so every consumer that needs to hash a password — the API's
``AuthService`` and the bootstrap-admin data migration (which has no other
route to application code) — hashes with the identical algorithm and
parameters, rather than each constructing its own ``PasswordHash``.
"""

from pwdlib import PasswordHash


def build_password_hash() -> PasswordHash:
    """Return the app's configured password hasher (argon2, recommended params)."""
    return PasswordHash.recommended()
