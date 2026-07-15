from __future__ import annotations

from .models import Team


def squad_key(team: Team, squad_id: str) -> str:
    """Return the canonical key for all squad-scoped state."""
    return f"{team.value}:{squad_id}"


__all__ = ["squad_key"]
