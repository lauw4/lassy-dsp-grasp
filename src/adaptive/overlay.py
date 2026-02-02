"""
Overlay runtime state for adaptive DSP execution.

Manages component states: locked, destroyed, done, skipped.
"""
from typing import Any, Set, Dict, List, Optional
from src.core import graph_api

def init_state() -> dict[str, object]:
    """Initialize runtime overlay state."""
    return {
        "locked": set(),
        "destroyed": set(),
        "done": set(),
        "skipped": set(),
        "notes": []
    }

def mark_done(state: dict, comp_id: str) -> None:
    """Mark component as processed."""
    state["done"].add(comp_id)

def lock_node(state: dict, comp_id: str) -> None:
    """Lock component (temporarily inaccessible)."""
    state["locked"].add(comp_id)

def unlock_node(state: dict, comp_id: str) -> None:
    """Unlock component."""
    state["locked"].discard(comp_id)

def mark_destroyed(state: dict, comp_id: str) -> None:
    """Mark component as destroyed."""
    state["destroyed"].add(comp_id)

def is_locked(state: dict, comp_id: str) -> bool:
    """True if component is locked."""
    return comp_id in state["locked"]

def is_destroyed(state: dict, comp_id: str) -> bool:
    """True if component is destroyed."""
    return comp_id in state["destroyed"]

def is_done(state: dict, comp_id: str) -> bool:
    """True if component is already processed."""
    return comp_id in state["done"]

def feasible_now_excluding(state: dict, exclude: set[str] | None = None) -> List[str]:
    """Wrapper around core.graph_api.feasible_now_nodes."""
    if exclude is None:
        exclude = set()
    return graph_api.feasible_now_nodes(state, exclude)
