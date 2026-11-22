"""Pure local activation model shared by bounded planners."""
from .._core.types import Coord

def _activate_local_pending_nodes(
    position: Coord,
    mask: int,
    *,
    base_activated: set[str],
    pending_nodes: tuple[str, ...],
    pending_index: dict[str, int],
    cell_to_nodes: dict[Coord, tuple[str, ...]],
    prerequisites: dict[str, list[set[str]]],
) -> int:
    """Mirror the environment's local activation closure for pending nodes."""

    active = set(base_activated)
    for idx, node in enumerate(pending_nodes):
        if mask & (1 << idx):
            active.add(node)

    local_nodes = cell_to_nodes.get(position, ())
    out_mask = mask
    while True:
        candidates: list[str] = []
        for node in local_nodes:
            if node in active:
                continue
            prerequisite_options = prerequisites.get(node, [set()])
            if any(prerequisite_option.issubset(active) for prerequisite_option in prerequisite_options):
                candidates.append(node)
        if not candidates:
            return out_mask
        for node in candidates:
            active.add(node)
            out_mask |= 1 << pending_index[node]


