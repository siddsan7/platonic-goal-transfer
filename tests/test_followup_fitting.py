from pgt.followup.fitting import topology_role
from pgt.world import N


def test_topology_role_is_deterministic_and_d4_invariant():
    walls = frozenset({(0, 1), (2, 3), (4, 4), (1, 5)})
    mirrored = frozenset((r, N - 1 - c) for r, c in walls)
    assert topology_role(walls) in ("fit", "eval")
    assert topology_role(walls) == topology_role(sorted(walls)) == topology_role(mirrored)
