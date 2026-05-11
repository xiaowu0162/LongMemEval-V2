"""Compute LAFS gain against the released LongMemEval-V2 reference frontier."""

import math
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List


@dataclass(frozen=True)
class Point:
    name: str
    acc: float  # accuracy in percentage points, e.g. 74.9
    latency: float  # query latency in seconds


# ---------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------

T_MIN = 1.0
T_MAX = 200.0
FLOOR_ACC = 0.0


# ---------------------------------------------------------------------
# Hard-coded released reference frontier
# ---------------------------------------------------------------------
# The leaderboard reference frontier is baseline + AgentRunbook:
#
#   {RAG slice+notes, Codex, AgentRunbook-R, AgentRunbook-C}
#
# Values below are copied from the paper's main results table. Keep these
# hard-coded because downstream score computation depends on the exact
# released operating points.

fixed_frontier_points: Dict[str, List[Point]] = {
    "small": [
        Point("RAG: query -> slice + notes", acc=51.0, latency=0.2),
        Point("Codex", acc=69.9, latency=177.2),
        Point("AgentRunbook-R", acc=58.6, latency=26.9),
        Point("AgentRunbook-C", acc=74.9, latency=108.3),
    ],
    "medium": [
        Point("RAG: query -> slice + notes", acc=45.9, latency=0.3),
        Point("Codex", acc=68.7, latency=185.8),
        Point("AgentRunbook-R", acc=57.0, latency=25.8),
        Point("AgentRunbook-C", acc=70.1, latency=139.9),
    ],
}

# Backwards-compatible alias for scripts that still expect the old name.
full_frontier_points = fixed_frontier_points


# ---------------------------------------------------------------------
# Core functions
# ---------------------------------------------------------------------

def pareto_frontier(points: List[Point]) -> List[Point]:
    """
    Return non-dominated points sorted by latency.

    A point is useful only if it improves accuracy over all faster points.
    """
    if not points:
        return []

    # For identical latencies, keep the highest-accuracy point first.
    sorted_points = sorted(points, key=lambda p: (p.latency, -p.acc))

    frontier = []
    best_acc = -float("inf")

    for point in sorted_points:
        if point.latency <= 0:
            raise ValueError(
                f"Latency must be positive, got {point.latency} for {point.name}."
            )

        if point.acc > best_acc:
            frontier.append(point)
            best_acc = point.acc

    return frontier


def best_acc_under_budget(
    points: List[Point],
    budget: float,
    floor_acc: float = FLOOR_ACC,
) -> float:
    """Best accuracy among points with latency <= budget."""
    valid = [point.acc for point in points if point.latency <= budget]
    return max(valid) if valid else floor_acc


def lafs(
    points: List[Point],
    t_min: float = T_MIN,
    t_max: float = T_MAX,
    floor_acc: float = FLOOR_ACC,
) -> float:
    """
    Exact LAFS under log-latency integration.

    LAFS = average_T best_accuracy_under_budget(T),
    where T is uniformly distributed over log latency.
    """
    if t_min <= 0 or t_max <= 0 or t_min >= t_max:
        raise ValueError("Require 0 < t_min < t_max.")

    frontier = pareto_frontier(points)

    # Step-function breakpoints on the latency axis.
    breakpoints = {t_min, t_max}
    for point in frontier:
        if t_min < point.latency < t_max:
            breakpoints.add(point.latency)

    breakpoints = sorted(breakpoints)
    denom = math.log(t_max / t_min)

    area = 0.0
    for left, right in zip(breakpoints[:-1], breakpoints[1:]):
        acc = best_acc_under_budget(frontier, left, floor_acc=floor_acc)
        area += acc * math.log(right / left)

    return area / denom


def lafs_gain_from_frontiers(
    fixed_points: List[Point],
    full_points: List[Point],
    t_min: float = T_MIN,
    t_max: float = T_MAX,
    floor_acc: float = FLOOR_ACC,
) -> float:
    """LAFS gain when full_points already contains fixed frontier + new points."""
    return lafs(full_points, t_min, t_max, floor_acc) - lafs(
        fixed_points,
        t_min,
        t_max,
        floor_acc,
    )


def lafs_gain_for_submission(
    fixed_points: List[Point],
    submission_points: List[Point],
    t_min: float = T_MIN,
    t_max: float = T_MAX,
    floor_acc: float = FLOOR_ACC,
) -> float:
    """
    LAFS gain for a new submission S against a fixed frontier F_ref:

        LAFS(F_ref union S) - LAFS(F_ref)
    """
    return lafs(fixed_points + submission_points, t_min, t_max, floor_acc) - lafs(
        fixed_points, t_min, t_max, floor_acc
    )


def lafs_summary_for_submission(
    tier: str,
    submission_points: List[Point],
    t_min: float = T_MIN,
    t_max: float = T_MAX,
    floor_acc: float = FLOOR_ACC,
) -> Dict[str, Any]:
    if tier not in fixed_frontier_points:
        supported = ", ".join(sorted(fixed_frontier_points))
        raise ValueError(f"Unsupported tier {tier!r}; expected one of: {supported}")
    reference_points = fixed_frontier_points[tier]
    reference_lafs = lafs(reference_points, t_min, t_max, floor_acc)
    combined_lafs = lafs(
        reference_points + submission_points,
        t_min,
        t_max,
        floor_acc,
    )
    return {
        "tier": tier,
        "t_min_seconds": t_min,
        "t_max_seconds": t_max,
        "floor_accuracy": floor_acc,
        "accuracy_unit": "percentage_points",
        "reference_lafs": reference_lafs,
        "submission_lafs": combined_lafs,
        "lafs_gain": combined_lafs - reference_lafs,
        "reference_frontier": [
            {
                "name": point.name,
                "accuracy": point.acc,
                "latency_seconds": point.latency,
            }
            for point in pareto_frontier(reference_points)
        ],
        "submission_frontier": [
            {
                "name": point.name,
                "accuracy": point.acc,
                "latency_seconds": point.latency,
            }
            for point in pareto_frontier(reference_points + submission_points)
        ],
    }


def format_points(points: Iterable[Point]) -> str:
    return ", ".join(f"{point.acc:g} @ {point.latency:g}s" for point in points)


# ---------------------------------------------------------------------
# Example usage
# ---------------------------------------------------------------------

if __name__ == "__main__":
    reference = fixed_frontier_points["small"]

    examples = {
        # A faster RAG-style submission that improves the mid-latency region.
        "Fast RAG++": [Point("Fast RAG++", acc=62.0, latency=15.0)],
        # A coding-agent operating point that is faster than AgentRunbook-C but
        # not quite as accurate.
        "Efficient coding agent": [
            Point("Efficient coding agent", acc=73.5, latency=70.0)
        ],
        # A very slow high-accuracy point only helps near the largest budgets.
        "Slow high-accuracy": [Point("Slow high-accuracy", acc=78.0, latency=190.0)],
        # This point is dominated by AgentRunbook-C and should receive zero gain.
        "Dominated slow method": [
            Point("Dominated slow method", acc=70.0, latency=150.0)
        ],
        # Multiple operating points can improve different latency budgets.
        "Balanced 2-point method": [
            Point("Balanced fast point", acc=60.0, latency=10.0),
            Point("Balanced accurate point", acc=73.0, latency=55.0),
        ],
        # A stronger multi-point submission improves a wider frontier region.
        "Strong 3-point method": [
            Point("Strong fast point", acc=60.0, latency=8.0),
            Point("Strong middle point", acc=72.0, latency=45.0),
            Point("Strong accurate point", acc=76.0, latency=100.0),
        ],
    }

    print("Reference frontier: LME-V2-Small")
    for point in pareto_frontier(reference):
        print(f"  {point.name}: acc={point.acc:.1f}, latency={point.latency:.1f}s")
    print(f"Reference LAFS: {lafs(reference):.2f}\n")

    print("| Example system | Operating points | LAFS Gain |")
    print("|---|---:|---:|")
    for name, points in examples.items():
        gain = lafs_gain_for_submission(reference, points)
        print(f"| {name} | {format_points(points)} | {gain:+.2f} |")
