"""Coverage qubolt (planned): maximize unique cross-platform reach under frequency caps.

Decision
    Which publishers / platforms / programs to buy, ``x_p`` binary (or a small
    spend ladder per publisher, as in the allocation qubolt).

Objective (maximize)
    Unique reach at an effective frequency. Second-order approximation:
    ``sum_p reach_p x_p - sum_{p<q} dup_{pq} x_p x_q``. For a *minimum effective
    frequency* target (reach people at least F times), add an indicator per
    audience cell, rewarded when enough selected publishers cover it.

Constraints
    Budget (``add_budget_constraint``), max publishers, channel mix bounds.

Inputs from Spark
    Reach estimates per publisher and a duplication matrix (panel data, clean
    room overlap counts, or sketch-based estimates such as HyperLogLog unions).

Prior art
    Recruit's TV ad allocation targeting Minimum Effective Frequency.
"""

from qubolts.core.planned import PlannedResolver


class Coverage(PlannedResolver):
    name = "ads.coverage"
