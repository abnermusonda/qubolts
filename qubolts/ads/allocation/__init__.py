"""Allocation qubolt (planned): split a budget across campaigns, channels and dayparts.

Decision
    For each line item ``i`` choose one spend level ``k`` from a discrete ladder
    (e.g. 0%, 10%, ... 100% of max). One-hot variables ``x_{i,k}``.

Objective (maximize)
    ``sum_{i,k} R_i(k) x_{i,k}`` where ``R_i`` is the predicted response curve
    (diminishing returns), minus cannibalization ``sum C_{ij} x_{i,k} x_{j,l}``
    between line items that reach the same users.

Constraints (penalties)
    * one-hot per line item: ``(sum_k x_{i,k} - 1)^2``
    * total budget: ``sum spend(k) x_{i,k} <= B`` (use ``add_budget_constraint``)
    * optional pacing bounds per day / per channel

Inputs from Spark
    Response curves per line item (from MMM or historical bid landscapes),
    pairwise audience overlap between line items, spend ladders, caps.

Prior art
    Recruit Communications + D-Wave showed QUBO budget pacing beating greedy
    allocation for display advertising.
"""

from qubolts.core.planned import PlannedResolver


class Allocation(PlannedResolver):
    name = "ads.allocation"
