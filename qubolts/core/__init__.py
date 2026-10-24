"""Shared core: the resolver contract, QUBO helpers, routing, solving and run logs."""

from .contract import Resolver, Solution
from .qubo import add_budget_constraint, default_penalty, is_slack, objective_scale
from .router import RoutingPolicy, choose_backend
from .runlog import RunLog
from .solve import solve

__all__ = [
    "Resolver",
    "RoutingPolicy",
    "RunLog",
    "Solution",
    "add_budget_constraint",
    "choose_backend",
    "default_penalty",
    "is_slack",
    "objective_scale",
    "solve",
]
