"""Scheduling qubolt (planned): time-slot jobs under rate limits and freshness SLAs.

Two uses with the same formulation
    * audience refreshes and syncs to ad platforms under API rate limits
    * journey / send-time scheduling for email, push and SMS under contact caps

Decision
    ``x_{j,t}`` = job (or message batch) ``j`` starts in slot ``t``. One-hot per job.

Objective (minimize)
    Staleness and lateness: ``sum_j sum_t lateness_j(t) x_{j,t}``, plus a
    preference term (best send-time per segment, off-peak API windows).

Constraints
    * capacity per slot per platform: ``sum_j load_j x_{j,t} <= cap_{p,t}``
    * dependencies: ``j`` after ``i`` (pairwise penalty on violating slot pairs)
    * contact-frequency caps per user segment

Inputs from Spark
    Job catalogue with durations and loads, platform rate limits, SLAs, history.
"""

from qubolts.core.planned import PlannedResolver


class Scheduling(PlannedResolver):
    name = "mktg.scheduling"
