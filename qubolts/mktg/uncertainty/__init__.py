"""Uncertainty qubolt (future): MMM and attribution confidence intervals and scenarios.

Why it is here
    Quantum amplitude estimation offers a quadratic speedup over classical Monte
    Carlo for estimating expectations (e.g. ROI distributions under budget
    scenarios). It needs fault-tolerant hardware that does not exist yet, so
    this module defines the interface and ships a classical Monte Carlo path
    when someone implements it.

Near-term contribution ideas
    * QUBO-based *feature selection* for MMM (which channels / lags / controls
      to include), a combinatorial problem current annealers can handle today
      (TNO reported early gains with this approach for ad campaign impact).
    * Synthetic-control donor selection as a QUBO.

Inputs from Spark
    Model posteriors or bootstrap samples, scenario definitions.
"""

from qubolts.core.planned import PlannedResolver


class Uncertainty(PlannedResolver):
    name = "mktg.uncertainty"
    status = "future"
