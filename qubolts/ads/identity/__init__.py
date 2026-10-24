"""Identity qubolt (planned): resolve ambiguous identity clusters as graph partitioning.

Scope
    Only the *hard* local clusters left after classical blocking and
    deterministic matching (shared hashed email, login IDs). Never the full
    graph: production identity graphs are orders of magnitude too large.

Decision
    Correlation clustering: ``x_{u,k}`` = node ``u`` (device, cookie, hashed
    email, MAID) belongs to person/household ``k``, one-hot per node.

Objective (minimize disagreement)
    For each candidate edge with match confidence ``w_uv`` (positive = likely
    same person, negative = likely different): reward placing positive edges in
    the same cluster and negative edges apart.

Constraints
    One-hot per node; optional cluster-size caps (a household rarely has 40 devices).

Inputs from Spark
    Candidate edge list with calibrated match scores, after blocking.
"""

from qubolts.core.planned import PlannedResolver


class Identity(PlannedResolver):
    name = "ads.identity"
