"""Assignment qubolt (planned): match creatives to audiences and placements.

Decision
    ``x_{c,a}`` = creative ``c`` runs against audience (or placement) ``a``.

Objective (maximize)
    ``sum predicted_response(c, a) x_{c,a}`` minus a diversity / fatigue penalty
    when the same creative is assigned to many overlapping audiences:
    ``sum_{a<b} overlap_{ab} * fatigue_c * x_{c,a} x_{c,b}``.

Constraints
    * each audience gets between ``min_k`` and ``max_k`` creatives
    * each creative is used at most ``u_c`` times (rotation, legal approvals)
    * brand-safety exclusions: fixed ``x_{c,a} = 0``

Inputs from Spark
    Predicted CTR/CVR per (creative, audience), audience overlap, fatigue curves.
"""

from qubolts.core.planned import PlannedResolver


class Assignment(PlannedResolver):
    name = "ads.assignment"
