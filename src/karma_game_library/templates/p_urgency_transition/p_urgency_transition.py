# Karma Game Library, Copyrights Kevin Riehl 2023, <kriehl@ethz.ch>

# Imports
from typing import Set

# Functions

def random(urgency_next: int, urgency: int, outcome: int, atype: int, set_urgencies: Set) -> float:
    """
    This function assumes that the transition probability from one urgency level to another is random.
    It is completely independent from the agent type, urgency, and interaction outcome.
    """
    return 1/len(set_urgencies)