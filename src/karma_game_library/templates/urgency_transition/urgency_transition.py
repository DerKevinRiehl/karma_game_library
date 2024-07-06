# Karma Game Library, Copyrights Kevin Riehl 2023, <kriehl@ethz.ch>

# Imports
import numpy as np
from typing import Set

# Functions

def random(atype: int, urgency: int, outcome: int, set_urgencies: Set) -> int:
    """
    This function transitions randomly from one urgency level to the next.
    This happens independent from agent type, urgency level, and outcome.
    """
    urgency_next = np.random.choice(set_urgencies, size=1)
    if(outcome==-1): # didnt participate interaction
        return urgency_next
    elif(outcome==0): # participated interaction, received the resource
        return urgency_next
    else: # participated interaction, did not receive resource
        return urgency_next