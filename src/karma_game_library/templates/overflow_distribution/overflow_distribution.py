# Karma Game Library, Copyrights Kevin Riehl 2023, <kriehl@ethz.ch>

# Imports
import numpy as np
import random
from typing import List

# Functions

def none(pop_karmas: List[int], overflow: int) -> List[int]:
    """
    This function does not distribute any karma overflow.
    """
    pop_karmas_next = pop_karmas
    return pop_karmas_next
    
def random_individual_all(pop_karmas: List[int], overflow: int) -> List[int]:
    """
    This function distributes the whole Karma overflow to one randomly chosen 
    individual.
    """
    pop_karmas_next = pop_karmas
    overflow_receiver = np.random.randint(0, len(pop_karmas_next))
    pop_karmas_next[overflow_receiver] += overflow
    return pop_karmas_next
    
def random_group_each_one(pop_karmas: List[int], overflow: int) -> List[int]:
    """
    This function distributes the Karma overflow randomly two n agents, where
    each agent receives one unit of Karma.
    """
    pop_karmas_next = pop_karmas
    overflow_receivers = random.choice([x for x in range(0, len(pop_karmas))], size=overflow, replace=False)
    for receiver in overflow_receivers:
        pop_karmas_next[receiver] += 1
    return pop_karmas_next