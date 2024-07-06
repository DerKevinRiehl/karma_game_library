# Karma Game Library, Copyrights Kevin Riehl 2023, <kriehl@ethz.ch>

# Imports
from typing import List

# Functions

def none(pop_karmas: List[int]) -> List[int]:
    """
    This function does not redistribute Karma.
    """
    pop_karmas_next = pop_karmas
    return pop_karmas_next