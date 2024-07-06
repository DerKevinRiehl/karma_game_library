# Karma Game Library, Copyrights Kevin Riehl 2023, <kriehl@ethz.ch>

# Imports
from typing import List

# Functions

def highest_bid(outcome: int, actions: List[int]) -> float:
    """
    The highest bid (action) wins. Tiebreaker: random.
    """
    action = actions[0]
    other_action = actions[1]
    if(action==other_action):
        if(outcome==0):
            return 0.5
        elif(outcome==1):
            return 0.5
    elif(action>other_action):
        if(outcome==0):
            return 0.0
        elif(outcome==1):
            return 1.0
    else:
        if(outcome==0):
            return 1.0
        elif(outcome==1):
            return 0.0

def second_highest_bid(outcome: int, actions: List[int]) -> float:
    """
    The second highest bid (action) wins. Tiebreaker: random.
    """
    action = actions[0]
    other_action = actions[1]
    if(action==other_action):
        if(outcome==0):
            return 0.5
        elif(outcome==1):
            return 0.5
    elif(action>other_action):
        if(outcome==0):
            return 1.0
        elif(outcome==1):
            return 0.0
    else:
        if(outcome==0):
            return 0.0
        elif(outcome==1):
            return 1.0