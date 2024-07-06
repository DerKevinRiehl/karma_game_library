# Karma Game Library, Copyrights Kevin Riehl 2023, <kriehl@ethz.ch>

# Imports
import numpy as np
from typing import List

# Functions

def highest_bid(actions: List[int]) -> List[int]:
    """
    The highest bid (action) wins. Tiebreaker: random.
    """
    highest_bid = max(actions)
    highest_bid_participants = [x==highest_bid for x in actions]
    winner = np.random.randint(0, sum(highest_bid_participants))
    winner_idx = 0
    for n in range(0, len(actions)):
        if(highest_bid_participants[n]):
            if(winner_idx==winner):
                winner_idx = n
                break
            else:
                winner_idx += 1
    outcomes = []
    for n in range(0,len(actions)):
        if(not winner_idx == n):
            outcomes.append(0)
        else:
            outcomes.append(1)
    return outcomes

def second_highest_bid(actions: List[int]) -> List[int]:
    """
    The second highest bid (action) wins. Tiebreaker: random.
    """
    second_highest_bid = sorted(set(actions))[-2]
    second_highest_bid_participants = [x==second_highest_bid for x in actions]
    winner = np.random.randint(0, sum(second_highest_bid_participants))
    winner_idx = 0
    for n in range(0, len(actions)):
        if(second_highest_bid_participants[n]):
            if(winner_idx==winner):
                winner_idx = n
                break
            else:
                winner_idx += 1
    outcomes = []
    for n in range(0,len(actions)):
        if(not winner_idx == n):
            outcomes.append(0)
        else:
            outcomes.append(1)
    return outcomes
