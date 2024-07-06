# Karma Game Library, Copyrights Kevin Riehl 2023, <kriehl@ethz.ch>

# Imports
from typing import List

# Functions

    # PAY HIGHEST BID
def highest_bid_to_peer(actions: List[int], outcomes: List[int]) -> List[int]:
    """
    The highest bid (action) is paid by the winner to the other peer.
    Only for Karma Games with exactly two participants per interaction.
    """
    if(len(actions)!=2):
        raise Exception("This payment function is only compatible for exactly two participants!")
    payments = []
    highest_bid = max(actions)
    for n in range(0, len(actions)):
        if(outcomes[n]==1):
            payments.append(-highest_bid)
        else:
            payments.append(+highest_bid)
    overflow = 0
    return payments, overflow

def highest_bid_to_society(actions: List[int], outcomes: List[int]) -> List[int]:
    """
    The highest bid (action) is paid by the winner to the society (Karma overflow).
    """
    payments = []
    highest_bid = max(actions)
    for n in range(0, len(actions)):
        if(outcomes[n]==1):
            payments.append(-highest_bid)
        else:
            payments.append(0)
    overflow = +highest_bid
    return payments, overflow

    # PAY SECOND HIGHEST BID
def second_highest_bid_to_peer(actions: List[int], outcomes: List[int]) -> List[int]:
    """
    The second highest bid (action) is paid by the winner to the other peer.
    Only for Karma Games with exactly two participants per interaction.
    """
    if(len(actions)!=2):
        raise Exception("This payment function is only compatible for exactly two participants!")
    payments = []
    second_highest_bid = sorted(set(actions))[-2]
    for n in range(0, len(actions)):
        if(outcomes[n]==1):
            payments.append(-second_highest_bid)
        else:
            payments.append(+second_highest_bid)
    overflow = 0
    return payments, overflow

def second_highest_bid_to_society(actions: List[int], outcomes: List[int]) -> List[int]:
    """
    The second highest bid (action) is paid by the winner to the society (Karma overflow).
    """
    payments = []
    second_highest_bid = sorted(set(actions))[-2]
    for n in range(0, len(actions)):
        if(outcomes[n]==1):
            payments.append(-second_highest_bid)
        else:
            payments.append(0)
    overflow = +second_highest_bid
    return payments, overflow

    # PAY DIFFERENCE OF TWO HIGHEST BIDS
def bid_difference_to_peer(actions: List[int], outcomes: List[int]) -> List[int]:
    """
    The winner pays the difference of highest and second highest bid to the 
    other peer. Only for Karma Games with exactly two participants per interaction.
    """
    if(len(actions)!=2):
        raise Exception("This payment function is only compatible for exactly two participants!")
    payments = []
    highest_bid = max(actions)
    second_highest_bid = sorted(set(actions))[-2]
    bid_difference = highest_bid - second_highest_bid
    for n in range(0, len(actions)):
        if(outcomes[n]==1):
            payments.append(-bid_difference)
        else:
            payments.append(+bid_difference)
    overflow = 0
    return payments, overflow

def bid_difference_to_society(actions: List[int], outcomes: List[int]) -> List[int]:
    """
    The winner pays the difference of highest and second highest bid to the 
    society (Karma overflow).
    """
    payments = []
    highest_bid = max(actions)
    second_highest_bid = sorted(set(actions))[-2]
    bid_difference = highest_bid - second_highest_bid
    for n in range(0, len(actions)):
        if(outcomes[n]==1):
            payments.append(-bid_difference)
        else:
            payments.append(0)
    overflow = +bid_difference
    return payments, overflow
    
    # PAY ONE
def one_to_peer(actions: List[int], outcomes: List[int]) -> List[int]:
    """
    The winner pays exactly one Karma unit to the other peer.
    Only for Karma Games with exactly two participants per interaction.
    """
    if(len(actions)!=2):
        raise Exception("This payment function is only compatible for exactly two participants!")
    payments = []
    for n in range(0, len(actions)):
        if(outcomes[n]==1):
            payments.append(-1)
        else:
            payments.append(+1)
    overflow = 0
    return payments, overflow

def one_to_society(actions: List[int], outcomes: List[int]) -> List[int]:
    """
    The winner pays exactly one Karma unit to the society (Karma overflow).
    """
    payments = []
    for n in range(0, len(actions)):
        if(outcomes[n]==1):
            payments.append(-1)
        else:
            payments.append(0)
    overflow = +1
    return payments, overflow