# Karma Game Library, Copyrights Kevin Riehl 2023, <kriehl@ethz.ch>

# Imports
import numpy as np
import karma_game_library
from typing import List
import numpy.typing as npt

# Functions

def _support_func_prob_karma_transition_highest_bid_to_peer_no_redistribution(karma_next: int, karma: int, actions: List[int], outcome: int) -> float:
    action = actions[0]
    other_action = actions[1]
    if(outcome==1): # receive resource but loose karma
        karma_next_fac = karma - action
        if(karma_next==karma_next_fac):
            return 1.0
    if(outcome==0): # dont receive resource but gain karma
        karma_next_fac = karma + other_action
        if(karma_next==karma_next_fac):
            return 1.0
    return 0.0

def highest_bid_to_peer_no_redistribution(karma_next: int, karma: int, action: int, outcome: int, state_mat: npt.NDArray, policy_mat: npt.NDArray, game_parameters, gamma_mat: npt.NDArray, v_mat: npt.NDArray) -> float:
    """
    This function returns the probability to reach a karma_next balance given a karma, action and outcome for the case
    that the highest bidding peer wins, pays the bid to the peer, and there is no Karma redistribution.
    """
    prob = 0
    for other_action in game_parameters._set_actions:
        prob += v_mat[other_action] * karma_game_library.templates.p_outcome.highest_bid(outcome, [action, other_action]) * _support_func_prob_karma_transition_highest_bid_to_peer_no_redistribution(karma_next, karma, [action, other_action], outcome) 
    if(gamma_mat[outcome][action] != 0):
        prob = prob / gamma_mat[outcome][action]
    return prob

def _support_func_average_highest_bid(parameters, gamma_mat: npt.NDArray, state_mat: npt.NDArray, policy_mat: npt.NDArray) -> float:
    average_bid = 0
    for atype in parameters._set_types:
        for urgency in parameters._set_urgencies:
            for karma in parameters._set_state_karmas:
                for action in parameters._set_actions:
                    average_bid += state_mat[atype][urgency][karma] * policy_mat[atype][urgency][karma][action] * gamma_mat[1][action] * action
    return average_bid

def highest_bid_to_society_no_redistribution(karma_next: int, karma: int, action: int, outcome: int, state_mat: npt.NDArray, policy_mat: npt.NDArray, game_parameters, gamma_mat: npt.NDArray, v_mat: npt.NDArray) -> float:
    """
    This function returns the probability to reach a karma_next balance given a karma, action and outcome for the case
    that the highest bidding peer wins, pays the bid to the society (Karma overflow), and there is no Karma redistribution.
    """
    average_bid = _support_func_average_highest_bid(game_parameters, gamma_mat, state_mat, policy_mat)
    lower_av_bid = np.floor(average_bid)
    higher_av_bid = np.ceil(average_bid)
    if((higher_av_bid - lower_av_bid)==0):
        fraction = 0.5
    else:
        fraction = (average_bid - lower_av_bid) / (higher_av_bid - lower_av_bid)
    if(outcome==1):
        if(karma_next == karma - action + lower_av_bid):
            return 1-fraction
        elif(karma_next == karma - action + higher_av_bid):
            return fraction
    if(outcome==0):
        if(karma_next == karma + lower_av_bid):
            return 1-fraction
        elif(karma_next == karma + higher_av_bid):
            return fraction
    return 0

