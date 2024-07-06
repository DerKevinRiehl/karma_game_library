# Karma Game Library, Copyrights Kevin Riehl 2023, <kriehl@ethz.ch>

# Imports
import numpy as np
import pickle
from typing import List, Dict, Callable
import karma_game_library

# Classes

class GameParameters:
    """
    A class to encapsulate all parameters necessary to describe a Karma
    game for policy computation and simulation purposes. It is used when
    instantiating StateDistribution and Policy objects, as well as it is used
    for Simulator and Optimizer.
    """
    
    ##### Parameters ####
        # scalars
    _num_agents = 0
    _num_participants = 0
    _num_types = 0
    _num_urgencies = 0
    _num_actions = 0 
    _num_state_karmas = 0 
        # sets
    _set_types = ()
    _set_urgencies = ()
    _set_state_karmas = ()
    _set_actions = ()
    _set_outcomes = ()
        # maps
    _map_type_temp_preference = {}
    _map_urgency_immediate_cost = {}
        # lsts of initial population
    _lst_init_types = []
    _lst_init_urgencies = []
    _lst_init_karmas = []
        # functions
    _func_cost = None
    _func_outcome = None
    _func_payment = None
    _func_overflow_distribution = None
    _func_karma_redistribution = None
    _func_urgency_transition = None
        # tollerances & thresholds
    _validation_tol_up = 1.005
    _validation_tol_dn = 0.995
    _creation_thr_state = 0.0005
    _creation_thr_action = 0.001
    _autocorrect_factor_state = 1.001
    ##### Constructor ####
    def __init__(self, 
                 num_agents : int, 
                 num_participants : int,
                 num_types : int, 
                 num_urgencies : int, 
                 num_outcomes : int,
                 num_average_karma : int,
                 map_type_temp_preference : Dict[int, float], 
                 lst_init_types : List[int], 
                 lst_init_urgencies : List[int], 
                 lst_init_karmas : List[int],
                 func_cost: Callable,
                 func_outcome: Callable, 
                 func_payment: Callable,
                 func_urgency_transition: Callable,
                 func_overflow_distribution: Callable,
                 func_karma_redistribution: Callable):
        """
        This constructor initializes the GameParameters instance and checks
        the validity of the given parameters.

        Parameters
        ----------
        num_agents : int
            The number agents in the population (the population size).
        num_participants : int
            The number of agents that participate in an interaction.
        num_types : int
            The finite number of possible agent types.
        num_urgencies : int
            The finite number of possible agent urgencies.
        num_outcomes : int
            The finite number of possible outcomes for a participant.
        num_average_karma : int
            The initial state is focussed on num_average_karma. The average 
            Karma will remain constant.
        map_type_temp_preference : Dict[int, float]: [type] -> [temporal preference]
            This map maps the given agent type to a temporal preference between 
            0.0 and 1.0.
        lst_init_types : List[int]
            This list of integers represents the list of agent types. 
            The agent types (temporal preferences) cannot change over time.
        lst_init_urgencies : List[int]
            This list of integers represents the list of initial agent 
            urgencies.
        lst_init_karmas : List[int]
            This list of integers represents the list of initial agents' 
            karma balances.
        func_cost : Callable, (urgency, outcome) -> (cost)
            This function takes the urgency and outcome of a participant and
            returns the costs that result of to it. The cost should be a
            positive number as the software framework works with negative 
            costs.
        func_outcome : Callable, (lst_actions) -> (outcomes)
            This function takes the interaction participant actions and returns
            an integers representing the interaction outcome. The outcomes is
            a vector of outcomes for each participant.
        func_payment : Callable, (actions, outcomes) -> (payments, karma_overflow)
            This function takes the participants actions and interaction 
            outcomes and returns the lists of payments and karma_overflow. 
            Note, payments are added to the karma balances of participants.
            Note2: This function cannot be probabilistic!
        func_urgency_transition : Callable, (atype, urgency, outcome, set_urgencies) -> (urgency_next)
            This function takes an agent type, urgency, and outcome, and 
            returns the next agent urgency. This enables the user to model the 
            urgency_next in dependence of outcome. The outcome is encoded as 
            follows: -1 means the agent didnt participate an interaction this 
            epoch, otherwise it is the max outcome of all interactions from
            this epoch the agent participated. One could model the urgency 
            transition with type specific finite state machines for instance.
        func_overflow_distribution : Callable, (pop_karmas, karma_overflow) -> (pop_karmas_next)
            This function takes the population karma balances, and the payment 
            overflow, and distributes it across the population.
        func_karma_redistribution : Callable, (pop_karmas) -> (pop_karmas_next)
            This function takes the population (represented by karma balances) 
            and returns a list of integers representing the new karma balances 
            of agents after redistribution.

        Raises
        ------
        Exception
            In case the input parameters are not plausible a specific exception 
            will be raised.
            
        Returns
        -------
        None
        """
        # Plausibility checks before
        karma_game_library.check.non_negative(num_agents, "num_agents")
        karma_game_library.check.non_negative(num_participants, "num_participants")
        karma_game_library.check.non_negative(num_types, "num_types")
        karma_game_library.check.non_negative(num_urgencies, "num_urgencies")
        karma_game_library.check.non_negative(num_average_karma, "num_average_karma")
        karma_game_library.check.map_values_bounded(map_type_temp_preference, 0.0, 1.0, "map_type_temp_preference")
        karma_game_library.check.list_length(lst_init_types, num_agents, "lst_init_types")
        karma_game_library.check.list_length(lst_init_urgencies, num_agents, "lst_init_urgencies")
        karma_game_library.check.list_length(lst_init_karmas, num_agents, "lst_init_karmas")
        karma_game_library.check.func_not_none(func_outcome, "func_outcome")
        karma_game_library.check.func_not_none(func_payment, "func_payment")
        karma_game_library.check.func_not_none(func_urgency_transition, "func_urgency_transition")
        karma_game_library.check.func_not_none(func_overflow_distribution, "func_overflow_distribution")
        karma_game_library.check.func_not_none(func_karma_redistribution, "func_karma_redistribution")
        # Setting parameters
            # scalars
        self._num_agents = num_agents
        self._num_participants = num_participants
        self._num_types = num_types
        self._num_urgencies = num_urgencies
        self._num_actions = num_average_karma 
        self._num_state_karmas = num_average_karma*4
        self._num_average_karma = num_average_karma
            # sets
        self._set_types = np.arange(num_types)
        self._set_urgencies = np.arange(num_urgencies)
        self._set_state_karmas = np.arange(self._num_state_karmas)
        self._set_actions = np.arange(self._num_actions)
        self._set_outcomes = np.arange(num_outcomes)
            # maps
        self._map_type_temp_preference = map_type_temp_preference
            # lsts of initial population
        self._lst_init_types = lst_init_types
        self._lst_init_urgencies = lst_init_urgencies
        self._lst_init_karmas = lst_init_karmas
            # functions
        self._func_cost = func_cost
        self._func_outcome = func_outcome
        self._func_payment = func_payment
        self._func_urgency_transition = func_urgency_transition
        self._func_overflow_distribution = func_overflow_distribution
        self._func_karma_redistribution = func_karma_redistribution
        # Plausibility checks after
        karma_game_library.check.map_keys(map_type_temp_preference, self._set_types, "map_type_temp_preference", "set_types")
    
    def _enlarge_karma_space(self):
        """
        This function enlarges the karma space in case it is computationally
        required.
        
        Parameters
        ----------
        None

        Returns
        -------
        None
        """
        self._set_state_karmas = np.append(self._set_state_karmas, self._num_state_karmas)
        self._num_state_karmas += 1
       
    def _enlarge_action_space(self):
        """
        This function enlarges the action space in case it is computationally
        required.
        
        Parameters
        ----------
        None

        Returns
        -------
        None
        """
        self._set_actions = np.append(self._set_actions, self._num_actions)
        self._num_actions += 1
       


def save(parameters: GameParameters, target_file: str):
    """
    This function stores the GameParameters to a file.

    Parameters
    ----------
    parameters : GameParameters
        A GameParameters instance.
    target_file : str
        The file to store the GameParameters.

    Returns
    -------
    None
    """
    file_writer = open(target_file, "wb")
    pickle.dump(parameters, file_writer, pickle.HIGHEST_PROTOCOL)
    file_writer.close()

def load(source_file: str):
    """
    This function loads the GameParameters from a file.

    Parameters
    ----------
    source_file : str
        The file to load the GameParameters from.

    Returns
    -------
    parameters : GameParameters
        A GameParameters instance.
    """
    file_reader = open(source_file, 'rb')
    parameters = pickle.load(file_reader)
    file_reader.close()
    return parameters

GameParameters.save = staticmethod(save)
GameParameters.load = staticmethod(load)