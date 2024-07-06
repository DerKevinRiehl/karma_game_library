# Karma Game Library, Copyrights Kevin Riehl 2023, <kriehl@ethz.ch>

# Imports
import numpy as np
from karma_game_library import GameParameters, Policy, StateDistribution
from typing import List
import numpy.typing as npt

# Classes

class Simulator:
    """
    A class to provide entities and functionality to perform the simulation of
    a Karma game.

    Attributes
    ----------
    epoch_counter : int
        The current epoch of the game (starts counting from zero).
    last_participants : List[int]
        List of participating agents from last interaction.
    last_actions : List[int]
        List of participant actions from last interaction.
    last_outcome : List[int]
        List of outcomes from last interaction.
    all_epoch_participants : List[int]
        List of participants during all interactions of the current epoch.
    all_epoch_outcomes : List[int]
        List of the maximum of outcomes during all interactions of the current epoch for the participants in all_epoch_participants.
    """
    
    #### Attributes
        # Constants
    _TYPE_COL = 0
    _URGENCY_COL = 1
    _KARMA_COL = 2
    _CUM_COST_COL = 3
    _ENCOUNTERS_COL = 4
        # General
    _parameters = None
    epoch_counter = 0
    _open_epoch = False
    _total_karma_epoch_begin = 0
        # Population
    _population = [] # num_agents x 5 [type, urgency, karma, cum_cost, encounters]
    _karma_overflow = 0
        # Recording
    _state_transition_counter = [[]] # [urgency_next, karma_next, urgency_before, karma_before]
    _state_transition_counter_participants = [[]] # [urgency_next, karma_next, urgency_before, karma_before]
        # Epoch specific
    all_epoch_participants = []
    all_epoch_outcomes = []
        # Interaction specific
    last_participants = None
    last_actions = None
    last_outcomes = None
    
    #### Constructor
    def __init__(self, game_parameters: GameParameters, state : StateDistribution, steady_state_init:bool=True):
        """
        This constructor initializes the internal variables of the simulator
        according to the simulation_parameters by calling the function reset().

        Parameters
        ----------
        game_parameters : GameParameters
            The game parameters for the simulator.
        steady_state_init : bool
            Optional, Default: True. This will define whether population
            initialization for the simulation is as given by the initial
            parameters for the optimization (lst_init_types, 
            lst_init_urgencies, lst_init_karmas), or the steady state computed
            by the optimizer will be used.

        Returns
        -------
        None
        """
        self._parameters = game_parameters
        self.reset(state, steady_state_init)
        
    ### Public Methods
    def reset(self, state : StateDistribution, steady_state_init:bool=True):
        """
        This function resets all internal variables of the simulator so that 
        a restart of the simulation from the beginning is possible.

        Parameters
        ----------
        state : StateDistribution
            The state distribution, that is used to reset the population in
            case steady_state_init is set to True.
        steady_state_init : bool
            Optional, Default: True. This will define whether population
            initialization for the simulation is as given by the initial
            parameters for the optimization (lst_init_types, 
            lst_init_urgencies, lst_init_karmas), or the steady state computed
            by the optimizer will be used.
        
        Returns
        -------
        None
        """
        self.epoch_counter = 0
        self._population = np.zeros((self._parameters._num_agents, 5))
        if steady_state_init:
            for n in range(0, self._parameters._num_agents):
                probs = [np.sum(state.dist_matrix[x]) for x in self._parameters._set_types]
                probs = probs / np.sum(probs)
                random_type = np.random.choice(self._parameters._set_types, size=1, p=probs)[0]
                probs = [np.sum(state.dist_matrix[random_type][x]) for x in self._parameters._set_urgencies]
                probs = probs / np.sum(probs)
                random_urgency = np.random.choice(self._parameters._set_urgencies, size=1, p=probs)[0]
                probs = [np.sum(state.dist_matrix[random_type][random_urgency][x]) for x in self._parameters._set_state_karmas]
                probs = probs / np.sum(probs)
                random_karma = np.random.choice(self._parameters._set_state_karmas, size=1, p=probs)[0]
                self._population[n,Simulator._TYPE_COL] = random_type
                self._population[n,Simulator._URGENCY_COL] = random_urgency
                self._population[n,Simulator._KARMA_COL] = random_karma
        else:
            self._population[:,Simulator._TYPE_COL] = self._parameters._lst_init_types
            self._population[:,Simulator._URGENCY_COL] = self._parameters._lst_init_urgencies
            self._population[:,Simulator._KARMA_COL] = self._parameters._lst_init_karmas
        self._state_transition_counter = np.zeros((self._parameters._num_urgencies, 
                                                   self._parameters._num_state_karmas, 
                                                   self._parameters._num_urgencies, 
                                                   self._parameters._num_state_karmas))
        self._state_transition_counter_participants = np.zeros((self._parameters._num_urgencies, 
                                                                self._parameters._num_state_karmas, 
                                                                self._parameters._num_urgencies, 
                                                                self._parameters._num_state_karmas))
        
        
    def begin_epoch(self):
        """
        This function begins the next epoch of the game.

        Parameters
        ----------
        None

        Raises
        ------
        Exception
            In case the previous epoch was not closed.
            In case the karma overflow is not empty due to invalid functions func_payment, func_overflow_distribution or func_karma_redistribution.

        Returns
        -------
        None
        """
        # check if previous epoch was closed
        if(self._open_epoch):
            raise Exception("Previous epoch was not closed! Cannot start next epoch!")
        self._open_epoch = True
        # check if _karma_overflow is empty
        if(not self._karma_overflow==0):
            raise Exception("Karma overflow is not empty, func_payment, func_overflow_distribution or func_karma_redistribution need to be corrected so that the total amount of Karma stays constant after each epoch!")
        # reset epoch outcomes and particiapnts
        self.all_epoch_participants = []
        self.all_epoch_outcomes = []
        # record karma before interactions
        self._total_karma_epoch_begin = self._karma_overflow + np.sum(self._population[:,Simulator._KARMA_COL])
        # record states before interactions
        self._population_states_before = self._return_participant_states(self.peer_selection_whole_population())
        
    def play_interaction(self, policy: Policy, participants: List[int]):
        """
        This function performs one interaction, consisting out of decision
        making, outcome determinantion and transaction (of costs, urgencies, 
        and karma balances exclusive overflow distribution and karma 
        redistribution).
        
        Parameters
        ----------
        policy : Policy
            The Karma policy that should be used to play in this interaction.
        
        participants : List[int]
            The list of participants of this interaction.
            
        Raises
        ------
        Exception
            In case the policy is invalid, an exception will be raised.

        Returns
        -------
        None
        """
        # check validity of policy
        if(not policy.validate()):
            raise Exception("Invalid policy, therefore epoch was not played!")
        # interaction
        actions, outcomes = self._decision_making_outcome_determination(policy, participants)
        # transaction
        self._transaction(participants, actions, outcomes)       
        # recording of participants and outcomes
        for agent_idx in participants:
            if agent_idx not in self.all_epoch_participants:
                self.all_epoch_participants.append(agent_idx)
                self.all_epoch_outcomes.append(outcomes[participants.index(agent_idx)])
            else:
                idx_position = self.all_epoch_participants.index(agent_idx)
                self.all_epoch_outcomes[idx_position] = max(self.all_epoch_outcomes[idx_position], outcomes[participants.index(agent_idx)])
    
    def close_epoch(self):
        """
        This function closes / ends the current epoch of the game. This 
        includes the distribution of karma overflow, and the redistribution of 
        karma balances.

        Parameters
        ----------
        None
        
        Raises
        ------
        Exception
            In case there is no open epoch started yet.
            In case the karma overflow is not empty due to invalid functions 
            func_payment, func_overflow_distribution or func_karma_redistribution.
            In case the total amount of karma did not stay constant due to 
            invalid functions func_payment, func_overflow_distribution or 
            func_karma_redistribution.

        Returns
        -------
        epoch_counter : int
            The increment epoch_counter after playing the epoch.
        """
        # urgency transition
        self._urgency_transition(self.all_epoch_participants, self.all_epoch_outcomes)
        # distribute overflow karma and redistribute karma
        self._karma_re_distribution()
        # record state transition
        population_states_after = self._return_participant_states(self.peer_selection_whole_population())        
        self._record_state_transitions(self._state_transition_counter,
                                       self.peer_selection_whole_population(), 
                                       self._population_states_before, 
                                       population_states_after)
        self._record_state_transitions(self._state_transition_counter_participants,
                                       self.all_epoch_participants, 
                                       self._population_states_before, 
                                       population_states_after)
        
        # check if previous epoch was closed
        if(not self._open_epoch):
            raise Exception("No epoch started yet! Cannot close epoch!")
        self._open_epoch = False
        # record karma after interactions
        total_karma_end = self._karma_overflow + np.sum(self._population[:,Simulator._KARMA_COL])
        # check if total karma stayed constant
        if(not self._karma_overflow==0):
            raise Exception("Karma overflow is not empty, func_payment, func_overflow_distribution or func_karma_redistribution need to be corrected so that the total amount of Karma stays constant after each epoch!")
        if(not self._total_karma_epoch_begin==total_karma_end):
            raise Exception("Total Karma amount before and after playing the epoch is not equal, func_payment, func_overflow_distribution or func_karma_redistribution need to be corrected so that the total amount of Karma stays constant after each epoch!")
        # updates
        self.epoch_counter += 1
        return self.epoch_counter
        
    def peer_selection_random(self, excluded_indexes: List[int]=None):
        """
        This function simulates the random peer selection, which is the random
        selection of num_participants from a population of num_agents.

        Parameters
        ----------
        excluded_indexes : List[int]
            Optional. Default is None. If defined, certain agents can be 
            excluded from the peer_selection process.

        Returns
        -------
        participants : List[int]
            The agent indexes of the participants of the interaction.
        """        
        possible_indexes = [x for x in range(0,self._parameters._num_agents)]
        if excluded_indexes is not None:
            for index in excluded_indexes:
                possible_indexes.remove(index)
        #participants = random.sample(possible_indexes, self._parameters._num_participants)
        participants = np.random.choice(possible_indexes, size=self._parameters._num_participants, replace=False)
        return participants.tolist()
    
    def peer_selection_whole_population(self):
        """
        This function simulates the peer selection of all agents interacting
        with each other at the same time.

        Parameters
        ----------
        None

        Returns
        -------
        participants : List[int]
            The agent indexes of the participants of the interaction.
        """        
        participants = [x for x in range(0,self._parameters._num_agents)]
        return participants
    
    def peer_selection_random_non_exclusive_peer_groups(self, n_groups: int):
        """
        This function simulates the random peer selection, which is the random
        selection of num_participants from a population of num_agents. This is
        repeated for n_pairs times, and returns a list of participants lists.
        This function produces non-exclusive peer groups, meaning that it is
        possible that one agent participates in more than one interaction.

        Parameters
        ----------
        n_groups : int
            Number of participant groups that shall be returned.

        Returns
        -------
        lst_participants : List[List[int]]
            The list of participant lists (agent indexes of the participants) 
            of the interactions.
        """        
        lst_participants = []
        for n in range(0,n_groups):
            lst_participants.append(self.peer_selection_random())
        return lst_participants
    
    def peer_selection_random_exclusive_peer_groups(self, n_groups: int):
        """
        This function simulates the random peer selection, which is the random
        selection of num_participants from a population of num_agents. This is
        repeated for n_pairs times, and returns a list of participants lists.
        This function produces exclusive peer groups, meaning that it is
        not possible that one agent participates in more than one interaction.

        Parameters
        ----------
        n_groups : int
            Number of participant groups that shall be returned.

        Returns
        -------
        lst_participants : List[List[int]]
            The list of participant lists (agent indexes of the participants) of the interactions.
        """  
        lst_indexes_used = []
        lst_participants = []
        for n in range(0, n_groups):
            participants = self.peer_selection_random(excluded_indexes=lst_indexes_used)
            lst_participants.append(participants)
            for index in participants:
                if index not in lst_indexes_used:
                    lst_indexes_used.append(index)
        return lst_participants
    
    def get_agent_information(self, agent_idx: int, arr:bool=True):
        """
        This function returns the information of an agent.
        
        Parameters
        ----------
        agent_idx : int
            The agent index in the population.
        arr : bool
            Whether an array shall be returned, or a tuple of values. 
            Default is True.
            
        Returns
        -------
        arr : np.array
            The agent information as a 1x5 array, where each column is 
            represented by following enumeration: TYPE_COL = 0, 
            URGENCY_COL = 1, KARMA_COL = 2, CUM_COST_COL = 3,
            ENCOUNTERS_COL = 4.
        type : int
            (if parameter 'arr' set to False)
        urgency : int
            (if parameter 'arr' set to False)
        karma : int
            (if parameter 'arr' set to False)
        cum_cost : float
            (if parameter 'arr' set to False)
        encounters : int
            (if parameter 'arr' set to False)
        """
        agent_information = self._population[agent_idx]
        if arr:
            return agent_information
        else:
            return int(agent_information[Simulator._TYPE_COL]), int(agent_information[Simulator._URGENCY_COL]), int(agent_information[Simulator._KARMA_COL]), agent_information[Simulator._CUM_COST_COL], int(agent_information[Simulator._ENCOUNTERS_COL]) 
    
    def get_agent_population(self):
        """
        This function returns the complete population, meaning the information
        of all agents. Each column is represented by following enumeration: 
        TYPE_COL = 0, URGENCY_COL = 1, KARMA_COL = 2, CUM_COST_COL = 3,
        ENCOUNTERS_COL = 4.
        
        Parameters
        ----------
        None
        
        
        Returns
        -------
        population : np.array
            The agent information as a [num_agent x 5] array, where each column 
            is represented by following enumeration: TYPE_COL = 0, 
            URGENCY_COL = 1, KARMA_COL = 2, CUM_COST_COL = 3,
            ENCOUNTERS_COL = 4.
        """
        return self._population.copy()
    
    def get_karma_distribution(self):
        """
        This function returns the Karma distribution of the population.
        
        Parameters
        ----------
        None
        
        Returns
        -------
        distribution : np.array
            This distribution represents the relative share of the population
            that owns a specific amount of Karma. The distribution is
            calculated for all karmas in parameters._num_state_karmas. The
            last value of the distribution thus represents the share of agents
            with a karma balance of num_state_karmas-1 or higher.
        """
        pop_karma = self._population[:,Simulator._KARMA_COL]
        _, counts = np.unique(pop_karma, return_counts=True)
        distribution_state = np.zeros(self._parameters._num_state_karmas)
        max_distr_karma = self._parameters._set_state_karmas[-1]
        for karma in range(0,len(counts)):
            if(karma>=max_distr_karma):
                distribution_state[max_distr_karma] += counts[karma]
            else:
                distribution_state[karma] = counts[karma]
        distribution_state = distribution_state / self._parameters._num_agents
        return distribution_state
    
    def get_state_transition_counts(self, participants_only: bool=True):
        """
        This function returns the state transition counts.
        This means the times an agent reached state X (urgency, karma) from the
        original state Y (urgency, karma). 
        
        Parameters
        ----------
        participants_only : bool
            Optional. Default=True. This defines, whether the state transition
            distribution of all agents, or just of the agents that participated
            during an epoch shall be considered.
        Returns
        -------
        distribution : np.array
            This distribution is a four dimensional matrix with following
            indexes: [urgency_next][karma_next][urgency_before][karma_before]
        """
        if participants_only:
            return self._state_transition_counter_participants.copy()
        else:
            return self._state_transition_counter.copy()
        
    def get_population_cumulative_costs(self):
        """
        This function returns the cumulative, and discounted costs for each 
        agent of the population. Discounting takes place for each encounter 
        and not for each epoch!
        
        Parameters
        ----------
        None
        
        Returns
        -------
        pop_costs : np.array
            This array represents the cumulative, discounted costs for each
            agent of the population.
        """
        return self._population[:,Simulator._CUM_COST_COL].copy()
    
    def get_total_cumulative_costs(self):
        """
        This function returns the total cumulative, and discounted costs as
        sum of the cumulative, discounted costs over all agents. Discounting 
        takes place for each encounter and not for each epoch!
        
        Parameters
        ----------
        None
        
        Returns
        -------
        costs : float
            This cost is the sum of all cumulative, discounted costs over all
            agents.
        """
        return np.sum(self._population[:,Simulator._CUM_COST_COL])
        
    def get_population_encounters(self):
        """
        This function returns the number of encounters (being a participant in
        an interaction) for each agent of the population.
        
        Parameters
        ----------
        None
        
        Returns
        -------
        pop_encounters : np.array
            This represents the number of encounters for each agent during the game.
        """
        return self._population[:,Simulator._ENCOUNTERS_COL].copy().astype(int)
    
    def get_total_encounters(self):
        """
        This function returns the total number of encounters (participations in
        interactions) as sum over all agents of the population.
        
        Parameters
        ----------
        None
        
        Returns
        -------
        encounters : int
            This represents the number of total encounters during the game.
        """
        return int(np.sum(self._population[:,Simulator._ENCOUNTERS_COL].copy()))
       
    #### Main Simulation Epoch Functions
    def _decision_making_outcome_determination(self, policy: Policy, participants: List[int]):
        """
        This function simulates the decision making and outcome determination.
        The fields last_participants, last_actions and last_outcome are 
        updated for the user to be accessible.

        Parameters
        ----------
        policy : Policy
            The policy used by the agents for the decision making process.
        participants : List[int]
            The agent indexes of the participants of the interaction.

        Returns
        -------
        actions : List[int]
            The list of actions of the participants.
        outcomes : List[int]
            The outcomes of the interaction.
        """
        # peer selection
        self._record_encounters(participants)
        # agent decision making process
        actions = self._get_actions(policy, participants)
        # determine outcome
        outcomes = self._parameters._func_outcome(actions)
        # store results
        self.last_participants = participants
        self.last_actions = actions
        self.last_outcomes = outcomes
        return actions, outcomes
    
    def _transaction(self, participants: List[int], actions: List[int], outcomes: List[int]):
        """
        This function simulates the transation that consists of following
        steps: (i) to update karma balances and urgencies based on interaction,
        (ii) to update karma balances according to redistribution rule.

        Parameters
        ----------
        participants : List[int]
            The agent indexes of the participants of the interaction.
        actions : List[int]
            The list of actions of the participants.
        outcomes : List[int]
            The outcomes of the interaction.
            
        Returns
        -------
        None
        """
        # cost transition
        self._cost_transition(participants, outcomes)
        # karma transition
        self._karma_transition(participants, actions, outcomes)
    
    ### Deeper Epoch Functions  
    def _get_actions(self, policy: Policy, participants: List[int]):
        """
        This function simulates the decision making process of participants
        based on their private state information (type, urgency, karma) and
        the policy.

        Parameters
        ----------
        policy : Policy
            The policy used by the agents for the decision making process.
        participants : List[int]
            The agent indexes of the participants of the interaction.

        Returns
        -------
        actions : List[int]
            The list of actions of the participants.
        """
        actions = []
        for agent_idx in participants:
            agent_type = self._population[agent_idx,Simulator._TYPE_COL]
            agent_urgency = self._population[agent_idx,Simulator._URGENCY_COL]
            agent_karma = min(self._population[agent_idx,Simulator._KARMA_COL], self._parameters._num_actions-1)
            probabilities = policy.prob_matrix[int(agent_type)][int(agent_urgency)][int(agent_karma)]
            # action = random.choices(self._parameters._set_actions, weights=probabilities, k=1)[0]
            action = np.random.choice(a=self._parameters._set_actions, size=1, p=probabilities)[0]
            actions.append(action)
        return actions
    
    def _update_cum_costs(self, participants: List[int], immediate_costs : List[float]):
        """
        This function simulates the updates of the participants cumulated 
        costs.

        Parameters
        ----------
        participants : List[int]
            The agent indexes of the participants of the interaction.
        immediate_costs : List[float]
            The list of participants' immediate costs.

        Returns
        -------
        None
        """
        for agent_idx in participants:
            agent_type = self._population[agent_idx,Simulator._TYPE_COL]
            agent_temporal_preference = self._parameters._map_type_temp_preference[agent_type]
            agent_cum_cost_current = self._population[agent_idx,Simulator._CUM_COST_COL]
            self._population[agent_idx,Simulator._CUM_COST_COL] = immediate_costs[participants.index(agent_idx)] + agent_temporal_preference*agent_cum_cost_current
    
    def _cost_transition(self, participants: List[int], outcomes: List[int]):
        """
        This function updates the cumulated costs of all participants.
        The immediate cost is the product of outcome and urgency.
        
        Parameters
        ----------
        participants : List[int]
            The agent indexes of the participants of the interaction.
        outcomes : List[int]
            The outcomes of the interaction.
            
        Returns
        -------
        None
        """
        immediate_costs = []
        for n in range(0, len(participants)):
            cost = self._parameters._func_cost(self._population[participants[n],Simulator._URGENCY_COL],  outcomes[n])
            immediate_costs.append(cost)
        self._update_cum_costs(participants, immediate_costs)
    
    def _karma_transition(self, participants: List[int], actions: List[int], outcomes: List[int]):
        """
        This function updates the karma balances of all participants.
        
        Parameters
        ----------
        participants : List[int]
            The agent indexes of the participants of the interaction.
        actions : List[int]
            The actions of the participants of the interaction.
        outcomes : List[int]
            The outcomes of the interaction.
            
        Returns
        -------
        None
        """
            # payment
        payments, karma_overflow = self._parameters._func_payment(
            actions=actions, 
            outcomes=outcomes
        )
        self._karma_overflow = karma_overflow
        for i in range(0, len(payments)):
            self._population[participants[i],Simulator._KARMA_COL] += payments[i]
        
    def _karma_re_distribution(self):
        """
        This function distributes the karma overflow and redistirbutes karma
        according to func_overflow_distribution and func_karma_redistribution.
        
        Parameters
        ----------
        None
        
        Returns
        -------
        None
        """
            # overflow distribution
        pop_karmas_after_distribution = self._parameters._func_overflow_distribution(
            pop_karmas=list(self._population[:,Simulator._KARMA_COL]), 
            overflow=self._karma_overflow)
        self._population[:,Simulator._KARMA_COL] = pop_karmas_after_distribution
            # karma redistribution
        pop_karmas_after_redistribution = self._parameters._func_karma_redistribution(
            pop_karmas=list(self._population[:,Simulator._KARMA_COL])
        )
        self._karma_overflow = 0
        self._population[:,Simulator._KARMA_COL] = pop_karmas_after_redistribution
        
    def _urgency_transition(self, all_epoch_participants: List[int], max_epoch_outcomes: List[int]):
        """
        This function updates the urgency levels of all agents using the user
        specified function func_urgency_transition.
        
        Parameters
        ----------
        all_epoch_participants : List[int]
            The agent indexes of the participants of all the interactions 
            during this epoch.
        max_epoch_outcomes : List[int]
            The maximum outcomes of all the interactions during this epoch.
            
        Returns
        -------
        None
        """
        for agent_idx in range(0, self._parameters._num_agents):
            if(agent_idx not in all_epoch_participants):
                outcome_code = -1
            else:
                outcome_code = max_epoch_outcomes[all_epoch_participants.index(agent_idx)]
            self._population[agent_idx,Simulator._URGENCY_COL] = self._parameters._func_urgency_transition(self._population[agent_idx, Simulator._TYPE_COL],
                                                                                                          self._population[agent_idx, Simulator._URGENCY_COL], 
                                                                                                          outcome_code,
                                                                                                          self._parameters._set_urgencies)
    
    #### Recording Functions
    def _record_encounters(self, participants: List[int]):
        """
        This function updates the encounter counters for the participants.

        Parameters
        ----------
        participants : List[int]
            The agent indexes of the participants of the interaction.

        Returns
        -------
        None
        """
        for participant in participants:
            self._population[participant][Simulator._ENCOUNTERS_COL] += 1
            
    def _return_participant_states(self, participants: List[int]):
        """
        This function returns the states of participants.

        Parameters
        ----------
        participants : List[int]
            The agent indexes of the participants of the interaction.

        Returns
        -------
        participant_states : np.array
            The subset of participant states from the population. First column
            represents the urgencies, second column represents the karma
            balances.
        """
        participant_states = []
        for participant in participants:
            participant_states.append([
                    int(self._population[participant,Simulator._URGENCY_COL]),
                    min(int(self._population[participant,Simulator._KARMA_COL]), 
                    self._parameters._num_state_karmas-1),
                ])
        return np.asarray(participant_states)
    
    def _record_state_transitions(self, counter, participants: List[int], population_states_before: npt.NDArray, population_states_after: npt.NDArray):
        """
        This function records the state transitions of an interaction.

        Parameters
        ----------
        counter : npt.NDArray
            The counter array that should count the state transitions.
        participants : List[int]
            The agent indexes of the participants of the interaction.
        population_states_before : npt.NDArray
            The subset of participant states from the population before 
            the interaction. First column represents the urgencies, second 
            column represents the karma balances.
        population_states_after : npt.NDArray
            The subset of participant states from the population after 
            the interaction. First column represents the urgencies, second 
            column represents the karma balances.
            
        Returns
        -------
        None
        """
        for p in participants:
            counter[population_states_after[p][0]][population_states_after[p][1]][population_states_before[p][0]][population_states_before[p][1]] += 1
