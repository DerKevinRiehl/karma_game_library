# Karma Game Library, Copyrights Kevin Riehl 2023, <kriehl@ethz.ch>

# Imports
import numpy as np
import karma_game_library
from typing import Callable

# Classes

class Optimizer:
    """
    A class to provide functionality to iteratively optimize the agent policy.

    Attributes
    ----------
    hyper_lambda : float
        Exponential scaling factor for policy perturbation.
    hyper_dt : float
        Stepwidth for state updates. Only values between 0.0 and 1.0 allowed.
    hyper_nu : float
        Stepwidth for policy updates. Only values between 0.0 and 1.0 allowed.
    func_prob_outcome : Callable, (outcome, actions) -> (probability)
        This function returns the probability for the outcome given the actions.
    func_prob_karma_transition : Callable, (karma_next, karma, actions, outcome) -> (probability)
        This function returns the probability of karma_next given a prior 
        karma, actions and outcome.
    func_prob_urgency_transition : Callable, (urgency_next, urgency, outcome, atype, set_urgencies) -> (probability)
        This function returns the probability of urgency_next given a prior
        urgency and outcome and type.
    v : np.array [action]
        "Distribution of agents' actions"
        Intermediate result, returns probability of an action.
    gamma : np.array [outcome][action]
        "Probability of interaction outcome given bid"
        Intermediate result, returns the probability of an outcome given the action.
    kappa : np.array [karma_next][karma][action][outcome]
        "Probabilistic karma transition function"
        Intermediate result, returns the probability of karma_next given the
        prior karma, action and outcome.
    xi : np.array [urgency][action]
        "Immediate reward function"
        Intermediate result, returns the expected immediate reward (negative 
        cost) for an participant with specific urgency, and action.
    rho : np.array [type][urgency_next][karma_next][urgency][karma][action]
        "Probabilistic state transition function"
        Intermediate result, returns the probability that participant of type
        ends up in next state urgency_next and karma_next given a prior
        urgency and karma and action.
    R : np.array [type][urgency][karma]
        "Expected immediate reward"
        Imme result, returns the expected immediate reward 
        (negative cost) for an participant of a specific type, urgency, karma.
    P : np.array [type][urgency_next][karma_next][urgency][karma]
        "State transition probabilities"
        Intermediate result, returns the probability of an agent of specific
        type to end up in next state urgency_next, karma_next given his prior
        state urgency, karma
    V : np.array [type][urgency][karma]
        "Expected infinite horizon reward"
        Intermediate result, returns the expected inifinite horizon reward 
        (negative cost) for an agent of specific type, urgency, karma.
    Q : np.array [type][urgency][karma][action]
        "Single-stage deviation reward (from current policy)"
        Intermediate result, returns the expected reward (negative cost) for an
        participant with specific type, urgency, karma and action.
    pi_pbp : np.array [type][urgency][karma][bid]
        "Perturbed best response policy"
        Intermediate result, represents a modified, potentially better policy.
    delta_state : float
        The amount the state changed when compared to previous iteration. Can
        be used as an indicator for convergence.
    delta_policy : float
        The amount the policy changed when compared to previous iteration. Can
        be used as an indicator for convergence.
    """
    
    #### Attributes
        # Parameters for Optimization
    _parameters = None
    hyper_lambda = 150 # 1000
    hyper_dt = 0.01
    hyper_nu = 0.2
    _hyper_infinite_horizon_reward_max_iterations = 1000
    _hyper_infinite_horizon_reward_convergence_threshold = 0.00001
    func_prob_outcome = None
    func_prob_karma_transition = None
    func_prob_urgency_transition = None
        # Intermediate Optimization Results
    v = None
    gamma = None
    kappa = None
    xi = None
    rho = None
    R = None
    P = None
    V = None
    Q = None
    pi_pbp = None
    delta_state = None
    delta_policy = None
        # Recorder
    recorder = []
    #
    _enlarge_action_space = False
    _enlarge_state_space = False
    
    def __init__(self, 
                 game_parameters: karma_game_library.GameParameters,
                 hyper_lambda: float, 
                 hyper_dt: float, 
                 hyper_nu: float, 
                 func_prob_outcome : Callable, 
                 func_prob_karma_transition : Callable,
                 func_prob_urgency_transition : Callable):
        """
        This constructor initializes the PolicyOptimizer.

        Parameters
        ----------
        game_parameters : GameParameters
            The simulation parameters.
        hyper_lambda : float
            Exponential scaling factor for policy perturbation.
        hyper_dt : float
            Stepwidth for state updates. Only values between 0.0 and 1.0 
            allowed.
        hyper_nu : float
            Stepwidth for policy updates. Only values between 0.0 and 1.0 
            allowed.
        func_prob_outcome : Callable, (outcome, actions) -> (probability)
            This function returns the probability for the outcome given the 
            actions.
        func_prob_karma_transition: Callable, (outcome, actions) -> (probability)
            This function returns the probability of karma_next given a prior 
            karma, actions and outcome.
        func_prob_urgency_transition: Callable, (urgency_next, urgency, outcome, atype, set_urgencies) -> (probability)
            This function returns the probability of urgency_next given a prior
            urgency and outcome.
            
        Raises
        ------
        Exception
            In case the input parameters are not plausible a specific exception 
            will be raised.
            
        Returns
        -------
        None.

        """
        # Plausibility checks
        karma_game_library.check.non_negative(hyper_lambda, "hyper_lambda")
        karma_game_library.check.float_bounded(hyper_dt, 0.0, 1.0, "hyper_dt")
        karma_game_library.check.float_bounded(hyper_nu, 0.0, 1.0, "hyper_nu")
        karma_game_library.check.func_not_none(func_prob_outcome, "func_prob_outcome")
        karma_game_library.check.func_not_none(func_prob_karma_transition, "func_prob_karma_transition")
        karma_game_library.check.func_not_none(func_prob_urgency_transition, "func_prob_urgency_transition")
        # Initialization of matrix
        self._parameters = game_parameters
        self.hyper_lambda = hyper_lambda
        self.hyper_dt = hyper_dt
        self.hyper_nu = hyper_nu
        self.func_prob_outcome = func_prob_outcome
        self.func_prob_karma_transition = func_prob_karma_transition
        self.func_prob_urgency_transition = func_prob_urgency_transition
        self._enlarge_action_space = False
        self._enlarge_state_space = False

    def compute_iteration(self, state: karma_game_library.StateDistribution, policy: karma_game_library.Policy):
        """
        This function updates the social state distribution and policy 
        probability matrix according to an optimization algorithm.
        This algorithm is iterative, and shall be repeated iteratively until
        state and policy converge to a stationary Nash equilibrium.

        Parameters
        ----------
        state : StateDistribution
            The social state distribution that should be optimized.
        policy : Policy
            The policy that should be optimized.

        Raises
        ------
        Exception
            In case the policy or state are invalid, an exception will be 
            raised.

        Returns
        -------
        delta_state : float
            The absolute change to the state (can be used as an indicator for 
            convergence).
        delta_policy : float
            The absolute change to the policy (can be used as an indicator 
            for convergence).
        """
        # enlarge state and action spcae if required
        if self._enlarge_action_space:
            self._enlarge_action_space = False
            # print("enlarging action space as limitations reached")
            policy._enlarge_action_space()
            self._parameters._enlarge_action_space()
        if self._enlarge_state_space:
            self._enlarge_state_space = False
            # print("enlarging karma space as limitations reached")
            state._enlarge_karma_space()
            policy._enlarge_karma_space()
            self._parameters._enlarge_karma_space()
        # check validity of state and policy
        if(not state.validate()):
            state.autocorrect()
            # print("Needed to autocorrect state")
            if(not state.validate()):
                raise Exception("Invalid state, could not be autocorrected!")
        if(not policy.validate()):
            policy.autocorrect()
            # print("Needed to autocorrect policy")
            if(not policy.validate()):
                raise Exception("Invalid policy, could not be autocorrected!")
        # plausibility checks of hyper parameters
        karma_game_library.check.non_negative(self.hyper_lambda, "hyper_lambda")
        karma_game_library.check.float_bounded(self.hyper_dt, 0.0, 1.0, "hyper_dt")
        karma_game_library.check.float_bounded(self.hyper_nu, 0.0, 1.0, "hyper_nu")
        # compute iteration
        self.v = self._prob_distribution_of_action(state, policy)
        self.gamma = self._prob_distribution_outcome_bid(self.v)
        self.kappa = self._prob_karma_transition(policy, state, self.v, self.gamma)
        self.xi = self._immediate_reward(policy, self.gamma)
        self.rho = self._prob_state_transition_function(policy, self.gamma, self.kappa)
        self.R = self._expected_immediate_reward_policy(policy, self.xi)
        self.P = self._prob_state_transition_policy(policy, self.rho)
        self.V, _ = self._inifite_horizon_reward(self.R, self.P, self._hyper_infinite_horizon_reward_max_iterations, self._hyper_infinite_horizon_reward_convergence_threshold)
        self.Q = self._agent_single_state_deviation_reward(policy, self.xi, self.rho, self.V)
        self.pi_pbp = self._peturbed_best_response(policy, self.Q)
        delta_policy, direct_policy = self._calculate_new_policy(policy, self.pi_pbp)
        delta_state, direct_state = self._calculate_new_state(state, policy, self.P)
        self.recorder.append([delta_state, delta_policy, np.sum(self.V)])
        return delta_state, delta_policy, direct_state, direct_policy
        
    def _prob_distribution_of_action(self, state: karma_game_library.StateDistribution, policy: karma_game_library.Policy):
        """
        Intermediate result, returns probability of an action.

        Parameters
        ----------
        state : StateDistribution
            The current state.
        policy : Policy
            The current policy.

        Returns
        -------
        v : np.array [action]
            "Distribution of agents' actions"
        """
        v = np.zeros(len(self._parameters._set_actions))
        for action in self._parameters._set_actions:
            prob = 0
            for atype in self._parameters._set_types:
                for urgency in self._parameters._set_urgencies:
                    for karma in self._parameters._set_state_karmas:
                        if action in policy._map_possible_actions[karma]:
                            prob += state.dist_matrix[atype][urgency][karma] * policy.prob_matrix[atype][urgency][karma][action]
            v[action] = prob
        return v

    def _prob_distribution_outcome_bid(self, v):
        """
        Intermediate result, returns the probability of an outcome given the action.

        Parameters
        ----------
        v : np.array [action]
            "Distribution of agents' actions"
        Returns
        -------
        gamma : np.array [outcome][action]
            "Probability of resource competition outcome given bid"
        """
        gamma = np.zeros((len(self._parameters._set_outcomes), 
                          len(self._parameters._set_actions)
                          ))
        for outcome in self._parameters._set_outcomes:
            for action in self._parameters._set_actions:
                prob = 0
                for other_action in self._parameters._set_actions:
                    prob += v[other_action] * self.func_prob_outcome(outcome, [action, other_action])
                gamma[outcome][action] = prob

        return gamma
    
    def _prob_karma_transition(self, policy: karma_game_library.Policy, state: karma_game_library.StateDistribution, v, gamma):
        """
        Intermediate result, returns the probability of karma_next given the
        prior karma, action and outcome.

        Parameters
        ----------
        policy : Policy
            The current policy.
        v : np.array [action]
            "Distribution of agents' actions"
        gamma : np.array [outcome][action]
            "Probability of resource competition outcome given bid"
            
        Returns
        -------
        kappa : np.array [karma_next][karma][action][outcome]
            "Probabilistic karma transition function"
        """
        kappa = np.zeros((len(self._parameters._set_state_karmas), 
                          len(self._parameters._set_state_karmas), 
                          len(self._parameters._set_actions), 
                          len(self._parameters._set_outcomes)
                          ))
        for karma_next in self._parameters._set_state_karmas:
            for karma in self._parameters._set_state_karmas:
                for action in policy._map_possible_actions[karma]:
                    for outcome in self._parameters._set_outcomes:
                        kappa[karma_next][karma][action][outcome] = self.func_prob_karma_transition(karma_next, karma, action, outcome, state.dist_matrix, policy.prob_matrix, self._parameters, gamma, v) 
        return kappa

    def _immediate_reward(self, policy: karma_game_library.Policy, gamma):
        """
        Intermediate result, returns the expected immediate reward (negative 
        cost) for an participant with specific type, urgency, karma balance 
        and action.

        Parameters
        ----------
        policy : Policy
            The current policy.
        gamma : np.array [outcome][action]
            "Probability of resource competition outcome given bid"
            
        Returns
        -------
        xi : np.array [urgency][action]
            "Immediate reward function"
        """
        xi = np.zeros((len(self._parameters._set_urgencies), 
                       len(self._parameters._set_actions)
                       ))
        for urgency in self._parameters._set_urgencies:
            for action in self._parameters._set_actions:
                exp_cost = 0
                for outcome in self._parameters._set_outcomes:
                    exp_cost += self._parameters._func_cost(urgency, outcome) * gamma[outcome][action]
                xi[urgency][action] = - exp_cost
        return xi

    def _prob_state_transition_function(self, policy: karma_game_library.Policy, gamma, kappa):
        """
        Intermediate result, returns the probability that participant of type
        ends up in next state urgency_next and karma_next given a prior
        urgency and karma and action.

        Parameters
        ----------
        policy : Policy
            The current policy.
        gamma : np.array [outcome][action]
            "Probability of resource competition outcome given bid"
        kappa : np.array [karma_next][karma][action][outcome]
            "Probabilistic karma transition function"
            
        Returns
        -------
        rho : np.array [type][urgency_next][karma_next][urgency][karma][action]
            "Probabilistic state transition function"
        """
        rho = np.zeros((len(self._parameters._set_types),
                        len(self._parameters._set_urgencies),
                        len(self._parameters._set_state_karmas),
                        len(self._parameters._set_urgencies),
                        len(self._parameters._set_state_karmas),
                        len(self._parameters._set_actions),
                        ))
        for atype in self._parameters._set_types:
            for urgency_next in self._parameters._set_urgencies:
                for karma_next in self._parameters._set_state_karmas:
                    for urgency in self._parameters._set_urgencies:
                        for karma in self._parameters._set_state_karmas:
                            for action in policy._map_possible_actions[karma]:
                                prob = 0
                                for outcome in self._parameters._set_outcomes:
                                    prob += self.func_prob_urgency_transition(urgency_next, urgency, outcome, atype, self._parameters._set_urgencies) * gamma[outcome][action] * kappa[karma_next][karma][action][outcome]                           
                                rho[atype][urgency_next][karma_next][urgency][karma][action] = prob
        return rho
    
    def _expected_immediate_reward_policy(self, policy: karma_game_library.Policy, xi):
        """
        Intermediate result, returns the expected immediate reward 
        (negative cost) for an participant of a specific type, urgency, karma.

        Parameters
        ----------
        policy : Policy
            The current policy.
        xi : np.array [urgency][action]
            "Immediate reward function"
            
        Returns
        -------
        R : np.array [type][urgency][karma]
            "Expected immediate reward"
        """
        R = np.zeros((len(self._parameters._set_types),
                       len(self._parameters._set_urgencies),
                       len(self._parameters._set_state_karmas)
                       ))
        for atype in self._parameters._set_types:
            for urgency in self._parameters._set_urgencies:
                for karma in self._parameters._set_state_karmas:
                    reward = 0
                    for action in policy._map_possible_actions[karma]:
                        reward += policy.prob_matrix[atype][urgency][karma][action] * xi[urgency][action]
                    R[atype][urgency][karma] = reward
        return R
    
    def _prob_state_transition_policy(self, policy: karma_game_library.Policy, rho): 
        """
        Intermediate result, returns the probability of an agent of specific
        type to end up in next state urgency_next, karma_next given his prior
        state urgency, karma

        Parameters
        ----------
        policy : Policy
            The current policy.
        rho : np.array [type][urgency_next][karma_next][urgency][karma][action]
            "Probabilistic state transition function"
            
        Returns
        -------
        P : np.array [type][urgency_next][karma_next][urgency][karma]
            "State transition probabilities"
        """
        P = np.zeros((len(self._parameters._set_types),
                       len(self._parameters._set_urgencies),
                       len(self._parameters._set_state_karmas),
                       len(self._parameters._set_urgencies),
                       len(self._parameters._set_state_karmas),
                       ))
        for atype in self._parameters._set_types:
            for urgency_next in self._parameters._set_urgencies:
                for karma_next in self._parameters._set_state_karmas:
                    for urgency in self._parameters._set_urgencies:
                        for karma in self._parameters._set_state_karmas:
                            prob = 0
                            for action in policy._map_possible_actions[karma]:
                                prob += policy.prob_matrix[atype][urgency][karma][action] * rho[atype][urgency_next][karma_next][urgency][karma][action]
                            P[atype][urgency_next][karma_next][urgency][karma] = prob
        return P
    
    def _inifite_horizon_reward(self, R, P, max_iterations, convergence_threshold):
        """
        Intermediate result, returns the expected inifinite horizon reward 
        (negative cost) for an agent of specific type, urgency, karma.

        Parameters
        ----------
        R : np.array [type][urgency][karma]
            "Expected immediate reward"
        P : np.array [type][urgency_next][karma_next][urgency][karma]
            "State transition probabilities"
        max_iterations: int
            How many iterations at most shall be used to calculate the 
            infinite horizon reward.
        convergence_threshold : float
            After which difference to previous iteration the iterations can
            be aborted, as inifinite horizon reward converged sufficiently.
        Returns
        -------
        V : np.array [type][urgency][karma]
            "Expected infinite horizon reward"
        difference_V : float
            The convergence difference at which the calculation stopped.
        """
        V = np.zeros((len(self._parameters._set_types),
                       len(self._parameters._set_urgencies),
                       len(self._parameters._set_state_karmas),
                       ))
        # init with alpha = 0
        for atype in self._parameters._set_types:
            for urgency in self._parameters._set_urgencies:
                for karma in self._parameters._set_state_karmas:
                    V[atype][urgency][karma] = R[atype][urgency][karma]
        # calculate iteratively
        for it in range(0,max_iterations):
            last_V = np.copy(V)
            for atype in self._parameters._set_types:
                alpha = self._parameters._map_type_temp_preference[atype]
                for urgency in self._parameters._set_urgencies:
                    for karma in self._parameters._set_state_karmas:
                        reward = R[atype][urgency][karma]
                        for urgency_next in self._parameters._set_urgencies:
                            for karma_next in self._parameters._set_state_karmas:
                                reward += alpha * P[atype][urgency_next][karma_next][urgency][karma] * V[atype][urgency_next][karma_next]
                        V[atype][urgency][karma] = reward
            difference_V = np.sum(last_V-V)
            if(difference_V<convergence_threshold):
                break
        return V, difference_V
    
    def _agent_single_state_deviation_reward(self, policy: karma_game_library.Policy, xi, rho, V):
        """
        Intermediate result, returns the expected reward (negative cost) for an
        participant with specific type, urgency, karma and action.

        Parameters
        ----------
        policy : Policy
            The current policy.
        xi : np.array [urgency][action]
            "Immediate reward function"
        rho : np.array [type][urgency_next][karma_next][urgency][karma][action]
            "Probabilistic state transition function"
        V : np.array [type][urgency][karma]
            "Expected infinite horizon reward"

        Returns
        -------
        Q : np.array [type][urgency][karma][action]
            "Single-stage deviation reward (from current policy)"
        """
        Q = np.zeros((len(self._parameters._set_types),
                       len(self._parameters._set_urgencies),
                       len(self._parameters._set_state_karmas),
                       len(self._parameters._set_actions),
                       ))
        for atype in self._parameters._set_types:
            for urgency in self._parameters._set_urgencies:
                for karma in self._parameters._set_state_karmas:
                    for action in policy._map_possible_actions[karma]:
                        alpha = self._parameters._map_type_temp_preference[atype]
                        reward = xi[urgency][action]
                        for urgency_next in self._parameters._set_urgencies:
                            for karma_next in self._parameters._set_state_karmas:
                                reward += alpha * rho[atype][urgency_next][karma_next][urgency][karma][action] * V[atype][urgency_next][karma_next]
                        Q[atype][urgency][karma][action] = reward
        return Q
    
    def _peturbed_best_response(self, policy: karma_game_library.Policy, Q): 
        """
        Intermediate result, represents a modified, potentially better policy.

        Parameters
        ----------
        policy : Policy
            The current policy.
        Q : np.array [type][urgency][karma][action]
            "Single-stage deviation reward (from current policy)"

        Returns
        -------
        pi_pbp : np.array [type][urgency][karma][bid]
            "Perturbed best response policy"
        """
        pi_pbp = np.zeros((len(self._parameters._set_types),
                       len(self._parameters._set_urgencies),
                       len(self._parameters._set_state_karmas),
                       len(self._parameters._set_actions),
                       ))
        for atype in self._parameters._set_types:
            for urgency in self._parameters._set_urgencies:
                for karma in self._parameters._set_state_karmas:
                    for action in policy._map_possible_actions[karma]:
                        divisor = 0
                        Q_max = np.min(Q[atype][urgency][karma])
                        for other_action in policy._map_possible_actions[karma]:
                            if(Q[atype][urgency][karma][other_action] > Q_max):
                                Q_max = Q[atype][urgency][karma][other_action]
                        for other_action in policy._map_possible_actions[karma]:
                            divisor += np.exp(self.hyper_lambda * Q[atype][urgency][karma][other_action] - self.hyper_lambda * Q_max)
                        dividend = np.exp(self.hyper_lambda * Q[atype][urgency][karma][action] - self.hyper_lambda * Q_max)
                        if(divisor==0):
                            pi_pbp[atype][urgency][karma][action] = dividend
                        else:
                            pi_pbp[atype][urgency][karma][action] = dividend / divisor
        return pi_pbp
        
    def _determine_consecutive_distribution(self, d_in, P):
        """
        This function multiplies a given distribution with a state transition 
        probabilities matrix.

        Parameters
        ----------
        d_in : np.array
            The current distribution.
        P : np.array [type][urgency_next][karma_next][urgency][karma]
            "State transition probabilities"

        Returns
        -------
        d_out : np.array 
            New distribution.
        """
        d_out = np.zeros((len(self._parameters._set_urgencies), len(self._parameters._set_state_karmas)))
        for urgency_next in self._parameters._set_urgencies:
            for karma_next in self._parameters._set_state_karmas:
                prob = 0
                for urgency2 in self._parameters._set_urgencies:
                    for karma2 in self._parameters._set_state_karmas: 
                        prob += d_in[urgency2][karma2] * P[urgency_next][karma_next][urgency2][karma2]
                d_out[urgency_next][karma_next] = prob
        d_out = d_out / np.sum(d_out)
        return d_out
    
    def _calculate_new_state(self, state, policy: karma_game_library.Policy, P):
        """
        This function calculates the next state.
        
        Parameters
        ----------
        state : StateDistribution
           The current state.
        policy : Policy
            The current policy.
        P : np.array [type][urgency_next][karma_next][urgency][karma]
            "State transition probabilities"

        Returns
        -------
        update_difference : float
            How much the policy changed due to the update.
        """
        # calculate new state
        dist_matrix_next = state.dist_matrix.copy()
        for atype in self._parameters._set_types:
            product = self._determine_consecutive_distribution(state.dist_matrix[atype], P[atype])
            for urgency in self._parameters._set_urgencies:
                for karma in self._parameters._set_state_karmas:
                    dist_matrix_next[atype][urgency][karma] = (1-self.hyper_dt)*state.dist_matrix[atype][urgency][karma] + (self.hyper_dt)*product[urgency][karma]
        update_difference = np.sum(np.abs(dist_matrix_next - state.dist_matrix))
        update_direction = np.dot(dist_matrix_next.flatten(), state.dist_matrix.flatten()) / (np.linalg.norm(dist_matrix_next) * np.linalg.norm(state.dist_matrix))
        state.dist_matrix = dist_matrix_next
        
        # check if distribution gets too wide, then need to extend karma space
        agg_dist = np.sum(np.sum(dist_matrix_next, axis=0), axis=0)
        
        # especially important when optimizing for pay bid to peer
        # if(np.sum(av_dist[-int(len(av_dist)/2):])>self._parameters._creation_thr_state):
        if(np.sum(agg_dist[-4:])>self._parameters._creation_thr_state):
            self._enlarge_state_space = True
                    
        return update_difference, update_direction
    
    def _calculate_new_policy(self, policy: karma_game_library.Policy, pi_pbp):
        """
        This function calculates the next policy.

        Parameters
        ----------
        policy : Policy
            The current policy.
        pi_pbp : np.array [type][urgency][karma][bid]
            "Perturbated best response policy"

        Returns
        -------
        update_difference : float
            How much the policy changed due to the update.
        """
        # calculate new policy
        prob_mat_next = policy.prob_matrix.copy()
        for atype in self._parameters._set_types:
            for urgency in self._parameters._set_urgencies:
                for karma in self._parameters._set_state_karmas:
                    for action in policy._map_possible_actions[karma]:
                        prob_mat_next[atype][urgency][karma][action] = (1-self.hyper_nu*self.hyper_dt)*policy.prob_matrix[atype][urgency][karma][action] + (self.hyper_nu*self.hyper_dt)*pi_pbp[atype][urgency][karma][action]
        update_difference = np.sum(np.abs(prob_mat_next-policy.prob_matrix))
        update_direction = np.dot(prob_mat_next.flatten(), policy.prob_matrix.flatten()) / (np.linalg.norm(prob_mat_next) * np.linalg.norm(policy.prob_matrix))
        policy.prob_matrix = prob_mat_next
        
        # check if policy gets too high, then need to extend action space
        agg_pol = np.max(np.max(np.max(prob_mat_next, axis=0), axis=0), axis=0)
        if(agg_pol[-1:]>self._parameters._creation_thr_action):
            self._enlarge_action_space = True
            
        return update_difference, update_direction
    