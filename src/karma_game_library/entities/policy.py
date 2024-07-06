# Karma Game Library, Copyrights Kevin Riehl 2023, <kriehl@ethz.ch>

# Imports
import numpy as np
from sklearn.preprocessing import normalize
import karma_game_library

# Classes

class Policy:
    """
    A class to represent the agent policy, meaning for a specific type and 
    urgency the 2D probability matrix of karma balances (columns) and actions 
    taken (rows). The type and urgency specific 2D probability matrix will add 
    up to 1.0 for each column (for one given karma balance, the sum of 
    probabilities of different actions sum up to 1.0).

    Attributes
    ----------
    prob_matrix : np.array, indexes: [type][urgency][karma_balance][action]
        The probability matrix. Each value represents the probability, that an
        agent of a specific type and a specific urgency with a specific karma
        balance will choose a specific action. The probability matrix will add 
        up to 1.0 for each column (for one given karma balance, the sum of 
        probabilities of different actions sum up to 1.0).
    map_possible_actions : Dict[int,List[int]], [karma] -> [list of actions]
        This map maps the given karma balance to a list of possible actions
        during an interaction.
    """
    # Attributes
        # Constants
    _initialization_options = ["even", "bottom", "top"]
        # policy related
    prob_matrix = []
    _map_possible_actions = {}
    _mask_possible_actions = []
    _parameters = None
    
    def __init__(self, game_parameters: karma_game_library.game_parameters.GameParameters, initialization: str="even"):
        """
        This constructor initializes the Policy instance. The probability
        matrix will be initialized to an even distribution.

        Parameters
        ----------
        game_parameters : GameParameters
            The game parameters for the simulator.
        initialization : str
            Optional. The initialization of the policy. There are three 
            possible options: "even", "bottom", "top". Default value is 
            "even" (recommended).
            
            
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
        karma_game_library.check.non_negative(game_parameters._num_types, "num_types")
        karma_game_library.check.non_negative(game_parameters._num_urgencies, "num_urgencies")
        karma_game_library.check.non_negative(game_parameters._num_actions, "num_actions")
        if not initialization in Policy._initialization_options:
            raise Exception("Given policy initialization \""+str(initialization)+"\" not supported. Possible options are: "+str(Policy._initialization_options))
        # Initialization of matrix
        if initialization=="even":
            self.prob_matrix = np.ones((game_parameters._num_types,
                                        game_parameters._num_urgencies,
                                        game_parameters._num_state_karmas,
                                        game_parameters._num_actions))
        elif initialization=="bottom":
            self.prob_matrix = np.zeros((game_parameters._num_types,
                                        game_parameters._num_urgencies,
                                        game_parameters._num_state_karmas,
                                        game_parameters._num_actions))
            for atype in game_parameters._set_types:
                for urgency in game_parameters._set_urgencies:
                    for karma_balance in game_parameters._set_state_karmas:
                        self.prob_matrix[atype,urgency,karma_balance,0] = 1.0
        elif initialization=="top":
            self.prob_matrix = np.zeros((game_parameters._num_types,
                                        game_parameters._num_urgencies,
                                        game_parameters._num_state_karmas,
                                        game_parameters._num_actions))
            for atype in game_parameters._set_types:
                for urgency in game_parameters._set_urgencies:
                    for karma_balance in game_parameters._set_state_karmas:
                        self.prob_matrix[atype,urgency,karma_balance,min(karma_balance,game_parameters._num_actions-1)] = 1.0
        for atype in game_parameters._set_types:
            for urgency in game_parameters._set_urgencies:
                for karma_balance in game_parameters._set_state_karmas:
                    self.prob_matrix[atype,urgency,karma_balance,game_parameters._num_actions-1] = 0
                    
        # Initialization of possible actions
        self._map_possible_actions = {}
        for karma in range(0,game_parameters._num_state_karmas):
            lst_actions = []
            for action in range(0, min(karma+1, game_parameters._num_actions)):
                lst_actions.append(action)
            self._map_possible_actions[karma] = lst_actions
        self._mask_possible_actions = np.zeros((game_parameters._num_types,
                                                game_parameters._num_urgencies,
                                                game_parameters._num_state_karmas,
                                                game_parameters._num_actions))
        for karma_balance in self._map_possible_actions:
            lst_legal_actions = self._map_possible_actions[karma_balance]
            self._mask_possible_actions[:,:,karma_balance,lst_legal_actions] = 1.0
        # Finalize
        self.autocorrect()
        self._parameters = game_parameters
        
    def save(self, target_file: str):
        """
        This function saves the policy to a file.
        
        Parameters
        ----------
        target_file : str
            The file where to store the policy.
        
        Returns
        -------
        None
        """
        np.save(target_file, self.prob_matrix)
        
    def load(self, source_file: str):
        """
        This function loads the policy from a file.
        
        Parameters
        ----------
        source_file : str
            The file where to load the policy from.
        
        Returns
        -------
        None
        """
        self.prob_matrix = np.load(source_file)
        
    def validate(self):
        """
        This function validates whether the policy is valid. This means, that
        for each type and urgency specific 2D probability matrix, the columns
        add up to 1.0 (for one given karma balance, the sum of probabilities 
        of different actions sum up to 1.0). Moreover, it checks whether all
        actions are valid (one cannot bid more in an auction then one has 
        karma). If the outcome of the validation is negative you could try to 
        call function autocorrect().

        Parameters
        ----------
        None

        Returns
        -------
        valid : bool
            Whether the policy is valid or not.
        """
        for x in range(0, self.prob_matrix.shape[0]):
            for y in range(0, self.prob_matrix.shape[1]):
                for z in range(0, self.prob_matrix.shape[2]):
                    sum_probs = np.sum(self.prob_matrix[x][y][z])
                    if(not (sum_probs>self._parameters._validation_tol_dn and sum_probs<self._parameters._validation_tol_up)):
                        return False
        return True

    def copy_policy(self, other_policy):
        """
        This function copies the prob_matrix from other_policy into this
        instances prob_matrix.

        Parameters
        ----------
        other_policy : Policy
            Another policy from which the content shall be copied into this 
            instance.

        Raises
        ------
        Exception
            In case the Policy other_policy is invalid a specific exception 
            will be raised.
            
        Returns
        -------
        None
        """
        if(not other_policy.validate()):
            raise Exception("Given other_policy is invalid, therfore refused to copy()!")
        self.prob_matrix = other_policy.prob_matrix.copy()
            
    def autocorrect(self):
        """
        This function corrects the prob_matrix after changes by the user. 
        This involves (i) eliminating illegal actions for specific karma 
        balances as defined during initialization by map_possible_actions 
        (setting their probability to zero), and (ii) normalizing the columns, 
        so that the type and urgency specific 2D probability matrix will add 
        up to 1.0 for each column.

        Parameters
        ----------
        None

        Returns
        -------
        changed : bool
            Whether the prob_matrix was changed by the function or not.
        """
        prob_matrix_before = self.prob_matrix.copy()
        # mask illegal actions out
        self.prob_matrix = np.multiply(self.prob_matrix, self._mask_possible_actions)
        # normalize 
        for x in range(0, self.prob_matrix.shape[0]):
            for y in range(0, self.prob_matrix.shape[1]):
                self.prob_matrix[x][y] = normalize(self.prob_matrix[x][y], axis=1, norm="l1")
        # changed?
        changed = np.sum(np.subtract(self.prob_matrix, prob_matrix_before))==0
        return changed

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
        # probability matrix
        new_prob_matrix = np.zeros((self._parameters._num_types,
                                    self._parameters._num_urgencies,
                                    self._parameters._num_state_karmas+1,
                                    self._parameters._num_actions))
        for atype in self._parameters._set_types:
            for urgency in self._parameters._set_urgencies:
                for karma in self._parameters._set_state_karmas:
                    for action in self._map_possible_actions[karma]:
                        new_prob_matrix[atype][urgency][karma][action] = self.prob_matrix[atype][urgency][karma][action]
                # new policy columns is just last one
                new_prob_matrix[atype][urgency][self._parameters._num_state_karmas] = new_prob_matrix[atype][urgency][self._parameters._num_state_karmas-1]
        self.prob_matrix = new_prob_matrix
        # possible actions
        self._map_possible_actions = {}
        for karma in range(0,self._parameters._num_state_karmas+1):
            lst_actions = []
            for action in range(0, min(karma+1, self._parameters._num_actions)):
                lst_actions.append(action)
            self._map_possible_actions[karma] = lst_actions
        self._mask_possible_actions = np.zeros((self._parameters._num_types,
                                                self._parameters._num_urgencies,
                                                self._parameters._num_state_karmas+1,
                                                self._parameters._num_actions))
        for karma_balance in self._map_possible_actions:
            lst_legal_actions = self._map_possible_actions[karma_balance]
            self._mask_possible_actions[:,:,karma_balance,lst_legal_actions] = 1.0
        # finalize
        self.autocorrect()
    
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
        # probability matrix
        new_prob_matrix = np.zeros((self._parameters._num_types,
                                    self._parameters._num_urgencies,
                                    self._parameters._num_state_karmas,
                                    self._parameters._num_actions+1))
        for atype in self._parameters._set_types:
            for urgency in self._parameters._set_urgencies:
                for karma in self._parameters._set_state_karmas:
                    for action in self._map_possible_actions[karma]:
                        new_prob_matrix[atype][urgency][karma][action] = self.prob_matrix[atype][urgency][karma][action]
        for atype in self._parameters._set_types:
            for urgency in self._parameters._set_urgencies:
                for karma_balance in self._parameters._set_state_karmas:
                    new_prob_matrix[atype,urgency,karma_balance,self._parameters._num_actions] = 0
        self.prob_matrix = new_prob_matrix
        # possible actions
        self._map_possible_actions = {}
        for karma in range(0,self._parameters._num_state_karmas):
            lst_actions = []
            for action in range(0, min(karma+1, self._parameters._num_actions+1)):
                lst_actions.append(action)
            self._map_possible_actions[karma] = lst_actions
        self._mask_possible_actions = np.zeros((self._parameters._num_types,
                                                self._parameters._num_urgencies,
                                                self._parameters._num_state_karmas,
                                                self._parameters._num_actions+1))
        for karma_balance in self._map_possible_actions:
            lst_legal_actions = self._map_possible_actions[karma_balance]
            self._mask_possible_actions[:,:,karma_balance,lst_legal_actions] = 1.0
        # finalize
        self.autocorrect()
