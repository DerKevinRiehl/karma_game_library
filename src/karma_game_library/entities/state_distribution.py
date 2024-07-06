# Karma Game Library, Copyrights Kevin Riehl 2023, <kriehl@ethz.ch>

# Imports
import numpy as np
import karma_game_library

# Classes

class StateDistribution:
    """
    A class to represent the state distribution of a Karma system. This means, 
    the share of the population, that has a specific type, specific urgency, 
    and a specific karma balance. All shares together add up to 1.0.
    
    Attributes
    ----------
    dist_matrix : np.array, indexes: [type][urgency][karma_balance]
        The social state distribution matrix. Each value represents the share
        of the population, that has a specific type, specific urgency, and a
        specific karma balance. All shares together add up to 1.0.  Note, that
        the share for the last karma balance index represents agents with a 
        karma balance of exactly this value or higher.
    """
    
    # Attributes
        # Constants
    _KARMA_COL = 2
        # Dist matrix
    dist_matrix = []
    _parameters = None
    
    def __init__(self, game_parameters: karma_game_library.game_parameters.GameParameters):
        """
        This constructor initializes the state distribution instance. 
        The distribution matrix will be initialized to an even distribution.

        Parameters
        ----------
        game_parameters : GameParameters
            The game parameters for the simulator.
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
        karma_game_library.check.non_negative(game_parameters._num_state_karmas, "num_state_karmas")
        # Creation of matrix
        self.dist_matrix = np.zeros((game_parameters._num_types,
                                    game_parameters._num_urgencies,
                                    game_parameters._num_state_karmas))
        # Initialize with a distribution as given by the user
        for agent in range(0, game_parameters._num_agents):
            atype = int(game_parameters._lst_init_types[agent])
            urgency = int(game_parameters._lst_init_urgencies[agent])
            karma = int(game_parameters._lst_init_karmas[agent])
            self.dist_matrix[atype][urgency][karma] += 1
        self.dist_matrix = self.dist_matrix / np.sum(self.dist_matrix)   
        # Finalize
        self._parameters = game_parameters
        self.autocorrect()
        
    def get_average_karma(self):
        """
        Returns the average amount of Karma.

        Returns
        -------
        average : float
            The empirical, average amount of Karma.
        """
        av_dist = np.sum(np.sum(self.dist_matrix, axis=0), axis=0)
        average = np.dot(np.arange(self._parameters._num_state_karmas), av_dist)
        return average
        
    def validate(self):
        """
        This function validates whether the social state distribution is valid. 
        This means, that all shares add up to 1.0, and that the average karma
        balance is preserved as constant.

        Parameters
        ----------
        None

        Returns
        -------
        valid : bool
            Whether the policy is valid or not.
        """
        total_sum = np.sum(self.dist_matrix)
        av_karma_ratio = self.get_average_karma()/self._parameters._num_average_karma
        condition_A =  (total_sum < self._parameters._validation_tol_up and total_sum > self._parameters._validation_tol_dn)
        condition_B = (av_karma_ratio < self._parameters._validation_tol_up and av_karma_ratio > self._parameters._validation_tol_dn)
        return (condition_A and condition_B)

    def copy_social_state_distribution(self, other_social_state):
        """
        This function copies the dist_matrix from other_social_state into this
        instances dist_matrix.

        Parameters
        ----------
        other_social_state_distribution : StateDistribution
            Another social state distribution from which the content shall be 
            copied into this instance.

        Raises
        ------
        Exception
            In case the StateDistribution other_social_state_distribution
            is invalid a specific exception will be raised.
            
        Returns
        -------
        None
        """
        if(not other_social_state.validate()):
            raise Exception("Given other_social_state_distribution is invalid, therfore refused to copy()!")
        self.dist_matrix = other_social_state.dist_matrix.copy()
            
    def autocorrect(self):
        """
        This function corrects the state_distribution after changes by the 
        user. This involves normalizing the matrix, so that all shares add
        up to 1.0, and transforming the distribution so that the average karma
        balance is preserved as constant.

        Parameters
        ----------
        None

        Returns
        -------
        changed : bool
            Whether the state_distribution was changed by the function or not.
        """
        if self.validate():
            return False
        dist_matrix_before = self.dist_matrix.copy()
        while not self.validate():
            # average karma must be constant
            av_karma = self.get_average_karma()
            if(av_karma>self._parameters._num_average_karma):
                self.dist_matrix[:,:,0:self._parameters._num_average_karma] *= self._parameters._autocorrect_factor_state
            else:
                self.dist_matrix[:,:,-self._parameters._num_average_karma:] *= self._parameters._autocorrect_factor_state
            # must sum up to 1.0
            self.dist_matrix = self.dist_matrix / np.sum(self.dist_matrix)        
        # changed?
        changed = np.sum(np.subtract(self.dist_matrix, dist_matrix_before))==0
        return changed

    def get_limited_dist_matrix(self, limit: int):
        """
        This function returns a karma balance limited state_distribution. If
        the limit is larger than the _num_state_karmas, the additional fields
        are added up with zeros. If the limit is smaller, then the state 
        distribution is aggregated, meaning that the share for the returned 
        distribution for karma=limit-1 represents the share of all balances 
        with karmas>=limit-1.

        Parameters
        ----------
        limit : int
            The limit for the limited distribution matrix.

        Returns
        -------
        limited_dist_matrix : np.array
            The limited distribution matrix.

        """
        if(limit == self.dist_matrix.shape[StateDistribution._KARMA_COL]):
            return self.dist_matrix
        else:
            dist_matrix_to_return = np.zeros((self._parameters._num_types,
                                              self._parameters._num_urgencies,
                                              limit))
            if(limit > self.dist_matrix.shape[StateDistribution._KARMA_COL]):
                for atype in self._parameters._set_types:
                    for urgency in self._parameters._set_urgencies:
                        for karma in self._parameters._set_state_karmas:
                            dist_matrix_to_return[atype][urgency][karma] = self.dist_matrix[atype][urgency][karma]
                return dist_matrix_to_return
            else:
                for atype in self._parameters._set_types:
                    for urgency in self._parameters._set_urgencies:
                        for karma in self._parameters._set_state_karmas:
                            if(karma>=limit-1):
                                dist_matrix_to_return[atype][urgency][limit-1] += self.dist_matrix[atype][urgency][karma]
                            else:
                                dist_matrix_to_return[atype][urgency][karma] = self.dist_matrix[atype][urgency][karma]
                return dist_matrix_to_return

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
        new_dist_matrix = np.zeros((self._parameters._num_types,
                                    self._parameters._num_urgencies,
                                    self.dist_matrix.shape[StateDistribution._KARMA_COL]+1))
        for atype in self._parameters._set_types:
            for urgency in self._parameters._set_urgencies:
                for karma in self._parameters._set_state_karmas:
                    new_dist_matrix[atype, urgency, karma] = self.dist_matrix[atype][urgency][karma]
        self.dist_matrix = new_dist_matrix
        
    def save(self, target_file: str):
        """
        This function saves the state distribution to a file.
        
        Parameters
        ----------
        target_file : str
            The file where to store the state distribution.
        
        Returns
        -------
        None
        """
        np.save(target_file, self.dist_matrix)
        
    def load(self, source_file: str):
        """
        This function loads the state distribution from a file into this 
        instance.
        
        Parameters
        ----------
        source_file : str
            The file where to load the state distribution from.
        
        Returns
        -------
        None
        """
        self.dist_matrix = np.load(source_file)
        