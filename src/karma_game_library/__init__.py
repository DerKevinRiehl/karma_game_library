# Karma Game Library, Copyrights Kevin Riehl 2023, <kriehl@ethz.ch>

# Imports
from karma_game_library.utils import visualizer
from karma_game_library.utils import check

from karma_game_library import templates

from karma_game_library.entities import game_parameters
from karma_game_library.entities import state_distribution
from karma_game_library.entities import policy
GameParameters = game_parameters.GameParameters
StateDistribution = state_distribution.StateDistribution
Policy = policy.Policy

from karma_game_library.algorithms import simulator
from karma_game_library.algorithms import optimizer
Simulator = simulator.Simulator
Optimizer = optimizer.Optimizer

# Functions
def version():
    print("Karma Game Library, v1.0.0 successfully installed")