# Karma Game Library, Copyrights Kevin Riehl 2023, <kriehl@ethz.ch>

# Imports
import numpy as np
from sklearn.preprocessing import normalize
import PIL.Image
import matplotlib.pyplot as plt
import numpy.typing as npt
from typing import Set, List, Iterable

# Functions

def _convert_set_to_labels(set_instance: Set):
    """
    This function converts a set to a list of string.
    
    Parameters
    ----------
    set_instance : Set
        The set to be converted.
    
    Returns
    -------
    list_return : List[str]
        The list of strings generated from the set.
    """
    return [str(x) for x in set_instance]

def draw_distribution_bar(set_karmas: Set[int], values: Iterable[float], std_errs: Iterable[float]=None, bins:int=None):
    """
    This function draws a bar chart from a given vector of values using 
    matplotlib.pyplot.

    Parameters
    ----------
    set_karmas : Set[int]
        The set_policy_karmas.
    values : Iterable[float]
        The values to plot.
    std_errs: Iterable[float]
        Optional parameter, representing standard errors. 
    bins: int
        Optional parameter, Default: None. If provided, only 'bins' ticks are
        displayed on the x axis.
    Returns
    -------
    None
    """
    labels = _convert_set_to_labels(set_karmas)
    plt.bar(labels, values, yerr=std_errs, tick_label=labels)
    plt.setp(plt.gca().get_xticklabels(), rotation=90)
    if bins is not None:
        plt.locator_params(axis='x', nbins=bins)
    
def draw_heatmap(matrix: npt.NDArray, labels_x: List[str], labels_y: List[str], cmap:str="Blues", bins:int=None):
    """
    This function draws a heatmap from a given matrix of values using 
    matplotlib.pyplot.

    Parameters
    ----------
    matrix : npt.NDArray
        The matrix to be drawn.
    labels_x : List[str]
        The list of labels for the abscissa.
    labels_y : List[str]
        The list of labels for the ordinate.
    cmap: str
        Optional parameter, representing the colormap. 
        Default is 'Blues'.

    Returns
    -------
    None
    """
    plt.imshow(matrix, cmap=cmap, interpolation="nearest", origin='lower', extent=[0,1,0,1])
    n = len(labels_x)
    plt.gca().set_xticks([1/n*(i+1) - (1/(2*n)) for i in range(0,n)])
    plt.gca().set_xticklabels(labels_x)
    plt.setp(plt.gca().get_xticklabels(), rotation=90)
    if bins is not None:
        plt.locator_params(axis='x', nbins=bins)
    n = len(labels_y)
    plt.gca().set_yticks([1/n*(i+1) - (1/(2*n)) for i in range(0,n)])
    plt.gca().set_yticklabels(labels_y)

def draw_specific_policy(policy, game_parameters, atype, urgency, bins:int=None):
    """
    This function draws the probability of certain actions according to
    karma balances as heatmap for a specific type and urgency using 
    matplotlib.pyplot.

    Parameters
    ----------
    policy : Policy
        The policy used in simulation.
    game_parameters : GameParameters
    atype : int
        The specific agent type.
    urgency : int
        The specific agent urgency.
    bins: int
        Optional parameter, Default: None. If provided, only 'bins' ticks are
        displayed on the x axis.
        
    Returns
    -------
    None
    """
    karma_action_probabilities = policy.prob_matrix[atype][urgency]
    labels_x = _convert_set_to_labels(game_parameters._set_state_karmas)
    labels_y = _convert_set_to_labels(game_parameters._set_actions)
    draw_heatmap(karma_action_probabilities.transpose(), labels_x, labels_y, bins=bins)
    plt.xlabel("Karma balance")
    plt.ylabel("Action")

def draw_karma_distribution_from_state(state, game_parameters, limit:int=None, atype: int=None, urgency:int=None, bins:int=None):
    """
    This function draw the Karma distirbution from a given state distribution
    for a specific agent type. The karma distribution shows the average across
    all urgencies, if the urgency level is unspecified (None).

    Parameters
    ----------
    state : StateDistribution
        The state distribution to draw from.
    game_parameters : GameParameters
        The game parameters used.
    limit : int
        Optional. Default is None. If not defined, the full distribution is
        displayed, otherwise only until the limit. The last bar will then
        represent the karma balances greater or equal that value.
    atype : int
        Optional. Default is None. If not defined, the average across all
        types is displayed, otherwise for the agent type specifically.
    urgency : int
        Optional. Default is None. If not defined, the average across all 
        urgencies is displayed, otherwise for the urgency level specifically.
    bins: int
        Optional parameter, Default: None. If provided, only 'bins' ticks are
        displayed on the x axis.

    Returns
    -------
    None
    """
    if limit is None:
        labels_to_draw = game_parameters._set_state_karmas
        matrix_to_draw = state.dist_matrix
    else:
        labels_to_draw = np.arange(0,limit+1).tolist()
        matrix_to_draw = state.get_limited_dist_matrix(limit=limit+1)
    if atype is None:
        matrix_to_draw = np.mean(matrix_to_draw, axis=0)
    else:
        matrix_to_draw = matrix_to_draw[atype]
    if urgency is None:
        matrix_to_draw = np.mean(matrix_to_draw, axis=0)
    else:
        matrix_to_draw = matrix_to_draw[urgency]
    draw_distribution_bar(labels_to_draw, matrix_to_draw, bins=bins)
    plt.xlabel("Karma balance")
    plt.ylabel("Share of population")

def draw_karma_distribution_from_simulator(simulator, game_parameters, bins:int=None):
    """
    This function draw the Karma distribution from a given state distribution
    for a specific agent type. The karma distribution shows the average across
    all urgencies, if the urgency level is unspecified (None).

    Parameters
    ----------
    simulator : Simulator
        The state distribution to draw from.
    game_parameters : GameParameters
        The game parameters used.
    bins: int
        Optional parameter, Default: None. If provided, only 'bins' ticks are
        displayed on the x axis.

    Returns
    -------
    None
    """
    distribution = simulator.get_karma_distribution()
    distribution = distribution / len(distribution)
    draw_distribution_bar(np.arange(0, len(distribution)), distribution, bins=bins)
    plt.xlabel("Karma balance")
    plt.ylabel("Share of population")
    
def draw_distribution_from_simulator(simulator, game_parameters, column: int, mode: str, bins:int=15):
    """
    This function draw the interaction distribution from a given state 
    distribution for a specific agent type. The karma distribution shows the 
    average across all urgencies, if the urgency level is unspecified (None).

    Parameters
    ----------
    simulator : Simulator
        The state distribution to draw from.
    game_parameters : GameParameters
        The game parameters used.
    column : int
        The column from the agent population. Each column is represented by 
        following enumeration: TYPE_COL = 0, URGENCY_COL = 1, KARMA_COL = 2, 
        CUM_COST_COL = 3, ENCOUNTERS_COL = 4.
    mode : str
        Which mode for the distribution. Possible options are: 'unique' to 
        display how often all unique elements of the population occur; 
        'histogram' to display how often values occur in predefined bins.
    bins : int
        Optional. Default=15. This defines how many bins are used for the mode
        'histogram'.
        
    Returns
    -------
    None
    """
    population = simulator._population[:, column]
    if mode=='unique':
        vals, counts = np.unique(population, return_counts=True)
    elif mode=='histogram':
        counts, vals = np.histogram(population, bins=bins)
        vals = vals[:-1]
        vals = [f'{x:.3f}' for x in vals]
    draw_distribution_bar(vals, counts)
    plt.ylabel("Occurences in population")

def draw_karma_transition_heatmap_from_simulator(simulator, game_parameters):
    """
    This function draws a karma transition heatmap from a given state 
    transition matrix of a simulation as average over types and urgencies 
    using matplotlib.pyplot.

    Parameters
    ----------
    simulator : Simulator
        The state_transition_distribution from a KarmaSimulationInstance.
    game_parameters : The game parameters
        The set_state_karmas.

    Returns
    -------
    None
    """
    mean_over_urgencies = simulator.get_state_transition_counts()
    mean_over_urgencies = np.sum(mean_over_urgencies, axis=0)
    mean_over_urgencies = np.sum(mean_over_urgencies, axis=1)
    mean_over_urgencies = normalize(mean_over_urgencies, axis=0, norm="l1")
    labels = _convert_set_to_labels(game_parameters._set_state_karmas)
    draw_heatmap(mean_over_urgencies, labels, labels)
    plt.xlabel("Karma before")
    plt.ylabel("Karma after")

def render_gif_animation(lst_image_files: List[str], target_file: str, speed:int=100, first_last_slow:bool=True):
    """
    This function reads static images from a list of image files, and stores 
    all of them in an GIF animation.
    
    Parameters
    ----------
    lst_image_files: List[str]
        A list with image files that shall be connected to a GIF animation.
    target_file : str
        The target file to store the GIF into.
    speed : int
        Optional, Defualt: 100. The time per image. The slower the faster the animation.
    first_last_slow : bool
        Optional, Default : True. This will repeat the first and the last 
        image for ten times, so the animation does not directly run.
        
    Returns
    -------
    None
    """
    frames = []
    if first_last_slow:
        for x in range(0,10):
            image = PIL.Image.open(lst_image_files[0])
            frames.append(image)
    for file in lst_image_files:
        image = PIL.Image.open(file)
        frames.append(image)
    if first_last_slow:
        for x in range(0,10):
            frames.append(frames[-1])
    frames[0].save(target_file, format='GIF',
                    append_images=frames[1:],
                    save_all=True,
                    duration=speed, loop=0)
