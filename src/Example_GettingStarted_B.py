# Karma Game Library, Copyrights Kevin Riehl 2023, <kriehl@ethz.ch>

# Imports
import karma_game_library as karma

# Load Karma Game
parameters = karma.GameParameters.load("parameters.txt")
state = karma.StateDistribution(game_parameters=parameters)
state.load("state.txt.npy")
policy = karma.Policy(game_parameters=parameters, initialization="even")
policy.load("policy.txt.npy")

# Define Simulator
simulator = karma.Simulator(game_parameters=parameters, state=state)

# Compute Simulation
num_iterations = 1000
for i in range(0, num_iterations):
    simulator.begin_epoch()

    # option 1: peer_selection_random()
    participants = simulator.peer_selection_random()
    simulator.play_interaction(policy,participants)

    epoch_counter = simulator.close_epoch()
    print(epoch_counter)
    
# Visualize Karma Game after 1000 iterations
xlabels = []
counter = 0
for s in parameters._set_state_karmas:
    if counter%5==0:
        xlabels.append(str(s))
    else:
        xlabels.append("")
    counter += 1
import matplotlib.pyplot as plt
plt.figure("Karma Game Simulation", figsize=(3*2,3*2))
plt.rc('font', family='sans-serif') 
plt.rc('font', serif='Arial') 
plt.title("Karma Game Simulation")
plt.subplot(2,2,1)
plt.title("Karma Distribution")
karma.visualizer.draw_karma_distribution_from_simulator(simulator, parameters)
plt.gca().set_xticklabels(xlabels)
plt.subplot(2,2,2)
plt.title("Encounter Distribution")
karma.visualizer.draw_distribution_from_simulator(simulator, parameters, simulator._ENCOUNTERS_COL, 'unique')
plt.xlabel("Number of interactions")
plt.subplot(2,2,3)
plt.title("Cum. Cost Distribution")
karma.visualizer.draw_distribution_from_simulator(simulator, parameters, simulator._CUM_COST_COL, 'histogram')
plt.xlabel("Cum. Cost Intervals")
plt.tight_layout()
plt.subplot(2,2,4)
plt.title("Karma Balance Transitions")
x = karma.visualizer.draw_karma_transition_heatmap_from_simulator(simulator, parameters)
plt.gca().set_xticklabels(xlabels)
plt.gca().set_yticklabels(xlabels)
plt.tight_layout()
