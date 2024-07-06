# Karma Game Library, Copyrights Kevin Riehl 2023, <kriehl@ethz.ch>

# Imports
import numpy as np
import karma_game_library as karma

# Define Karma Game
num_agents = 200
num_types = 1
num_urgencies = 2
num_average_karma = 6

lst_init_types = np.random.randint(num_types, size=num_agents)
lst_init_urgencies = np.random.randint(num_urgencies, size=num_agents)
lst_init_karmas = 6*np.ones(num_agents)

parameters = karma.GameParameters(num_agents=num_agents,
				  num_participants=2,
				  num_types=num_types,
				  num_urgencies=num_urgencies,
				  num_average_karma=num_average_karma,
				  map_type_temp_preference={0: 0.8},
				  lst_init_types=lst_init_types,
				  lst_init_urgencies=lst_init_urgencies,
				  lst_init_karmas=lst_init_karmas,
                  func_cost=karma.templates.cost.default,
				  func_outcome=karma.templates.outcome.highest_bid,
				  func_payment=karma.templates.payment.highest_bid_to_peer,
				  func_urgency_transition=karma.templates.urgency_transition.random,
				  func_overflow_distribution=karma.templates.overflow_distribution.none,
				  func_karma_redistribution=karma.templates.karma_redistribution.none)

# Define entities for optimization
policy = karma.Policy(game_parameters=parameters, initialization="even")
state = karma.StateDistribution(game_parameters=parameters)
optimizer = karma.Optimizer(game_parameters=parameters,
                            hyper_lambda=1000,
                            hyper_dt=0.20,
                            hyper_nu=0.50,
                            func_prob_outcome=karma.templates.p_outcome.highest_bid,
                            func_prob_karma_transition=karma.templates.p_karma_transition.highest_bid_to_peer_no_redistribution,
                            func_prob_urgency_transition=karma.templates.p_urgency_transition.random)

# Compute stationary Nash equilibrium
max_iterations = 1000
threshold_state = 0.002
threshold_policy = 0.001
for iteration in range(0, max_iterations):
    delta_state, delta_policy, direct_state, direct_policy = optimizer.compute_iteration(state, policy)
    if iteration%10 == 0:
        print("Current iteration %d, delta_state=%.4f, delta_policy=%.4f" % (iteration, delta_state, delta_policy))
    if delta_state <= threshold_state and delta_policy <= threshold_policy:
        break
    
# Visualize Optimization Results: Policy and StateDistribution
import matplotlib.pyplot as plt
plt.figure("Social State in the Stationary Nash Equilibrium", figsize=(6*3,6))
plt.title("Social State in the Stationary Nash Equilibrium")
plt.subplot(1,3,1)
plt.title("Karma Distribution")
karma.visualizer.draw_karma_distribution_from_state(state=state, 
                        game_parameters=parameters)
plt.subplot(1,3,2)
plt.title("Policy (Urgency=0)")
karma.visualizer.draw_specific_policy(policy=policy, 
                                      game_parameters=parameters,
                                      atype=0, 
                                      urgency=0)
plt.subplot(1,3,3)
plt.title("Policy (Urgency=1)")
karma.visualizer.draw_specific_policy(policy=policy, 
                                      game_parameters=parameters,
                                      atype=0, 
                                      urgency=1)
plt.tight_layout()

# Save Karma Game 
state.save("state.txt")
policy.save("policy.txt")
karma.GameParameters.save(parameters, "parameters.txt")

def draw_optimization_progress(optimizer, parameters, state, policy, iteration):
    plt.rc('font', family='sans-serif') 
    plt.rc('font', serif='Arial') 
    
    xlabels = []
    counter = 0
    for s in parameters._set_state_karmas:
        if counter%5==0:
            xlabels.append(str(s))
        else:
            xlabels.append("")
        counter += 1
        
    fig = plt.figure("Optimization Results", figsize=(6,9))
    fig.set_tight_layout(True)
    plt.suptitle("Optimization Results, iteration="+str(iteration))
    plt.subplot(3,2,1)
    plt.title("Karma Distribution")
    karma.visualizer.draw_karma_distribution_from_state(state=state, game_parameters=parameters)
    plt.gca().set_xticklabels(xlabels)
    plt.subplot(3,2,2)
    plt.title("Action Distribution")
    karma.visualizer.draw_distribution_bar(parameters._set_actions, optimizer.v)
    plt.xlabel("Action")
    plt.ylabel("Probability")
    plt.subplot(3,2,3)
    plt.title("Karma Policy (T 0, U 0)")
    karma.visualizer.draw_specific_policy(policy, parameters, atype=0, urgency=0)
    plt.gca().set_xticklabels(xlabels)
    plt.subplot(3,2,4)
    plt.title("Karma Policy (T 0, U 1)")
    karma.visualizer.draw_specific_policy(policy, parameters, atype=0, urgency=1)
    plt.gca().set_xticklabels(xlabels)
    plt.subplot(3,2,5)
    plt.title("Infinite Horizon Reward (T 0, U 1)")
    karma.visualizer.draw_distribution_bar(parameters._set_state_karmas, optimizer.V[0][1])
    plt.xlabel("Karma balance")
    plt.ylabel("Expected Reward")
    plt.gca().set_xticklabels(xlabels)
    plt.subplot(3,2,6)
    plt.title("Peturb. Best Response (T 0, U 1)")
    labels_x = karma.visualizer._convert_set_to_labels(parameters._set_state_karmas)
    labels_y = karma.visualizer._convert_set_to_labels(parameters._set_actions)
    karma.visualizer.draw_heatmap(optimizer.Q[0][1].transpose(), labels_x, labels_y)
    plt.xlabel("Karma balance")
    plt.ylabel("Action")
    plt.gca().set_xticklabels(xlabels)

    # finish up
    plt.tight_layout()
draw_optimization_progress(optimizer, parameters, state, policy, 1000)


# Define Simulator
simulator = karma.Simulator(game_parameters=parameters, state=state)

# Compute Simulation
num_iterations = 10000
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
    
xlabels2 = []
counter = 0
for x in range(0,49):
    if counter%20==0:
        xlabels2.append(str(int(counter*2.5)))
    else:
        xlabels2.append("")
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
plt.gca().set_xticklabels(xlabels2)
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
