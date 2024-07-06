# Karma Game Library, Copyrights Kevin Riehl 2023, <kriehl@ethz.ch>

# Imports
import numpy as np
import karma_game_library as karma

# Define Karma Game
num_agents = 200
num_types = 1
num_urgencies = 2
num_outcomes=2
num_average_karma = 6

lst_init_types = np.random.randint(num_types, size=num_agents)
lst_init_urgencies = np.random.randint(num_urgencies, size=num_agents)
lst_init_karmas = num_average_karma*np.ones(num_agents)

parameters = karma.GameParameters(num_agents=num_agents,
				  num_participants=2,
				  num_types=num_types,
				  num_urgencies=num_urgencies,
                  num_outcomes=num_outcomes,
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
xlabels = []
counter = 0
for s in parameters._set_state_karmas:
    if counter%5==0:
        xlabels.append(str(s))
    else:
        xlabels.append("")
    counter += 1
plt.figure("Social State in the Stationary Nash Equilibrium", figsize=(3*3,3))
plt.title("Social State in the Stationary Nash Equilibrium")
plt.subplot(1,3,1)
plt.title("Karma Distribution")
karma.visualizer.draw_karma_distribution_from_state(state=state, 
                        game_parameters=parameters)    
plt.gca().set_xticklabels(xlabels)
plt.subplot(1,3,2)
plt.title("Policy (Urgency=0)")
karma.visualizer.draw_specific_policy(policy=policy, 
                                      game_parameters=parameters,
                                      atype=0, 
                                      urgency=0)
plt.gca().set_xticklabels(xlabels)
plt.subplot(1,3,3)
plt.title("Policy (Urgency=1)")
karma.visualizer.draw_specific_policy(policy=policy, 
                                      game_parameters=parameters,
                                      atype=0, 
                                      urgency=1)
plt.gca().set_xticklabels(xlabels)
plt.tight_layout()

# Save Karma Game 
state.save("state.txt")
policy.save("policy.txt")
karma.GameParameters.save(parameters, "parameters.txt")