# Karma Game Library, Copyrights Kevin Riehl 2023, <kriehl@ethz.ch>

# Imports
import numpy as np
import karma_game_library as karma

# Define Karma Game
num_agents = 200
num_types = 1
num_urgencies = 2
num_outcomes = 2
num_average_karma = 6

lst_init_types = np.random.randint(num_types, size=num_agents)
lst_init_urgencies = np.random.randint(num_urgencies, size=num_agents)
lst_init_karmas = 6*np.ones(num_agents)

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

# Define visualization function
import matplotlib.pyplot as plt
def draw_optimization_progress(optimizer, parameters, state, policy, iteration, save=False, close=False, last_n=200, folder="figs"):
    xlabels = []
    counter = 0
    for s in parameters._set_state_karmas:
        if counter%5==0:
            xlabels.append(str(s))
        else:
            xlabels.append("")
        counter += 1
    fig = plt.figure("Optimization Results", figsize=(9,9))
    fig.set_tight_layout(True)
    plt.suptitle("Optimization Results, iteration="+str(iteration))
    plt.subplot(3,3,1)
    plt.title("Karma Distribution")
    karma.visualizer.draw_karma_distribution_from_state(state=state, game_parameters=parameters)
    plt.gca().set_xticklabels(xlabels)
    plt.subplot(3,3,2)
    plt.title("Karma Policy (T 0, U 0)")
    karma.visualizer.draw_specific_policy(policy, parameters, atype=0, urgency=0)
    plt.gca().set_xticklabels(xlabels)
    plt.subplot(3,3,3)
    plt.title("Karma Policy (T 0, U 1)")
    karma.visualizer.draw_specific_policy(policy, parameters, atype=0, urgency=1)
    plt.gca().set_xticklabels(xlabels)
    plt.subplot(3,3,4)
    plt.title("Infinite Horizon Reward (T 0, U 1)")
    karma.visualizer.draw_distribution_bar(parameters._set_state_karmas, optimizer.V[0][1])
    plt.xlabel("Karma balance")
    plt.ylabel("Expected Reward")
    plt.gca().set_xticklabels(xlabels)
    plt.subplot(3,3,5)
    plt.title("Peturb. Best Response (T 0, U 1)")
    labels_x = karma.visualizer._convert_set_to_labels(parameters._set_state_karmas)
    labels_y = karma.visualizer._convert_set_to_labels(parameters._set_actions)
    karma.visualizer.draw_heatmap(optimizer.Q[0][1].transpose(), labels_x, labels_y)
    plt.xlabel("Karma balance")
    plt.ylabel("Action")
    plt.gca().set_xticklabels(xlabels)
    plt.subplot(3,3,6)
    plt.title("Action Distribution")
    karma.visualizer.draw_distribution_bar(parameters._set_actions, optimizer.v)
    plt.xlabel("Action")
    plt.ylabel("Probability")
    plt.subplot(3,3,7)
    plt.title("Change of State")
    plt.plot(np.arange(max(0, iteration-last_n), iteration+1), np.asarray(optimizer.recorder)[-last_n-1:,0])
    plt.xlabel("Iteration")
    plt.ylabel("Difference to previous")
    plt.gca().get_yaxis().set_ticks([])
    if iteration==0:
        plt.xlim([0, 10])
    plt.xticks(rotation = 90)
    plt.subplot(3,3,8)
    plt.title("Change of Policy")
    plt.plot(np.arange(max(0, iteration-last_n), iteration+1), np.asarray(optimizer.recorder)[-last_n-1:,1])
    plt.xlabel("Iteration")
    plt.ylabel("Difference to previous")
    plt.gca().get_yaxis().set_ticks([])
    if iteration==0:
        plt.xlim([0, 10])
    plt.xticks(rotation = 90)
    plt.subplot(3,3,9)
    plt.title("Expected Population Costs")
    plt.plot(np.arange(max(0, iteration-last_n), iteration+1), np.asarray(optimizer.recorder)[-last_n-1:,2])
    plt.xlabel("Iteration")
    plt.ylabel("Total Costs")
    if iteration==0:
        plt.xlim([0, 10])
    plt.xticks(rotation = 90)
    # finish up
    plt.tight_layout()
    if(save):
        plt.savefig(fname=folder+"/Iteration-"+str(iteration)+".png")
    if(close):
        plt.close()
    return fig

# Compute stationary Nash equilibrium
plt.ioff()
images = []
for it in range(0,1000):
    print(it,"\t",optimizer.compute_iteration(state, policy))    
    if it%10==0:
        draw_optimization_progress(optimizer, parameters, state, policy, it, save=True, close=True, last_n=400, folder="figs")
        images.append("figs/Iteration-"+str(it)+".png")
karma.visualizer.render_gif_animation(images, "animation.gif")
