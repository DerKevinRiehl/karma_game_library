Documentation
##############################################

**Version:** 1.0.0

**Historical documentation:** `1.0.0 <http://google.com>`_

**Useful links:** `Source Repository <http://google.com>`_  | `Issue Tracker & Support <http://google.com>`_  | `Contact & Mailing List <http://google.com>`_
 


Objects
##############################################


GameParameters
**********************************************

.. autoclass:: karma_game_library.entities.game_parameters.GameParameters
   :members:

StateDistribution
**********************************************
.. autoclass:: karma_game_library.StateDistribution
   :members:
   :exclude-members: dist_matrix
   
Policy
**********************************************
.. autoclass:: karma_game_library.Policy
   :members:
   :exclude-members: prob_matrix
   
|
|
|
|
   
Algorithms
##############################################

Simulator
**********************************************
.. autoclass:: karma_game_library.Simulator
   :members:
   :exclude-members: epoch_counter, last_participants, last_actions, last_outcomes, all_epoch_participants, all_epoch_outcomes
   
Optimizer
**********************************************
.. autoclass:: karma_game_library.Optimizer
   :members:
   :exclude-members: recorder, hyper_lambda, hyper_dt, hyper_nu, func_prob_outcome, func_prob_karma_transition, func_prob_urgency_transition, v, gamma, kappa, xi, rho, R, P, V, Q, pi_pbp, delta_state, delta_policy
   
|
|
|
|

Modelling Templates
##############################################

Game Logic Functions
**********************************************

Cost
==============================================
.. automodule:: karma_game_library.templates.cost.cost
   :members:
   
Outcome
==============================================
.. automodule:: karma_game_library.templates.outcome.outcome
   :members:

Payment
==============================================
.. automodule:: karma_game_library.templates.payment.payment
   :members:
   
Urgency Transition
==============================================
.. automodule:: karma_game_library.templates.urgency_transition.urgency_transition
   :members:
      
Overflow Distribution
==============================================
.. automodule:: karma_game_library.templates.overflow_distribution.overflow_distribution
   :members:
   
Karma Redistribution
==============================================
.. automodule:: karma_game_library.templates.karma_redistribution.karma_redistribution
   :members:
   
Probabilistic Functions
**********************************************

Outcome
==============================================
.. automodule:: karma_game_library.templates.p_outcome.p_outcome
   :members:
   
Karma Transition
==============================================
.. automodule:: karma_game_library.templates.p_karma_transition.p_karma_transition
   :members:

Urgency Transition
==============================================
.. automodule:: karma_game_library.templates.p_urgency_transition.p_urgency_transition
   :members:

|
|
|
|

Visualization Utilities
##############################################

.. automodule:: karma_game_library.utils.visualizer
   :members:
   
   
   
   


|
|
|
|

Indices and tables
##############################################

* :ref:`genindex`
* :ref:`search`

