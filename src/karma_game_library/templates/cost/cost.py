# Karma Game Library, Copyrights Kevin Riehl 2023, <kriehl@ethz.ch>

# Imports

# Functions

def default(urgency:int, outcome:int) -> float:
    """
    The default cost function. Outcome=0 means resource not received, therefore
    costs created. Cost = urgency_level (urgency_level=0 has no costs)
    """
    if(outcome==1):
        return 0
    else:
        return urgency
