from rel_network import RelKnowledge

import random

## \defgroup AnalyticalBlocks Analytical neuron related classes
#
# The Analytical neuron is a class that analyzes RelKnowledge instances
# to solve ambiguities
# @{
#

## Analytical neuron
class AnalyticalNeuron:

    ## The constructor
    def __init__(self):
        return

    ## Solve ambiguities
    # @param rel_knowledge_v Vector of relational knowledge
    # @retval h_id Integer. Hearing id of maximum weight relation
    def solve_ambiguity(self, rel_knowledge_v):
        # Store relation with maximum weight
        max_weight_rel = rel_knowledge_v[0]
        for element in rel_knowledge_v:
            # Get element of maximum weight
            if element.get_weight() > max_weight_rel.get_weight():
                max_weight_rel = element
            # If current maximum weight equals weight of element,
            # randomly decide to reassign max_weight_rel
            if element.get_weight() == max_weight_rel.get_weight():
                if random.randint(0,1) == 1:
                    max_weight_rel = element
        # Return hearing id of maximum weight relation
        return max_weight_rel.get_h_id()


## @}
#