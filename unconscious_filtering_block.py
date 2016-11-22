from internal_state import InternalState, BiologyCultureFeelings
from cultural_network import CulturalNetwork,CulturalGroup,CulturalNeuron

## \addtogroup Intentions
#  Unconscious filtering block
# @{
class UnconsciousFilteringBlock:

    ## The constructor
    def __init__(self):
        # Takes as inputs a set of memories (Cultural Groups)
        self.inputs = None
        # Output: 3 memories (Biology, Culture, Feelings)
        self.outputs = None
        # Internal state
        self.internal_state = None
        # Desired state
        self.desired_state = None
        return

    ## Set block inputs
    # @param inputs CulturalGroup with tail knowledge of class BiologyCultureFeelings
    def set_inputs(self, inputs):
        self.inputs = inputs

    ## Get block inputs
    # @retval inputs CulturalGroup with tail knowledge of class BiologyCultureFeelings
    def get_inputs(self):
        return self.inputs

    ## Set internal state
    # @param internal_state InternalState. Entity's internal state
    def set_internal_state(self, internal_state):
        if internal_state.__class__ != InternalState:
            return False
        self.internal_state = internal_state
        return True

    ## Set desired state
    # @param desired_state InternalState. Entity's desired state
    def set_desired_state(self, desired_state):
        if desired_state.__class__ != InternalState:
            return False
        self.desired_state = desired_state
        return True

    ## Return uncounsciously filtered memories
    # @retval output Vector of three memories (cultural groups) [Biology, Culture, Feelings]
    def get_outputs(self):
        self._filter()
        return self.outputs

    ## Filter inputs by Biology, Culture and Feelings
    def _filter(self):
        best_biology = self._filter_biology()
        best_culture = self._filter_culture()
        best_feelings = self._filter_feelings()
        self.outputs = [best_biology, best_culture, best_feelings]

    ## Select best biology input
    def _filter_biology(self):
        best_biology = self.inputs[0]
        bcf = best_biology.get_tail_knowledge()
        min_distance = abs((bcf.get_biology() + self.internal_state.get_biology())/2.0 - self.desired_state.get_biology())
        for memory in self.inputs:
            # BCF for every memory is stored in the tail of the cultural group
            bcf = memory.get_tail_knowledge()
            distance = abs((bcf.get_biology() + self.internal_state.get_biology())/2.0 - self.desired_state.get_biology())
            if distance < min_distance:
                best_biology = memory
                min_distance = distance
        return best_biology

    ## Select best culture input
    def _filter_culture(self):
        best_culture = self.inputs[0]
        bcf = best_culture.get_tail_knowledge()
        max = bcf.get_culture()
        for memory in self.inputs:
            # BCF for every memory is stored in the tail of the cultural group
            bcf = memory.get_tail_knowledge()
            if bcf.get_culture() > max:
                best_culture = memory
                max = bcf.get_culture()
        return best_culture

    ## Select best feelings input.
    def _filter_feelings(self):
        best_feelings = self.inputs[0]
        bcf = best_feelings.get_tail_knowledge()
        max = bcf.get_feelings()
        for memory in self.inputs:
            # BCF for every memory is stored in the tail of the cultural group
            bcf = memory.get_tail_knowledge()
            if bcf.get_feelings() > max:
                best_feelings = memory
                max = bcf.get_feelings()
        return best_feelings
## @}
#


# Tests
if __name__ == '__main__':

    #Memories
    MEMORIES_COUNT = 6
    memories = [CulturalGroup() for i in range(MEMORIES_COUNT)]
    import random
    bcf = []
    for i in range(MEMORIES_COUNT):
        memories[i].bum()
        memories[i].learn(i)
        bcf.append(BiologyCultureFeelings())
        new_state = [random.random(), random.random(), random.random()]
        bcf[i].set_state(new_state)
        memories[i].clack(bcf[i])
        print "Memory ", i, " bcf is", memories[i].get_tail_knowledge().get_state()

    ufb = UnconsciousFilteringBlock()
    internal_state = InternalState()
    internal_state.set_state([0.5, 1, 1])
    ufb.set_internal_state(internal_state)
    desired_state = InternalState()
    desired_state.set_state([0.5, 1, 1])
    ufb.set_desired_state(desired_state)
    ufb.set_inputs(memories)
    outputs = ufb.get_outputs()
    print "UFB outputs are: "
    for output in outputs:
        print output.group[0].get_knowledge()
