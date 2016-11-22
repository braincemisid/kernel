import pickle

from multiclass_single_layer_network import MulticlassSingleLayerNetwork
from internal_state import InternalState

## \addtogroup Intentions
#  Unconscious filtering block
# @{

## The DecisionByPredictionBlock is a class aimed at modeling how decisions can be
#    made through prediction. The brain seems to simulate a world and internal (self) model
#    in order to predict the outcomes of the different options it is faced to. The decision is
#    thus made by selecting the option that produces the closest outcome to a desired world and
#    internal state
class DecisionByPredictionBlock:

    ## Inputs number
    INPUTS_NUMBER = 3
    ## Input size (Each input's number of variables)
    INPUT_SIZE = 3
    ## Internal state size
    INTERNAL_STATE_SIZE = 3
    ## Output size
    OUTPUT_SIZE = 3


    ## Create multiclass perceptron network with input size equal to
    # the size of a BCF(input size) plus the size of the internal state
    def __init__(self):

        self.predictive_net = MulticlassSingleLayerNetwork( DecisionByPredictionBlock.INPUT_SIZE
                                                         + DecisionByPredictionBlock.INTERNAL_STATE_SIZE,
                                                         DecisionByPredictionBlock.OUTPUT_SIZE)
        self.desired_state = None
        self.inputs = None
        self.output = None
        self.internal_state = None
        # Vector of predicted outcomes for the given inputs
        self.predicted_outcomes = None
        # Vector of calculated distances between the predicted outcomes and
        # the desired state
        self.distances = None

    ## Set desired state
    # @param desired_state InternalState
    def set_desired_state(self, desired_state):
        if isinstance(desired_state, InternalState):
            self.desired_state = desired_state
            return True
        return False

    ## Get desired state
    # @retval desired_state InternalState.
    def get_desired_state(self):
        return self.desired_state

    ## Set network inputs
    # @param inputs For example, *[[0.5, 0.9, 0.2],[0.5, 0.9, 0.3],[0.4, 0.7, 0.9]]*
    def set_inputs(self, inputs):
        if len(inputs) != DecisionByPredictionBlock.INPUTS_NUMBER:
            return False
        self.inputs = inputs

    ## Get network inputs
    # @retval inputs For example *[[0.5, 0.9, 0.2],[0.5, 0.9, 0.3],[0.4, 0.7, 0.9]]*
    def get_inputs(self):
        return self.inputs

    ## Get network output
    # @retval output Integer. Index of selected input
    def get_output(self):
        self._make_decision()
        return self.output

    ## Get predicted outcomes calculated in the decision taking process
    # @retval predicted_outcomes A vector, for instance, *[[0.43, 0.31, 0.35], [0.44, 0.32, 0.37], [0.52, 0.37, 0.45]]*,
    # where each component is the predicted new internal state componentes (BCF) to get if the corresponding decision
    # (0 for input 0, 1 for input 1, and so on) is taken.
    def get_predicted_outcomes(self):
        return self.predicted_outcomes

    ## Get distances (absolute value of de difference of the components) between the desired stated and every
    # predicted outcome
    # @retval distances Floats vector.
    def get_distances(self):
        return self.distances

    ## Set the entity's internal state
    # @param internal_state InternalState. New internal state
    def set_internal_state(self, internal_state):
        if isinstance(internal_state, InternalState):
            self.internal_state = internal_state
            return True
        return False

    ## Re-model or re-train predictive net
    # @param training_set A vector of 2-tuples, where each tuple is as in the following example:
    # *([0.61, 0.18, 0.16, 0.10, 0.13, 0.21], [0.36, 0.16, 0.19])*
    #
    # The first element of the tuple is a vector of real numbers between 0 and 1, which first three elements are the
    # components of the internal state, and the last three elements are the components of the BCF associated with this
    # particular decision.
    #
    # The las element of the tuple is the new internal state to get if the corresponding decision is taken.
    def remodel_predictive_net(self, training_set):
        self.predictive_net.training(training_set)

    ## Make a decision where the decision that yields the closest prediction to the desired state is taken
    # The result of the decision is stored in the output variable of the class.
    def _make_decision(self):
        # No decision is made if there is lack of information
        if self.inputs is None or self.desired_state is None or self.internal_state is None:
            return False
        predicted_outcomes = []
        for input in self.inputs:
            net_input = self.internal_state.get_state() + input
            self.predictive_net.set_inputs(net_input)
            predicted_outcomes.append( self.predictive_net.get_outputs() )
        decision = self._select_from_predicted_outcomes(predicted_outcomes)
        self.output = decision
        self.predicted_outcomes = predicted_outcomes
        return True

    ## Select the closest outcome to desired state
    # @param predicted_outcomes Set of predictive_net outcomes that correspond to input vector
    # @retval: index Integer. Index of predicted_outcome
    def _select_from_predicted_outcomes(self, predicted_outcomes):
        self.distances = []
        for outcome in predicted_outcomes:
            distance = 0
            for outcome_j, desired_j in zip(outcome,self.desired_state.get_state()):
                distance += abs(desired_j-outcome_j)
            self.distances.append(distance)
        return self.distances.index(min(self.distances))


## @}
#


# Tests
if __name__ == '__main__':

    import random

    desired_state = InternalState()
    desired_state.set_state([0.5,1,1])
    internal_state = InternalState([0.5,0.5,0.5])

    decision_prediction = DecisionByPredictionBlock()
    decision_prediction.set_desired_state(desired_state)

    # Create a random training set so that the net can learn the relation prediction = (ei + choice.bcf)/2
    # We require a minimum of 18 points
    training_set = []
    for index in range(10):
        ei = [random.random(), random.random(), random.random() ]
        choice_bcf = [ random.random(), random.random(), random.random()]
        prediction = [ ei_j/2.0 + choice_bcf_j/2.0 for ei_j, choice_bcf_j in zip(ei, choice_bcf) ]
        training_set.append( (ei + choice_bcf, prediction ) )

    decision_prediction.remodel_predictive_net(training_set)

    decision_prediction.set_internal_state(internal_state)
    decision_prediction.inputs = [[0.5, 0.9, 0.2],[0.5, 0.9, 0.3],[0.4, 0.7, 0.9]]

    print "Training set = ", training_set
    print "Decision = ", decision_prediction.get_output()
    print "Distances = ", decision_prediction.distances
    print "Predicted outcomes = ", decision_prediction.predicted_outcomes
    print "Weights = ", decision_prediction.predictive_net.weights

    # Weights to posibly be used as starting points for the brainCEMISID
    # Weights =  [[0.18812028041881285, 0.13849727997709485, 0.1381004345719828, 0.14446560005017783,
    # 0.12956668791495393, 0.12336769506630386],
    # [0.1455684236435773, 0.12788199329931116, 0.10319785645118171, 0.11201382530597757, 0.13567769257922965,
    # 0.08704799225129042],
    # [0.14427679571022395, 0.12150567866945464, 0.2118510950796081, 0.13439692578417978, 0.0900649700576292,
    # 0.17960324138197806]]
