import pickle
import random


from decision_by_prediction_block import DecisionByPredictionBlock
from internal_state import BiologyCultureFeelings, InternalState


## \defgroup Intentions Intentions related classes
#
# These are the set of classes that support decision making
# for intentions creation
# @{


##
# The Conscious Decisions Block is in charge of 'rationally' evaluating
#     current internal state and goals in order to make decisions. The quotes in
#     'rationally' stand for the agent trying to choose what he thinks is the best
#     course of action according to past experiences, but not what is really
#     best in the mathematical sense (objective function). The result from
#     the last decision taken can be fed back for evaluation and evolution of its
#     behaviour
#
class ConsciousDecisionsBlock():

    ## Input size
    INPUT_SIZE = 9

    ## Class constructor
    def __init__(self):
        # Desired state
        self.desired_state = InternalState()
        self.desired_state.set_state([0.5,1,1])
        # Initial internal state
        self.internal_state = InternalState([0.5,0.5,0.5])

        # Decision by prediction network
        self.decision_prediction_block = DecisionByPredictionBlock()
        self.decision_prediction_block.set_desired_state(self.desired_state)
        self.decision_prediction_block.set_internal_state(self.internal_state)

        # DEFAULT TRAINING, IT CAN LATER BE OVERRIDEN
        # Create a random training set so that the net can learn the relation prediction = (ei + choice.bcf)/2
        # We require a minimum of 18 points
        training_set = []
        for index in range(20):
            ei = [random.random(), random.random(), random.random()]
            choice_bcf = [random.random(), random.random(), random.random()]
            prediction = [ei_j / 2.0 + choice_bcf_j / 2.0 for ei_j, choice_bcf_j in zip(ei, choice_bcf)]
            training_set.append((ei + choice_bcf, prediction))

        # Remodel predictive net
        self.decision_prediction_block.remodel_predictive_net(training_set)

        self._inputs = None
        self._new_inputs = False
        self.decision = None
        self._last_decision_type = None
        self._last_selected_input = None
        self._last_decision_internal_state = None

    ## Set desired state
    # @param desired_state InternalState. Desired internal state
    # @retval result Boolean. True if desired state correctly set, False in any other case
    def set_desired_state(self, desired_state):
        if desired_state.__class__ == InternalState:
            self.desired_state = desired_state
            return True
        return False

    ## Get desired state
    # @retval desired_state InternalState. Stored desired state
    def get_desired_state(self):
        return self.desired_state

    ## Set internal state
    # @param internal_state InternalState. New internal state.
    # @retval result Boolean. True if internal state correctly set, False in any other case
    def set_internal_state(self, internal_state):
        if internal_state.__class__ == InternalState:
            self.internal_state = internal_state
            return True
        return False

    ## Get internal state
    # @retval internal_state InternalState. Stored internal state
    def get_internal_state(self):
        return self.internal_state

    ## Get last decision type
    # @retval last_decision_type Enumeration with values 'BIOLOGY_ALARM', 'FREE_WILL' or 'PREDICTED'
    def get_last_decision_type(self):
        return self._last_decision_type

    ## Set conscious decisions block inputs
    # @param inputs: vector of inputs of the form [bcf, bcf, bcf]
    # (1st input, 2nd input, 3rd input)
    def set_inputs(self, inputs):
        self._inputs = inputs
        self._new_inputs = True

    ## Get block inputs
    # @retval inputs vector of the form [bcf, bcf, bcf]
    def get_inputs(self):
        return self._inputs

    ## Get decision
    # @ret_val decision Integer. Index of the selected input.
    def get_decision(self):
        self._calc_decision()
        return self.decision

    def _calc_decision(self):
        if not self._new_inputs:
            return
        # If there is a biology alarm, make best decision for biology
        if self.internal_state.biology_alarm():
            self._make_best_biology_decision()
            self._last_decision_type = "BIOLOGY_ALARM"
        # Else, make either a decision by simulation or
        # a random (Free-will like) decision
        else:
            predicted_decision = self._decision_by_prediction()
            free_will_decision = self._free_will_decision()
            self.decision = self._select_predicted_or_free_will(predicted_decision, free_will_decision)
            self._new_inputs = False
        # Store last selected input
        self._last_selected_input = self._inputs[self.decision].get_state()
        # Store internal state of last decision
        self._last_decision_internal_state = self.internal_state.get_state()

    def _make_best_biology_decision(self):
        # Biology in alarm due to violated upper threshold
        index_best_biology = 0
        if self.internal_state.biology_up_alarm():
            # Select option with the lowest biology val
            for index in range(len(self._inputs)):
                if self._inputs[index].get_biology() < self._inputs[index_best_biology].get_biology():
                    index_best_biology = index
        # Biology in alarm due to violated lower threshold
        else:
            # Select option with the geatest biology val
            for index in range(len(self._inputs)):
                if self._inputs[index].get_biology() > self._inputs[index_best_biology].get_biology():
                    index_best_biology = index
        self.decision = index_best_biology
        self._new_inputs = False

    def _decision_by_prediction(self):
        prediction_inputs = [self._inputs[0].get_state(), self._inputs[1].get_state(), self._inputs[2].get_state()]
        self.decision_prediction_block.set_internal_state(self.internal_state)
        self.decision_prediction_block.set_desired_state(self.desired_state)
        self.decision_prediction_block.set_inputs(prediction_inputs)
        return self.decision_prediction_block.get_output()

    ## If free will really exists, it is no random for the person who decides. But it can't be
    #   predicted by others, i.e., for an external observer, its result is a random one. And that's what we are,
    #    external observers of the kernel
    def _free_will_decision(self):
        return random.randint(0,2)

    ## Most of the time, decisions are not concerned with free will, but with previous experiences
    def _select_predicted_or_free_will(self, predicted_decision, free_will_decision):
        rand_number = random.random()
        if rand_number > 0.90:
            self._last_decision_type = "FREE_WILL"
            return free_will_decision
        else:
            self._last_decision_type = "PREDICTED"
            return predicted_decision

    ## Train predictive network
    # @param training_set Vector of the form [ [bcf_is bcf_i], bcf_o ]
    # where bcf_is is the internal state BCF
    # and bcf_i is the input BCF
    # and bcf_o is the expected or predicted new internal state bcf
    def training(self, training_set ):
        self.decision_prediction_block.remodel_predictive_net(training_set)

    ## Feedback a new internal state to prediction network
    # @param new_internal_state InternalState. New internal state after making a decision and acting on environment
    def feedback(self, new_internal_state):
        if not self.set_internal_state(new_internal_state):
            return
        # Only the prediction network can be affected by feedback
        if self._last_decision_type != "PREDICTED":
            return
        predictive_net_training_data = [(self._last_decision_internal_state + self._last_selected_input,
                                    self.internal_state.get_state())]
        self.decision_prediction_block.remodel_predictive_net(predictive_net_training_data)

    @classmethod
    ## Serialize object and store in given file
    # @param cls CulturalNetwork class
    # @param obj CulturalNetwork object to be serialized
    # @param name Name of the file where the serialization is to be stored
    def serialize(cls, obj, name):
        pickle.dump(obj, open(name, "wb"))

    @classmethod
    ## Deserialize object stored in given file
    # @param cls CulturalNetwork class
    # @param name Name of the file where the object is serialize
    def deserialize(cls, name):
        try:
            retval = pickle.load(open(name, "rb"))
        except IOError:
            retval = ConsciousDecisionsBlock()
        return retval


## @}
#


# Tests
if __name__ == '__main__':
    cdb = ConsciousDecisionsBlock()


    # FREE WILL DECISIONS 20% of the time
    # Inputs
    i0 = BiologyCultureFeelings([0.5,0.9,0.9])
    i1 = BiologyCultureFeelings([0.5,0.9,0.9])
    i2 = BiologyCultureFeelings([0.4,0.7,0.9])
    inputs = [i0, i1, i2]
    cdb.set_inputs(inputs)
    cdb.internal_state.set_state([0.5,0.5,0.5])

    # Show free will decisions
    print('-'*60)
    print('FREE WILL')
    print 'Inputs: ', inputs[0].get_state(), inputs[1].get_state(), inputs[2].get_state()
    for i in range(10):
        cdb.set_inputs(inputs)
        d = cdb.get_decision()
        print "Decision is: ", d, " made by ", cdb.get_last_decision_type()

    # BIOLOGY ALARMS
    cdb.internal_state.set_state([0.9,1,1])
    cdb.set_inputs(inputs)
    print('-' * 60)
    print 'BIOLOGY ALARM'
    print 'Internal state: ', cdb.internal_state.get_state()
    print 'Decision is: ', cdb.get_decision(), ' made by ', cdb.get_last_decision_type()
    cdb.internal_state.set_state([0.1, 1, 1])
    cdb.set_inputs(inputs)
    print 'Internal state: ', cdb.internal_state.get_state()
    print 'Decision is: ', cdb.get_decision(), ' made by ', cdb.get_last_decision_type()

    # FEEDBACK TEST
    test = True
    internal_state = InternalState()
    cdb.internal_state.set_state([0.5, 1, 1])
    while test:
        print 'FEEDBACK TEST'
        print('-'*60)
        i0.set_state(input('Enter input #0 ([B,C,F]): '))
        i1.set_state(input('Enter input #1 ([B,C,F]): '))
        i2.set_state(input('Enter input #2 ([B,C,F]): '))
        cdb.set_inputs([i0, i1, i2])
        print "Internal state: ", cdb.internal_state.get_state()
        print "Decision: ", cdb.get_decision(), " made by ", cdb.get_last_decision_type()
        internal_state.set_state(input('Feedback new internal state ([B,C,F]): '))
        cdb.feedback(internal_state)
        cdb.set_inputs([i0, i1, i2])
        print "New decision would be: ", cdb.get_decision(), " made by ", cdb.get_last_decision_type()
        test = input("Continue testing? (True/False): " )

