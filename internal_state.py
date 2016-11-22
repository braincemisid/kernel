import pickle

## \defgroup BCF BCF classes
#
# There are two BCF classes. The first one, BiologyCultureFeelings
# models the state of an entity by using a three elements vector
# which values go from 0 to 1 and correspond to the Biology, Culture and
# Feelings situation of the entity. The second one, is the InternalState class
# that inherits from the BiologyCultureFeelings class and provides averaging methods
# to modify a given state.
# @{
#

## BiologyCultureFeelings
# models the state of an entity by using a three elements vector
# which values go from 0 to 1 and correspond to the Biology, Culture and
# Feelings situation of the entity
class BiologyCultureFeelings:

    ## Number of variables = 3 (Biology, Culture, Feelings)
    VARIABLES_NUMBER = 3

    ## The constructor
    def __init__(self, initial_state=[0.5,1,1]):
        self.biology = None
        self.culture = None
        self.feelings = None
        self.set_biology(initial_state[0])
        self.set_culture(initial_state[1])
        self.set_feelings(initial_state[2])
        return

    ## Get biology state
    # @retval biology Float from 0 to 1.
    def get_biology(self):
        return self.biology

    ## Get culture state
    # @retval culture Float from 0 to 1.
    def get_culture(self):
        return self.culture

    ## Get feelings state
    # @retval feelings Float from 0 to 1.
    def get_feelings(self):
        return self.feelings

    ## Set biology state
    # @param val Float form 0 to 1. New biology state value.
    def set_biology(self, val):
        if 0 <= val <= 1:
            self.biology = val
            return True
        return False

    ## Set culture state
    # @param val Float form 0 to 1. New culture state value.
    def set_culture(self, val):
        if 0 <= val <= 1:
            self.culture = val
            return True
        return False

    ## Set feelings state
    # @param val Float form 0 to 1. New feelings state value.
    def set_feelings(self, val):
        if 0 <= val <= 1:
            self.feelings = val
            return True
        return False

    ## Set state
    # @param @vals 0 to 1 floats vector with the new state
    def set_state(self, vals):
        if len(vals) != BiologyCultureFeelings.VARIABLES_NUMBER:
            return False
        return self.set_biology(vals[0]) and self.set_culture(vals[1]) and self.set_feelings(vals[2])

    ## Get state
    # @retval state Entity's state
    def get_state(self):
        return [self.get_biology(), self.get_culture(), self.get_feelings()]

## his class represents a very simplified version of an entity's
#    internal state. It is a type of BCF, because such state has biological,
#    cultural and emotional (here reduced to its less primitive and reflected
#    counterpart 'feelings') components. Average methods are added to model the effect
#   of external influences
class InternalState(BiologyCultureFeelings):

    ## Biology upper threshold
    BIOLOGY_UPPER_THRESHOLD = 0.8
    ## Biology lower threshold
    BIOLOGY_LOWER_THRESHOLD = 0.2

    ## The constructor
    def __init__(self, initial_state=None):
        if initial_state is None:
            BiologyCultureFeelings.__init__(self)
        else:
            BiologyCultureFeelings.__init__(self, initial_state)

    ## Average biology state with a given value
    # @param val Float from 0 to 1 to be averaged with the stored biology value.
    def average_biology(self, val):
        if val < 0 or val > 1:
            return False
        self.biology = (self.biology + val) / 2.0
        return True

    ## Average culture state with a given value
    # @param val Float from 0 to 1 to be averaged with the stored culture value.
    def average_culture(self, val):
        if val < 0 or val > 1:
            return False
        self.culture = (self.culture + val) / 2.0
        return True

    ## Average feelings state with a given value
    # @param val Float from 0 to 1 to be averaged with the stored feelings value.
    def average_feelings(self, val):
        if val < 0 or val > 1:
            return False
        self.feelings = (self.feelings + val) / 2.0
        return True

    ## Average complete state with a given vector of values
    # @param states_vector 0 to 1 floats vector to be averaged with the stored state.
    def average_state(self, states_vector ):
        if len(states_vector) != 3:
            return False
        for element in states_vector:
            if element > 1 or element < 0:
                return False
        self.average_biology(states_vector[0])
        self.average_culture(states_vector[1])
        self.average_feelings(states_vector[2])
        return True

    ## Return True if there is a biology alarm, i.e., if the biology component of the
    # internal state is above (below) the BIOLOGY_UPPER_THRESHOLD (BIOLOGY_LOWER_THRESHOLD)
    # @retval alarm Boolean. True if there is an alarm, False in any other case.
    def biology_alarm(self):
        if (self.biology >= InternalState.BIOLOGY_UPPER_THRESHOLD or
                self.biology <= InternalState.BIOLOGY_LOWER_THRESHOLD):
            return True
        return False

    ## Return True if there is an upper biology alarm, i.e., if the biology component of the
    # internal state is above the BIOLOGY_UPPER_THRESHOLD
    # @retval up_alarm Boolean. True if there is an upper biology alarm, False in any other case.
    def biology_up_alarm(self):
        if self.biology >= InternalState.BIOLOGY_UPPER_THRESHOLD:
            return True
        return False


## @}
#


# Tests
if __name__ == '__main__':
    ie = InternalState()
    print ie.get_state()

    ie.set_biology(1)
    ie.set_culture(1)
    ie.set_feelings(1)
    print ie.get_state()

    ie.average_biology(0)
    ie.average_culture(0.25)
    ie.average_feelings(0.5)
    print ie.get_state()
