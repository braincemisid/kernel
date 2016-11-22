import pickle
from math import fabs

from neuron import Neuron

## \defgroup RbfBlocks RBF network related classes
#
# RBF network related classes are a group of classes that
# represent and use RBF knowledge, neurons and networks
# @{


## RBF knowledge. A tuple composed of a pattern, a class and a set.
# The class also provides a method for calculating the Manhattan distance
# between its pattern and the pattern of another RbfKnowledge instance
class RbfKnowledge:

    ## Size of data or knowledge in bytes
    PATTERN_SIZE = 4

    ## The constructor
    def __init__(self, rbf_pattern, rbf_class, rbf_set="NoSet"):
        self.set_pattern(rbf_pattern)
        self.set_class(rbf_class)
        self.set_set(rbf_set)

    ## Set pattern
    # @param pattern Pattern to be stored. Integers vector of size PATTERN_SIZE
    def set_pattern(self, pattern):
        self._pattern = pattern

    ## Set pattern class
    # @param rbf_class Class of the pattern.
    def set_class(self, rbf_class):
        self._class = rbf_class

    ## Set pattern set
    # @param rbf_set Set of the pattern
    def set_set(self, rbf_set):
        self._set = rbf_set

    ## Get stored pattern
    # @retval pattern Stored pattern. Integers vector of size PATTERN_SIZE
    def get_pattern(self):
        return self._pattern

    ## Get stored pattern class.
    #   @retval class Stored pattern class.
    def get_class(self):
        return self._class

    ## Get stored pattern set
    # @retval set Stored pattern set
    def get_set(self):
        return self._set

    def calc_manhattan_distance(self, pattern_or_knowledge):
        # If given parameter is of class knowledge, obtain pattern
        try:
            pattern = pattern_or_knowledge.get_pattern()
        # Else it must be a pattern
        except AttributeError:
            pattern = pattern_or_knowledge
        # Check patterns sizes are equal
        if len(pattern) != RbfKnowledge.PATTERN_SIZE:
            return False
        # Initialize distance variable to zero
        distance = 0
        # Calculate Manhattan distance
        for index in range(len(self._pattern)):
            distance += fabs(self._pattern[index] - pattern[index])
        # Return distance
        return distance

## Neuron that stores RbfKnowledge.
# This class stores an instance of RbfKnowledge at its center and uses a radius value
# to determine whether or not it recognizes a given pattern
class RbfNeuron(Neuron):
    ## Number of class instances
    instances_count = 0
    ## Default radius
    DEFAULT_RADIUS = 10

    ## Minimun radius before neuron is degraded
    MIN_RADIUS = 1

    ## Maximum radius
    MAX_RADIUS = 50

    ## Class constructor
    def __init__(self):
        super(RbfNeuron, self).__init__()
        # Neuron has no knowledge when created
        self._has_knowledge = False
        # Neuron has default radius when created
        self.set_radius(RbfNeuron.DEFAULT_RADIUS)
        # Increment number of class instances
        RbfNeuron.instances_count += 1
        # Initialize degraded state
        self._degraded = False

    ## Returns whether neuron is member of the set
    # @param test_set Set to be tested
    # @retval is_member Boolean. True if neuron is member of set, false in any other case
    def is_member(self, test_set):
        return self.get_set() == test_set

    ## Sets neuron radius
    # @param radius New neuron radius
    def set_radius(self, radius):
        self._radius = radius

    ## Get neuron radius
    # @retval radius Integer. Neuron radius
    def get_radius(self):
        return self._radius

    ## Get class of stored RbfKnowledge instance
    # @retval class RbfKnowledge class if neuron has knowledge, None in any other case
    def get_class(self):
        if self.has_knowledge():
            return self._knowledge.get_class()
        else:
            return None

    ## Get set of stored RbfKnowledge instance
    # @retval set RbfKnowledge set if neuron has knowledge, None in any other case
    def get_set(self):
        if self.has_knowledge():
            return self._knowledge.get_set()
        else:
            return None

    ## Get pattern of stored RbfKnowledge instance
    # @retval pattern RbfKnowledgem pattern if neuron has knowledge, None in any other case
    def get_pattern(self):
        if (self.has_knowledge()):
            return self._knowledge.get_pattern()
        else:
            return None

    ## Return True if last call to recognize() was a hit and False in any other case
    # @retval hit Boolean
    def is_hit(self):
        return self._hit

    ## Learns a new piece of knowledge
    # @retval learned Boolean. True if successfully learned,False in any other case
    def learn(self, knowledge):
        self._knowledge = knowledge
        # Indicate that this neuron has knowledge
        self._has_knowledge = True
        # Return True to indicate proper learning process
        return True

    ## Recognize a piece of knowledge
    # @retval recognized Boolean. True if successfully  recognized, False in any other case
    def recognize(self, pattern):
        # If neuron degraded, do not recognize
        if self._degraded:
            return False

        # If Manhattan distance to pattern is less than neuron radius,
        # there is a hit
        self._distance = self._knowledge.calc_manhattan_distance(pattern)
        if self._distance < self.get_radius():
            self._hit = True
        else:
            self._hit = False
        # Return whether there has been a hit or not
        return self._hit

    ## Get distance to last instance or RbfKnowledge pattern that tried to be recognized
    # @retval distance integer
    def get_distance(self):
        return self._distance

    ## Reduce radius by las recognition process's distance
    # @retval success Boolean. True if radius successfully reduced, False in any other case
    def reduce_radius_last_distance(self):
        if not self._hit:
            return False
        try:
            success = self.reduce_radius_by(self._radius - self._distance)
        except:
            return False
        return success

    ## Reduce neuron radius by certain amount
    # @param value Integer
    # retval success True if neuron has not been degraded after radius reduction and False in any other case
    def reduce_radius_by(self, value):
        # type: (value) -> integer
        if value < 0:
            raise ValueError("value must be positive")
        if self._radius < value:
            raise ValueError("value must be less than radius of neuron")
        self._radius -= value
        # If radius of neuron is under minimum allowed value,
        # the neuron has been degraded and is no longer functional
        if self._radius < RbfNeuron.MIN_RADIUS:
            self._degraded = True
        # Return true if neuron has not been degraded after radius reduction and false
        # in any other case
        return not self._degraded

    ## Increase neuron radius by certain amount
    # @param value Integer
    # retval success True if neuron has not been degraded after radius reduction and False in any other case
    def increase_radius_by(self, value):
        if value < 0:
            raise ValueError("value must be positive")
        self._radius += value
        # If radius of neuron is over maximum allowed value,
        # the neuron has been degraded and is no longer functional
        if self._radius > RbfNeuron.MAX_RADIUS:
            self._degraded = True
        return not self._degraded

    ## Return whether neuron is degraded
    # @retval degraded Boolean. True if neuron is degraded.
    def is_degraded(self):
        # Returns whether neuron is degraded
        return self._degraded

## RBF Neural Network
class RbfNetwork:

    ## Size of data or knowledge in bytes
    PATTERN_SIZE = 4.0
    ## Default radius
    DEFAULT_RADIUS = 5

    ## Class constructor, takes 'neuron_count' as parameter
    #   for setting network size
    def __init__(self, neuron_count):
        # Set data size of neuron to be created
        RbfNeuron.PATTERN_SIZE = RbfNetwork.PATTERN_SIZE
        # Set default radius of neurons
        RbfNeuron.DEFAULT_RADIUS = RbfNetwork.DEFAULT_RADIUS
        # Create neuron list
        self.neuron_list = []
        # Create list of neurons' indexes that recognized knowledge
        self._index_recognize = []
        # Set network state as MISS
        self._state = "MISS"
        # Fill neuron list with nre RbfNeuron instances
        for index in range(neuron_count):
            self.neuron_list.append(RbfNeuron())
        # Set index of neuron ready to learn as 0
        self._index_ready_to_learn = 0
        # Id of neuron that learned last given knowledge
        self._last_learned_id = -1

    ## get number of neurons in network
    # @retval count Integer. Number of neurons in network
    def get_neuron_count(self):
        return len(self.neuron_list)

    ## Recognize a given pattern
    # @param pattern RbfKnowledge pattern to be recognized
    # @retval result 'HIT' if the given pattern is recognized, 'MISS' if the network does not recognize the pattern and
    #    'DIFF' if the network identifies the pattern as pertaining to
    #    different classes
    def recognize(self, pattern):
        # Erase indexes of neurons that recognized in previous recognition processes
        self._index_recognize = []
        for index in range(self._index_ready_to_learn):
            if self.neuron_list[index].recognize(pattern):
                # Store all knowledge recognized
                self._index_recognize.append(index)

        # If no knowledge recognized
        if len(self._index_recognize) == 0:
            self._state = "MISS"
            return self._state

        # Check if all neurons recognize pattern as related to the same
        # class and set
        recognized_class = self.neuron_list[self._index_recognize[0]].get_class()
        recognized_set = self.neuron_list[self._index_recognize[0]].get_set()
        for index in self._index_recognize:
            neuron = self.neuron_list[index]
            if neuron.get_class() != recognized_class or neuron.get_set() != recognized_set:
                self._state = "DIFF"
                return self._state
        self._state = "HIT"
        return self._state

    ## Get RbfKnowledge related to last recognized pattern.
    # @retval knowledge RbfKnowledge if "HIT" in last recognition, None in any other case
    def get_knowledge(self):
        if self._state == "HIT":
            return self.neuron_list[self._index_recognize[0]].get_knowledge()
        else:
            return None

    ## Get network state
    # @retval state 'HIT' if the given pattern is recognized, 'MISS' if the network does not recognize the pattern and
    #    'DIFF' if the network identifies the pattern as pertaining to different classes
    def get_state(self):
        return self._state

    ## Learn an instance of RbfKnowledge
    #  @param knowledge RbfKnowledge to be learned
    # @retval Boolean. True if successfully learned, False in any other case.
    def learn(self, knowledge):
        # Learn procedure when pattern has not been recognized
        self.recognize(knowledge.get_pattern())
        # If the pattern has not been recognized by any neuron in the net
        if self._state == 'MISS':
            # Learn in ready-to-learn neuron
            return self._learn_ready_to_learn(knowledge)
        # If various neurons have recognized the pattern as pertaining to different classes
        elif self._state == 'DIFF':
            # Get correct class
            correct_class = knowledge.get_class()
            # Correct class identified by at least one neuron flag
            correct_flag = False
            # Reduce radius of all neurons that do not recognize the pattern
            for index in self._index_recognize:
                neuron = self.neuron_list[index]
                if neuron.get_class() != correct_class:
                    neuron.reduce_radius_last_distance()
                else:
                    self._last_learned_id = index
                    correct_flag = True
            if correct_flag:
                return True
            # If correct class was not identified by any recognizing neuron, learn
            return self._learn_ready_to_learn(knowledge)

        # If at least one neuron recognizes the pattern as pertaining to a unique class
        else:
            correct_class = knowledge.get_class()
            # If the class to be learned is different from the class identified
            if correct_class != self.neuron_list[self._index_recognize[0]].get_class():
                # Min distance from recognizing neurons to class
                min_distance = RbfNeuron.DEFAULT_RADIUS
                # Reduce radius to all recognizing neurons
                for index in self._index_recognize:
                    neuron = self.neuron_list[index]
                    neuron.reduce_radius_last_distance()
                    # Get distance
                    neuron_distance = neuron.get_distance()
                    # If distance is less than current minimum distance, store
                    if neuron_distance < min_distance:
                        min_distance = neuron_distance
                # Learn new knowledge with radius = min_distance
                self._learn_ready_to_learn(knowledge, min_distance)
            return True

    def _learn_ready_to_learn(self, knowledge, radius=RbfNeuron.DEFAULT_RADIUS):
        # Learn new pattern in ready-to-learn neuron
        # Select ready-to-learn neuron
        ready_to_learn_neuron = self.neuron_list[self._index_ready_to_learn]
        # Learn and store result (True or False) in auxiliary variable 'ret_val'
        learned = ready_to_learn_neuron.learn(knowledge)
        # Set radius
        ready_to_learn_neuron.set_radius(radius)
        # Increment ready-to-learn neuron index
        if learned:
            self._last_learned_id = self._index_ready_to_learn
            self._index_ready_to_learn += 1
        # Return whether net succesfully learned the given pattern
        return learned

    ## Get ids of recognizing set neurons
    # @retval ids Integers list
    def get_rneurons_ids(self):
        return self._index_recognize

    ## Get id of neuron affected in the last learning process
    # @retval id Integer
    def get_last_learned_id(self):
        return self._last_learned_id

    ## Get index of ready-to-learn neuron
    # @retval index Integer
    def get_index_ready_to_learn(self):
        return self._index_ready_to_learn

    @classmethod
    ## Serialize object and store in given file
    # @param cls RbfNetwork class
    # @param obj RbfNetwork object to be serialized
    # @param name Name of the file where the serialization is to be stored
    def serialize(cls, obj, name):
        pickle.dump(obj, open(name, "wb"))

    @classmethod
    ## Deserialize object stored in given file
    # @param cls RbfNetwork class
    # @param name Name of the file where the object is serialized
    def deserialize(cls, name):
        return pickle.load(open(name, "rb"))


## Sensory Neural Block
# Stores sight and hearing RbfNetworks
class SensoryNeuralBlock:

    ## Number of neurons in sight network
    SIGHT_NEURON_COUNT = 100
    ## Number of neurons in hearing network
    HEARING_NEURON_COUNT = 100

    ## The constructor
    def __init__(self, sight_snb_file="NoFile", hearing_snb_file="NoFile"):
        # Create sight neural blocks
        if sight_snb_file != "NoFile":
            ## @var snb_s
            # Sight sensory neural block
            self.snb_s = RbfNetwork.deserialize(sight_snb_file)
        else:
            self.snb_s = RbfNetwork(SensoryNeuralBlock.SIGHT_NEURON_COUNT)
        # Create hearing neural blocks
        if hearing_snb_file != "NoFile":
            ## @var snb_h
            # Hearing sensory neural block
            self.snb_h = RbfNetwork.deserialize(hearing_snb_file)
        else:
            self.snb_h = RbfNetwork(SensoryNeuralBlock.SIGHT_NEURON_COUNT)
        self._last_learned_ids = None

    ## Recognize a sight pattern
    # @param pattern RBF sight pattern
    # @retval success True if pattern successfully recognized, False in any other case
    def recognize_sight(self, pattern ):
        return self.snb_s.recognize(pattern)

    ## Recognize a hearing pattern
    # @param pattern RBF hearing pattern
    # @retval success True if pattern successfully recognized, False in any other case
    def recognize_hearing(self, pattern ):
        return self.snb_h.recognize(pattern)

    ## Learn a hearing pattern
    # @param pattern RBF hearing pattern
    # @retval success True if pattern successfully learned, False in any other case
    def learn_hearing(self, knowledge ):
        return self.snb_h.learn(knowledge)

    ## Learn a visual pattern
    # @param pattern RBF sight pattern
    # @retval success True if pattern successfully recognized, False in any other case
    def learn_sight(self, knowledge ):
        return self.snb_s.learn(knowledge)

    ## Learn a pair of hearing and sight patterns relating both pieces of knowledge through the hearing id stored as the
    #  sight knowledge's pattern
    # @param knowledge_h RBF hearing knowledge
    # @param pattern_s RBF sight pattern
    # @retval success True if patterns successfully learned, False in any other case
    def learn(self, knowledge_h, pattern_s ):
        # If hearing knowledge learned
        if self.snb_h.learn(knowledge_h):
            # Get index of hearing neuron that has learned
            index_hearing = self.snb_h.get_last_learned_id()
            # Relate sight pattern and index of hearing neuron in just one piece of RbfKnowledge
            knowledge_s = RbfKnowledge(pattern_s, str(index_hearing) )
            # Learn, get learn status (True, False)
            learned = self.snb_s.learn(knowledge_s)
            # Get index of sight neuron that has learned
            index_sight = self.snb_s.get_last_learned_id()
            if learned:
                # Store indexes of hearing and sight neurons that just learned
                self._last_learned_ids = (index_hearing, index_sight )
            # Return sight learning state
            return learned
        # Could not learn hearing knowledge
        return False

    ## Return a 2-tuple of integeres representing the ids of hearing and sight neurons that learned in the last
    #  *learn_sight* process
    def get_last_learned_ids(self):
        return self._last_learned_ids

    ##  Return hearing knowledge related to given pattern or neuron id,
    #    if pattern or neuron_id in hearing network, and None in any other case
    def get_hearing_knowledge(self, pattern_or_id, is_id=False):
        # If given parameter pattern_or_id is id
        if is_id:
            neuron_id = pattern_or_id
            # If there is a neuron with corresponding id
            if neuron_id <= self.snb_h.get_last_learned_id():
                # If the neuron is not degraded
                if not self.snb_h.neuron_list[neuron_id].is_degraded():
                    # Return pattern
                    return self.snb_h.neuron_list[neuron_id].get_knowledge()
            return None
        else:
            pattern = pattern_or_id
            if self.snb_h.recognize(pattern) == "HIT":
                return self.snb_h.get_knowledge()
            return None
    ## Return hearing knowledge related to given pattern or neuron id,
    #   if pattern or neuron_id in sight network, and None in any other case
    def get_sight_knowledge(self, pattern_or_id, is_id=False):
        # If given parameter pattern_or_id is id
        if is_id:
            neuron_id = pattern_or_id
            # If there is a neuron with corresponding id
            if neuron_id < self.snb_s.get_index_ready_to_learn():
                # If the neuron is not degraded
                if not self.snb_s.neuron_list[neuron_id].is_degraded():
                    # Return pattern
                    return self.snb_s.neuron_list[neuron_id].get_knowledge()
            return None
        else:
            pattern = pattern_or_id
            if self.snb_s.recognize(pattern) == "HIT":
                return self.snb_s.get_knowledge()
            return None

    ## Save snb object in given files (one for the sight sensory neural block and the other for the
    # hearing neural block
    # @param sight_snb_file Filename where the sight sensory neural block is to be saved
    # @param  hearing_snb_file Filename where the hearing sensory neural block is to be saved
    def save(self, sight_snb_file, hearing_snb_file):
        RbfNetwork.serialize(self.snb_s, sight_snb_file)
        RbfNetwork.serialize(self.snb_h, hearing_snb_file)


## @}
#