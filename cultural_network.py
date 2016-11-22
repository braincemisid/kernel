import pickle

from neuron import Neuron

## \defgroup CultBlocks Cultural network related classes
#
# Relational network related classes are a group of classes that
# represent cultural neurons, groups and networks
# @{
#

## Cultural neuron
class CulturalNeuron(Neuron):
    pass

## Cultural group
class CulturalGroup:

    ## CulturalGroup class constructor
    def __init__(self):
        ## @var group Set of CulturalNeuron instances
        self.group = []
        self._index_bip = 0

    ## Learn new piece of cultural knowledge as part of the cultural group
    #  @param knowledge Object of type CulturalKnowledge. Knowledge to be learned
    def learn(self, knowledge):
        self.group.append(CulturalNeuron(knowledge))

    ## Initialize bbcc protocol
    def bum(self):

        # Reinitialize bip index
        self._index_bip = 0

    ## Return True if given knowledge equals the one store in current neuron. Increase index-bip
    # so that next comparison is made in the following neuron of the group. If there are no more neurons in
    # the group, return False.
    # @param knowledge Piece of knowledge to be compared
    # @retval result Boolean
    def bip(self, knowledge):
        # If there are still neurons in the group, make the comparison
        if self._index_bip < len(self.group):
            knowledge_eq = self.group[self._index_bip].get_knowledge() == knowledge
            self._index_bip += 1
            return knowledge_eq
        # If there are no more neurons in the group, reinitialize bip index and return False
        self._index_bip = 0
        return False

    ## Return true if given knowledge equals the one store in current neuron and there is exactly one more neuron
    #   in the group with the final knowledge related to de bbcc sequence. Return false in any other case.
    #    @param knowledge: Piece of knowledge to be compared
    #    @retval result Boolean
    def check(self, knowledge):
        return self.bip(knowledge) and (len(self.group)-1) == self._index_bip

    ## Learn new piece of cultural knowledge as part of the cultural group
    # @param knowledge Object of type CulturalKnowledge. Knowledge to be learned
    def clack(self, knowledge):
        self.learn(knowledge)

    ## Get last knowledge in group
    # @retval knowledge Last knowledge in group
    def get_tail_knowledge(self):
        return self.group[len(self.group)-1].get_knowledge()

    ## Return True if knowledge is contained by some neuron in the group and
    # False in any other case
    # @param knowledge
    # @retval result Boolean
    def contains(self, knowledge):
        for neuron in self.group:
            if neuron.get_knowledge() == knowledge:
                return True
        return False

    ## Erase all knowledge in group
    def reinit(self):
        self.group = []
        self._index_bip = 0


## Cultural network
#
# Set of CulturalGroup instances
class CulturalNetwork:

    ## CulturalNetwork class constructor
    def __init__(self, group_count=1):
        self.group_list = []
        for index in range(group_count):
            self.group_list.append(CulturalGroup())
        self._index_ready_to_learn = 0
        self._clack = False
        self._recognized_indexes = []


    ## Start of bbcc protocol
    def bum(self):
        # Resize network if needed
        if self._index_ready_to_learn >= len(self.group_list):
            self.resize()
        # Renintialize ready lo learn group
        self.group_list[self._index_ready_to_learn].reinit()
        # Reinitialize vector of recognized indexes
        self._recognized_indexes = []
        # Pass bum signal to all cultural groups with knowledge
        for group_index in range(self._index_ready_to_learn):
            self.group_list[group_index].bum()
            # Initialize a list of participating neurons
            # At this stage, all neurons with knowledge may recognize the sequence
            self._recognized_indexes.append(group_index)

    ## Pass an instance of knowledge to be compared or stored
    # @param knowledge
    def bip(self, knowledge):
        # Indexes of neural groups that recognized bip
        bip_indexes = []
        # Pass bip signal to all neurons that have recognized the given sequence
        # and store the indexes of all neurons that have recognized until now
        for group_index in self._recognized_indexes:
            if self.group_list[group_index].bip(knowledge):
                bip_indexes.append(group_index)
        # Store bip_indexes in instance attribute _recognized_indexes
        self._recognized_indexes = bip_indexes
        # Learn in ready to learn neuron
        self.group_list[self._index_ready_to_learn].learn(knowledge)

    ## Pass the second-to-last instance of knowledge to be compared or stored
    # @param knowledge
    def check(self, knowledge):
        # Indexes
        check_indexes = []
        for group_index in self._recognized_indexes:
            if self.group_list[group_index].check(knowledge):
                check_indexes.append(group_index)
        self._recognized_indexes = check_indexes
        # If no group has the knowledge related to the given sequence, keep learning
        if len(check_indexes) == 0:
            # Learn
            self.group_list[self._index_ready_to_learn].learn(knowledge)
            # Enable clack
            self._clack = True
            return None
        # Exactly one cultural group must have recognized the sequence, return index of that group
        elif len(check_indexes) == 1:
            # Do not learn
            self.group_list[self._index_ready_to_learn].reinit()
            # Return index of cultural group that has recognized the process
            return check_indexes[0]
        else:
            raise AttributeError("CulturalNet net has an inconsistent state")

    ## earn tail knowledge of cultural group
    # @param knowledge Tail knowledge
    def clack(self, knowledge):
        if not self._clack:
            return
        # Learn
        self.group_list[self._index_ready_to_learn].clack(knowledge)
        self._clack = False
        self._index_ready_to_learn += 1

    ## Resize network
    def resize(self):
        new_list = []
        # Fill neuron list with memories
        for index in range(len(self.group_list)):
            new_list.append(CulturalGroup())
        self.group_list = self.group_list + new_list

    ## Get tail knowledge of a given group id
    # @param group_id
    def get_tail_knowledge(self, group_id):
        return self.group_list[group_id].get_tail_knowledge()

    ## Get id of group that learned last sequence
    def get_last_clack_id(self):
        return self._index_ready_to_learn - 1

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
        return pickle.load(open(name, "rb"))

## @}
#

# Tests
if __name__ == '__main__':

    net = CulturalNetwork(5)

    net.bum()
    net.check("a")
    net.clack("a")

    net.bum()
    net.bip("b")
    net.check("a")
    net.clack("ba")

    net.bum()
    net.bip("l")
    net.bip("l")
    net.check("o")
    net.clack("llo")

    i = 1


