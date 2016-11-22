import pickle

from cultural_network import CulturalNetwork,CulturalGroup,CulturalNeuron

## \addtogroup Intentions
#  Episodic memories block
# @{

## The EpisodicMemoriesBlock is a specialization of CulturalNetwork
# from which a set of CulturalGroup s can be retrieved given a set of triggers, just
# as in humans. An exact memory can also be retrieved.
class EpisodicMemoriesBlock(CulturalNetwork):

    ## The constructor
    def __init__(self):
        CulturalNetwork.__init__(self)

    ## Return a list of memories (Cultural Groups) that contain the list of given
    # memory triggers
    # @retval retrieved_memories CulturalGroup vector.
    def retrieve_memories(self, trigger_list):
        # List of retrieved memories, initialized as and empty list
        retrieved_memories = []

        # For every trigger
        for trigger in trigger_list:
            # For every group (memory) in list of memories
            for group in self.group_list:
                # If the memory contains the given trigger
                if group.contains(trigger):
                    # Append the memory to list of retrieved memories
                    retrieved_memories.append(group)

        return retrieved_memories

    ##  Return the exact memory (except for last element in trigger)
    # @retval memory CulturalGroup
    def retrieve_exact_memory(self, trigger ):
        # Use bbcc protocol
        self.bum()
        for index in range(len(trigger)):
            if index != len(trigger)-1:
                self.bip(trigger[index])
            else:
                return self.group_list[self.check(trigger[index])]


    @classmethod
    ## Serialize object and store it in given file
    # @param cls EpisodicMemory class
    # @param obj EpisodicMemory object to be serialized
    # @param name Name of the file where the serialization is to be stored
    def serialize(cls, obj, name):
        pickle.dump(obj, open(name, "wb"))

    @classmethod
    ## Deserialize object stored in given file
    # @param cls EpisodicMemory class
    # @param name Name of the file where the object is serialized
    def deserialize(cls, name):
        return pickle.load(open(name, "rb"))

## @}
#

# Tests
if __name__ == '__main__':

    em = EpisodicMemoriesBlock()

    # Learn a set of memories related to school
    # and its given [B C F]

    em.bum()
    em.bip('pencil')
    em.bip('eraser')
    em.check('sharpener')
    bcf = [0.1, 1, 0.6]
    em.clack(bcf)

    em.bum()
    em.bip('board')
    em.bip('eraser')
    em.check('pupils')
    bcf = [0.5, 0.7, 0.4]
    em.clack(bcf)

    em.bum()
    em.bip('board')
    em.bip('notebook')
    em.check('pupils')
    bcf = [0.4, 0.7, 0.4]
    em.clack(bcf)

    # Test memories retrieval
    print "Retrieving memories related to 'house' "
    if len(em.retrieve_memories(['house'])) == 0:
        print "No memories found"
    else:
        for memory in em.retrieve_memories(['house']):
            for episode in memory:
                print episode.get_knowledge()

    print "Retrieving memories related to 'eraser'"
    if len(em.retrieve_memories(['eraser'])) == 0:
        print "No memories found"
    else:
        for memory in em.retrieve_memories(['eraser']):
            for episode in memory.group:
                print episode.get_knowledge()

    print "Retrieving memories related to 'board and eraser'"
    if len(em.retrieve_memories(['board', 'eraser'])) == 0:
        print "No memories found"
    else:
        for memory in em.retrieve_memories(['board', 'eraser']):
            for episode in memory.group:
                print episode.get_knowledge()