class Neuron(object):
    def __init__(self, knowledge=None):
        if knowledge is None:
            self._has_knowledge = False
            self._knowledge = None
        else:
            self.set_knowledge(knowledge)

    def set_knowledge(self, knowledge):
        self._has_knowledge = True
        self._knowledge = knowledge

    def get_knowledge(self):
        return self._knowledge

    def has_knowledge(self):
        """ Return true if the neuron has already learned some kind of knowledge
        and false in any other case """
        return self._has_knowledge
