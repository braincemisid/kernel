import pickle

from neuron import Neuron

## \defgroup GeometricNeuralBlock Geometric Neural Block classes
#
#   Geometric Neural Block classes are a group of classes that
# neurally represent concepts in the brain. For instance,
# the QuantityOrderGroup models a relation between the concepts of
# quantity and order.
# @{
#


## The QuantityNeuron class is a kind of neuron
# that signals the cardinality of its relation in a QuantityOrderGroup
class QuantityNeuron(Neuron):
    pass

## The QuantityNeuron class is a kind of neuron
# whose position in a QuantityOrderNetwork
# signals certain ordinality.
class OrderNeuron(Neuron):
    pass

## A QuantityOrderGroup is a pair composed of an OrderNeuron and a QuantityNeuron.
# It is the basic element of a QuantityOrderNetwork.
class QuantityOrderGroup:

    ## The constructor
    def __init__(self):
        ## @var _has_quantity
        # Boolean. True if the group has a quantity stored, False in any other case.
        self._has_quantity= False
        ## @var _quantity_neuron
        # QuantityNeuron of the group
        self._quantity_neuron = None
        ## @var _order_neuron
        # OrderNeuron of the group
        self._order_neuron = OrderNeuron()

    ## Store a quantity
    def clack(self, knowledge):
        self._quantity_neuron = QuantityNeuron(knowledge)
        self._has_quantity = True

    ## Retrieve a quantity
    def get_quantity(self):
        return self._quantity_neuron

    ## Return true if the group has a quantity stored
    # and false in any other case
    def has_quantity(self):
        return self._has_quantity

    ## Compare certain piece of knowledge with the quantity
    # stored in the group
    # @param knowledge The knowledge to be compared
    def compare(self, knowledge):
        if self._has_quantity:
            return knowledge == self._quantity_neuron.get_knowledge()
        return False

## A set of QuantityOrderGroup instances that act together
# in order to store Order and Quantity information
class QuantityOrderNetwork:

    ## The constructor
    def __init__(self):
        ## @var group_list
        # Set of QuantityOrderGroup instances
        self.group_list = []
        self._index = 0

    ## Start count
    def bum(self):
        self._index = 0

    ## Point to next QuantityOrderGroup
    def bip(self):
        if len(self.group_list) <= self._index:
            self.group_list.append(QuantityOrderGroup())
        self._index += 1

    ## Store quantity in currently pointed QuantityOrderGroup
    def clack(self, knowledge=None):
        if not self.group_list[self._index-1].has_quantity():
            if knowledge is None:
                raise ValueError("GeometricNeuralBlock: a kind of order knowledge must be passed")
            self.group_list[self._index-1].clack(knowledge)
        return self.group_list[self._index-1].get_quantity()

    ## Get position of the currently pointed QuantityOrderGroup as a number of *bips*
    # since the beginning (*bum*).
    def get_bip_count(self, knowledge):
        for index in range(len(self.group_list)):
            if self.group_list[index].compare(knowledge):
                return index+1
        return None

## The addition structure class provides a decimal numeric system neural representation
#
# This structure has a set of ten neurons and a carry-over neuron. It also has an structure index
# which is sequentially incremented over the ten neurons in order to make an addition.
class AdditionStructure:

    ## The constructor.
    def __init__(self):
        ## @var neurons
        # List of neurons in the structure
        self.neurons = [Neuron() for index in range(10)]
        ## @var carry_over
        # Boolean variable. True signals a carry over.
        self.carry_over = False
        ## @var index
        # Structure index. Points to one of the ten neurons in the structure.
        self.index = 0

    ## Start an addition operation
    # and set the carry over variable to zero
    def bum(self):
        if self.carry_over:
            self.index = 1
            self.carry_over = False
        else:
            self.index = 0

    ## Point to next neuron in structure
    def bip(self):
        self.index += 1
        if self.index >= len(self.neurons):
            self.index = 0
            self.carry_over = True

    ## Return results
    def clack(self):
        return self.index

    ## Return True if the structure has a carry over
    # and False in any other case
    def has_carry(self):
        return self.carry_over

    ## Clear carry over
    def clear_carry(self):
        self.carry_over = False

## Class that envelopes all neural geometries that
# represent concepts in the brain
class GeometricNeuralBlock:

    ## The constructor
    def __init__(self):
        # Quantity-Order and Addition Structures
        self._order_structure = QuantityOrderNetwork()
        self._addition_structure = AdditionStructure()
        # Default operation
        self._operation = "COUNT"
        # Queues for operands
        self._op1_queue = []
        self._op2_queue = []
        # Operator
        self._operator = None
        # Sum operator
        self._add_operator = None
        # Equal sign
        self._equal_sign = None
        # Zero
        self._zero = None

    ## Set an operation to be executed by the block
    # @param operation The operation to be executed. Can take on the values "COUNT" or "ADD"
    def set_operation(self, operation):
        if operation == "COUNT":
            self._operation = "COUNT"
        elif operation == "ADD":
            self._operation = "ADD"
        else:
            raise ValueError("invalid operation")

    ## Get the type of operation to be executed by the block
    # @retval operation "COUNT" or "ADD"
    def get_operation(self):
        return self._operation

    ## Set the knowledge that represents an addition operator.
    # In the Brain-CEMISID project this knowledge is just the
    # hearing neuron id that stores the the corresponding pattern
    def set_add_operator(self, knowledge):
        self._add_operator = knowledge

    ## Return the knowledge that represents the addition operator
    # @retval operator Addition operator
    def get_add_operator(self):
        return self._add_operator

    ## Set the knowledge that represents an equal sign.
    # In the Brain-CEMISID project this knowledge is just the
    # hearing neuron id that stores the the corresponding pattern
    def set_equal_sign(self, knowledge):
        self._equal_sign = knowledge

    ## Return the knowledge that represents the equal sign
    # @retval knowledge Equal sign
    def get_equal_sign(self):
        return self._equal_sign

    ## Set the knowledge that represents a zero.
    # In the Brain-CEMISID project this knowledge is just the
    # hearing neuron id that stores the corresponding pattern
    def set_zero(self, knowledge):
        self._zero = knowledge

    ## Start operation
    def bum(self):
        if self._operation == "COUNT":
            self._bum_count()
        elif self._operation == "ADD":
            self._bum_add()
        # Add here future operations
        else:
            return

    ## Either count or pass addition operands and operator
    # @param knowledge Optional parameter that stores the information of the operands and operator for and addition operation. In
    # the BrainCEMISID project it is just the hearing neuron id of the corresponding pattern.
    def bip(self, knowledge=None):
        if self._operation == "COUNT":
            self._bip_count()
        elif self._operation == "ADD":
            self._bip_add(knowledge)
        # Add here future operations
        else:
            return

    ## Either finish counting or adding
    def clack(self, knowledge=None):
        if self._operation == "COUNT":
            self._clack_count(knowledge)
        elif self._operation == "ADD":
            self._bip_add(knowledge)
            # Add here future operations
        else:
            return

    ## Get addition result
    # @retval addition_result a list of the hearing neuron ids corresponding to the digits of the result
    def get_addition_result(self):
        return self.addition_result

    def _bum_count(self):
        self._order_structure.bum()

    def _bip_count(self):
        self._order_structure.bip()

    def _clack_count(self, knowledge):
        self._order_structure.clack(knowledge)

    def _bum_add(self):
        self._addition_structure.bum()
        self._op1_queue = []
        self._op2_queue = []
        self._operator = None
        self.addition_result = []

    def _bip_add(self, knowledge):

        # if knowledge correspond to some operator, store in _operator attribute
        if knowledge == self._add_operator:
            self._operator = knowledge
        elif knowledge == self._equal_sign:
            self._check_add(knowledge)
        # Else if the oprator has not been introduced yet, knowledge is part of first operand
        elif self._operator is None:
            self._op1_queue.append(knowledge)
        # Else it is part of second operand
        else:
            self._op2_queue.append(knowledge)

    def _check_add(self, knowledge):
        # If the given knowledge corresponds to what the brain understands to be an equal sign
        if knowledge == self._equal_sign:
            # If the given operator corresponds to what the brain knows that is an addition operator, add
            if self._operator == self._add_operator:
                return self._add()
        else:
            return False

    def _add(self):
        addition_result = []
        # While there is a digit to be added
        while len(self._op1_queue) != 0 or len(self._op2_queue) != 0:
            # Get first operand
            if len(self._op1_queue) != 0:
                digit_op_1 = self._op1_queue.pop()
            else:
                digit_op_1 = self._zero
            # Get second operand
            if len(self._op2_queue) != 0:
                digit_op_2 = self._op2_queue.pop()
            else:
                digit_op_2 = self._zero

            # Get bip count of operands
            bip_count_1 = self._get_bip_count(digit_op_1)
            bip_count_2 = self._get_bip_count(digit_op_2)

            # Validate
            if bip_count_1 is None or bip_count_2 is None:
                raise ValueError("An operand cannot be recognized")

            # Add using addition_structure
            self._addition_structure.bum()
            for index in range(bip_count_1):
                self._addition_structure.bip()
            for index in range(bip_count_2):
                self._addition_structure.bip()
            addition_result.append(self._addition_structure.clack())

        if self._addition_structure.has_carry():
            self._addition_structure.bum()
            addition_result.append(self._addition_structure.clack())

        self.addition_result = []
        for digit in addition_result:
            # The zero is a special concept to be addressed possibly in future versions
            if digit == 0:
                self.addition_result.append(self._zero)
            # For the rest of digits, use the Quantity-order structure
            else:
                self._order_structure.bum()
                for index in range(digit):
                    self._order_structure.bip()
                digit_representation = self._order_structure.clack().get_knowledge()
                self.addition_result.append(digit_representation)
        self.addition_result.reverse()

    def _get_bip_count(self, digit):
        if digit == self._zero:
            return 0
        return self._order_structure.get_bip_count(digit)

    def _get_operand_1(self):
        if len(self._op1_queue) != 0:
            return self._op1_queue.pop()
        return self._zero

    def _get_operand_2(self):
        if len(self._op2_queue) != 0:
            return self._op2_queue.pop()
        return self._zero



    @classmethod
    ## Serialize object and store in given file
    # @param cls GeometricNeuralBlock class
    # @param obj GeometricNeuralBlock object to be serialized
    # @param name Name of the file where the serialization is to be stored
    def serialize(cls, obj, name):
        pickle.dump(obj, open(name, "wb"))

    @classmethod
    ## Deserialize object stored in given file
    # @param cls GeometricNeuralBlock class
    # @param name Name of the file where the object is serialized
    def deserialize(cls, name):
        return pickle.load(open(name, "rb"))

## @}
#

# Tests
if __name__ == '__main__':

    net = QuantityOrderNetwork()

    net.bum()
    net.bip()
    net.clack(1)

    net.bum()
    net.bip()
    net.bip()
    net.bip()
    net.bip()
    net.clack(4)

    net.bum()
    net.bip()
    quantity_1 = net.clack().get_knowledge()
    net.bum()
    net.bip()
    net.bip()
    net.bip()
    net.bip()
    quantity_2 = net.clack().get_knowledge()

    print "Quantity 1 is: ", quantity_1
    print "Quantity 2 is: ", quantity_2

    # AdditionStructure

    print "Adition Struture:  "
    add_s = AdditionStructure()
    add_s.bum()
    for i in range(15):
        add_s.bip()
    if add_s.has_carry():
        print "1", add_s.clack()
    else:
        print add_s.clack()

    # Geometric Neural Block
    gnb = GeometricNeuralBlock()

    # Start count
    gnb.bum()
    gnb.bip()
    gnb.clack('1')

    gnb.bum()
    gnb.bip()
    gnb.bip()
    gnb.clack('2')

    gnb.bum()
    gnb.bip()
    gnb.bip()
    gnb.bip()
    gnb.clack('3')

    gnb.bum()
    gnb.bip()
    gnb.bip()
    gnb.bip()
    gnb.bip()
    gnb.clack('4')

    gnb.bum()
    gnb.bip()
    gnb.bip()
    gnb.bip()
    gnb.bip()
    gnb.bip()
    gnb.clack('5')

    gnb.bum()
    gnb.bip()
    gnb.bip()
    gnb.bip()
    gnb.bip()
    gnb.bip()
    gnb.bip()
    gnb.clack('6')

    gnb.bum()
    gnb.bip()
    gnb.bip()
    gnb.bip()
    gnb.bip()
    gnb.bip()
    gnb.bip()
    gnb.bip()
    gnb.clack('7')

    gnb.bum()
    gnb.bip()
    gnb.bip()
    gnb.bip()
    gnb.bip()
    gnb.bip()
    gnb.bip()
    gnb.bip()
    gnb.bip()
    gnb.clack('7')

    gnb.bum()
    gnb.bip()
    gnb.bip()
    gnb.bip()
    gnb.bip()
    gnb.bip()
    gnb.bip()
    gnb.bip()
    gnb.bip()
    gnb.bip()
    gnb.clack('9')

    gnb.set_add_operator('+')
    gnb.set_equal_sign('=')
    gnb.set_zero('0')

    gnb.set_operation("ADD")

    gnb.bum()
    gnb.bip('2')
    gnb.bip('+')
    gnb.bip('1')
    gnb.bip('=')

    print gnb.get_addition_result()

    gnb.bum()
    gnb.bip('5')
    gnb.bip('2')
    gnb.bip('6')
    gnb.bip('+')
    gnb.bip('5')
    gnb.bip('7')
    gnb.bip('4')
    gnb.bip('=')
    print gnb.get_addition_result()
