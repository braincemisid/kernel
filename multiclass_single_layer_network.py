import pickle


## \addtogroup Intentions MulticlassSingleLayerNetwork
# 
#  Single layer feedforward network with multiple outputs for input classification
# @{
class MulticlassSingleLayerNetwork():

    ## Multiclass PerceptronNetwork constructor
    # @param inputs_number Integer. Number of inputs in perceptron network
    # @param outputs_number Integer. Number of outputs in perceptron network
    def __init__(self, inputs_number, outputs_number):
        # Number of inputs in perceptron net must be a positive integer
        if type(inputs_number) is not int or inputs_number <= 0:
            raise TypeError('MulticlassSingleLayerNetwork constructor expects inputs_number to be a positive integer')
        # Number of outputs in perceptron net must be a positive integer
        if type(outputs_number) is not int or inputs_number <= 0:
            raise TypeError('MulticlassSingleLayerNetwork constructor expects outputs_number to be a positive integer')
        # Weight vector for outputs
        self.weights = []
        for index in range(outputs_number):
            self.weights.append( [0 for j in range(inputs_number)] )
        # Inputs
        self.inputs = [0 for index in range(inputs_number)]
        # Output
        self.outputs = [0 for index in range(inputs_number)]
        # Activation function
        self._activation_function = lambda x: x
        # Learning rate, 0.1 by default
        self.learning_rate = 0.1

    ## Get net inputs
    # @retval inputs Floats vector
    def get_inputs(self):
        return self.inputs

    ## Set net inputs
    # @retval inputs Floats vector
    def set_inputs(self, new_inputs):
        if len(new_inputs) != len(self.inputs):
            return False
        self.inputs = new_inputs
        return True

    ## Calculate and get outputs
    # @retval outputs Floats vector
    def get_outputs(self):
        self.outputs = self._calc_output()
        return self.outputs

    ## Set net learning rate according to the Widrow-Hoff equation.
    # @learning_rate Float.
    def set_learning_rate(self, learning_rate):
        self.learning_rate = learning_rate

    ## Get learning rate
    # @retval rate Float
    def get_learning_rate(self):
        return learning_rate

    ## Set activation function
    # @param new_func Function name. New activation function
    def set_activation_function(self, new_func ):
        self._activation_function = new_func

    ## Train network
    # @param training_set Network training set. See tests in code file for an example.
    def training(self, training_set):

        for input_vector, desired_output in training_set:
            current_output = self._calc_output(input_vector)
            error = [do_j-co_j for do_j, co_j in zip(desired_output, current_output)]
            self.update_weights(input_vector, error)

    def _calc_output(self, input_vector=None):
        if input_vector is None:
            input_vector = self.inputs
        dot_product = []
        ret_val = []
        for index in range(len(self.weights)):
            dot_product.append(sum(input_j * weight_j for input_j, weight_j in zip(input_vector, self.weights[index])))
            ret_val.append(self._activation_function(dot_product[index]))
        return ret_val

    ## Update weights in order to match desired output by applying the Widrow-Hoff equation
    # @param input_vector Input vector
    # @param error Prediction error
    def update_weights(self, input_vector, error):
        # Update weights in order to match desired output
        for row in range(len(self.weights)):
            for index in range(len(self.weights[0])):
                self.weights[row][index] = self.weights[row][index] + self.learning_rate*error[row]*input_vector[index]


##@}
#


# Tests
if __name__ == '__main__':
    net = MulticlassSingleLayerNetwork(3,2)

    training_set = [([0.9,0.5,0.4],[1,0]), ([0.5,0.7,0.4],[0,1]), ([0.1,0.3,0.8],[0,0]), ([0.7,0.3,0.1],[1,0]), ([0.35,0.8,0.2],[0,1]), ([0.20,0.35,0.83],[0,0]), ([0.97,0.52,0.43],[1,0]), ([0.4,0.9,0.5],[0,1]), ([0.6,0.3,0.8],[0,0])]

    print "Before training"
    net.inputs =[1,1,1]
    print "Inputs = [1,1,1], Outputs = ", net.get_outputs()


    def f(x):
        if x > 0:
            return 1
        return 0

    net.set_activation_function( f )

    net.training(training_set)
    print "After training"
    print "Weights1: ", net.weights[0], "Weights2", net.weights[1]
    net.inputs = [0.7, 0.3, 0.2]
    print "Inputs = ", net.inputs, "Outputs = ", net.get_outputs()
    net.inputs = [0.1, 0.8, 0.2]
    print "Inputs = ", net.inputs, "Outputs = ", net.get_outputs()
    net.inputs = [0.3, 0.1, 0.7]
    print "Inputs = ", net.inputs, "Outputs = ", net.get_outputs()
    net.inputs = [0.3,0.1,0.4]
    print "Inputs = ", net.inputs, "Outputs = ", net.get_outputs()


    # Just another example
    net2 = MulticlassSingleLayerNetwork(3,3)

    training_set = [([0.9, 0.5, 0.4],[0.9, 0.5, 0.4] ), ([0.5, 0.7, 0.4], [0.5, 0.7, 0.4]), ([0.1, 0.3, 0.8], [0.1, 0.3, 0.8]),
                    ([0.7, 0.3, 0.1], [0.7, 0.3, 0.1]), ([0.35, 0.8, 0.2], [0.35, 0.8, 0.2]), ([0.20, 0.35, 0.83],[0.20, 0.35, 0.83])]

    net2.training(training_set)
    net2.training(training_set)
    print('-'*60)
    print "Net 2"
    print "After training"
    print "Weights: ", net2.weights
    net2.inputs = [0.7, 0.3, 0.2]
    print "Inputs = ", net2.inputs, "Outputs = ", net2.get_outputs()
    net2.inputs = [0.1, 0.8, 0.2]
    print "Inputs = ", net2.inputs, "Outputs = ", net2.get_outputs()
    net2.inputs = [0.3, 0.1, 0.7]
    print "Inputs = ", net2.inputs, "Outputs = ", net2.get_outputs()
    net2.inputs = [1, 1, 1]
    print "Inputs = ", net2.inputs, "Outputs = ", net2.get_outputs()

