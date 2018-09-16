import numpy as np


class NeuralNetwork(object):
    def __init__(self, input_nodes, hidden_nodes, output_nodes, learning_rate):
        # Set number of nodes in input, hidden and output layers.
        self.input_nodes = input_nodes
        self.hidden_nodes = hidden_nodes
        self.output_nodes = output_nodes

        # Initialize weights
        self.weights_input_to_hidden = np.random.normal(0.0, self.input_nodes**-0.5,
                                       (self.input_nodes, self.hidden_nodes))

        self.weights_hidden_to_output = np.random.normal(0.0, self.hidden_nodes**-0.5,
                                       (self.hidden_nodes, self.output_nodes))
        self.lr = learning_rate

        self.activation_function = lambda x : (1 / (1 + np.exp(-x)))


    def train(self, features, targets):
        ''' Train the network on batch of features and targets.
            Arguments
            ---------
            features: 2D array, each row is one data record, each column is a feature
            targets: 1D array of target values
        '''
        n_records = features.shape[0]
        delta_weights_i_h = np.zeros(self.weights_input_to_hidden.shape)
        delta_weights_h_o = np.zeros(self.weights_hidden_to_output.shape)
        for X, y in zip(features, targets):

            #### Implement the forward pass here ####
            ### Forward pass ###

            # Signals into hidden layer
            hidden_inputs = np.dot(X, self.weights_input_to_hidden)

            # Signals from hidden layer
            hidden_output = self.activation_function(hidden_inputs)

            # Signals into from the hidden layer to the final output layer
            final_inputs = np.dot(hidden_output, self.weights_hidden_to_output)

            # Signals from final output layer
            final_output = final_inputs

            ### End forward pass ###

            #### Implement the backward pass here ####
            ### Backward pass ###

            # Determine the error
            error = y - final_output

            output_error_term = error

            # Hidden layer's contribution to the error
            hidden_error = np.dot(self.weights_hidden_to_output, output_error_term)

            # Backpropagated error terms
            hidden_error_term = hidden_error * hidden_output * (1 - hidden_output)

            ### End backward pass ###

            ### Weight steps ###

            # Weight step (input to hidden)
            delta_weights_i_h += hidden_error_term * X[:, None]
            # Weight step (hidden to output)
            delta_weights_h_o += output_error_term * hidden_output[:, None]


        # Update the weights with gradient descent step
        self.weights_hidden_to_output += self.lr * delta_weights_h_o / n_records
        self.weights_input_to_hidden += self.lr * delta_weights_i_h / n_records

    def run(self, features):
        ''' Run a forward pass through the network with input features
            Arguments
            ---------
            features: 1D array of feature values
        '''

        #### Implement the forward pass here ####
        # Hidden layer
        hidden_inputs = np.dot(features, self.weights_input_to_hidden) # signals into hidden layer
        hidden_output = self.activation_function(hidden_inputs) # signals from hidden layer

        # Output layer
        final_inputs = np.dot(hidden_output, self.weights_hidden_to_output) # signals into final output layer
        final_output = final_inputs # signals from final output layer

        return final_output


#########################################################
# Set your hyperparameters here
##########################################################
iterations = 100
learning_rate = 0.1
hidden_nodes = 2
output_nodes = 1
