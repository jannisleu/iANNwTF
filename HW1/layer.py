import numpy as np

class Layer():

    def __init__(self, input_size: int, num_units: int, activation):
        self.activation = activation
        self.num_units = num_units
        self.input_size = input_size
        self.weights = np.random.normal(loc = 0, scale = 0.2, size = (input_size, num_units))
        self.bias = np.zeros((1, num_units))
        self.prev_layer_output = None

    def forward(self, input):
        self.pre_activation = np.dot(input, self.weights) + self.bias
        self.post_activation = self.activation(self.pre_activation)
        return self.post_activation

    def weights_backwards(self, output, gradients):
        self.d_weights = np.dot(output.T, gradients)
        self.d_bias = np.mean(gradients, axis=0)
    
    def update(self, learning_rate):
        self.weights -= learning_rate * self.d_weights
        self.bias -= learning_rate * self.d_bias 
    

    def __repr__(self):
       return f"Layer(num_units={self.num_units}, input_size={self.input_size}, activation={self.activation})"