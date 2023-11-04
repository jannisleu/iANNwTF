import numpy as np

class Layer():

    def __init__(self, input_size: int, num_units: int, activation):
        self.activation = activation
        self.num_units = num_units
        self.input_size = input_size
        self.weights = np.random.normal(loc = 0, scale = 0.2, size = (input_size, num_units))
        self.bias = np.zeros((1, num_units))

    def forward(self, input):

        pre_activation = np.dot(input, self.weights) + self.bias
        post_activation = self.activation(pre_activation)
        return post_activation
    #batchsize, inputsize
    #batchsize, num_units

    def __repr__(self):
       return f"Layer(num_units={self.num_units}, input_size={self.input_size}, activation={self.activation})"

inp =[
    [1,2,3,4],
    [1,2,3,4],
    [1,2,3,4]
    ] #3x4

w = [
    [0.1, 0.2],
    [0.1, 0.2],
    [0.1, 0.2],
    [0.1, 0.2]
] #4x2

pre = [
    [1, 2],
    [1, 2],
    [1, 2]
] #3x2

