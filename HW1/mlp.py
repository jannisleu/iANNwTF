import numpy as np
from layer import Layer
from activations import Sigmoid, Softmax

class MLP():

    def __init__(self, input_size: int, layer_sizes: list):
        self.layers = []
        layer_sizes = [input_size] + layer_sizes
        for i in range(len(layer_sizes) - 1):
            #softmax activation for last layer, otherwise Sigmoid
            if i == len(layer_sizes) - 2:
                activation = Softmax()
            else:
                activation = Sigmoid()
            self.layers.append(Layer(layer_sizes[i], layer_sizes[i + 1], activation))
        self.info = print("\n".join([str(layer) for layer in self.layers]))

    def forward(self, input):
      output = input
      for layer in self.layers:
          output = layer.forward(output)
      return output
    
