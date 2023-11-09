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
          layer.prev_layer_output = output
          output = layer.forward(output)
      return output
    
    def backwards(self, grad):
        for l in reversed(self.layers):
            if l == self.layers[0]:
                grad = l.activation.backwards(l.post_activation, grad)
                l.weights_backwards(l.prev_layer_output, grad)
            else:
                if l == self.layers[-1]:
                    l.weights_backwards(l.prev_layer_output, grad)
                    grad = np.dot(l.weights, grad.T)
                else:
                    grad = l.activation.backwards(l.post_activation, grad)
                    l.weights_backwards(l.prev_layer_output, grad)
                    grad = np.dot(l.weights, grad.T)

    def update(self, learning_rate):
        for l in self.layers:
            l.update(learning_rate)
            
    
    
