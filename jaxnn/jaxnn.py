# How to run this?
# python jaxnn.py
#
# Medium-term goal: python jaxnn.py --input=[10.5,12,13,5,2] --labels=[1,1,0,0] --num_passes=10
# The above command will take the input tensor, run 10 passes on a 2x2 FC
# neural network optimizing for a binary classification on the labels given.
import jax.numpy as np
from jax import grad, jit, vmap, ops

class Neuron(object):
    inputs = []
    weights = []
    bias = 0
    learning_rate = 0.1 # Toy value. Real values are like 1e-6

    def __init__(self, num_inputs):
        self.weights = np.ones(num_inputs)

    def forward(self, inputs):
        self.inputs = inputs # store these for backward pass
        linear = np.dot(self.weights, inputs) + self.bias
        relu = lambda x: x if x > 1 else 0
        return relu(linear)

    def backward(self, input_loss_gradients):
        sum_loss_gradients = np.sum(input_loss_gradients) # This is dL/dq
        local_derivatives = grad(self.forward)(self.inputs) # can be computed and cached during forward pass. This is the dq/dx, dq/dy, etc.
        global_loss_derivatives = sum_loss_gradients * local_derivatives # Chain rule for global derivative.

        for i, weight in enumerate(self.weights): # Update weights
            self.weights = ops.index_update(self.weights, i, weight - (self.learning_rate * global_loss_derivatives[i])) # SGD optimizer

        return global_loss_derivatives
