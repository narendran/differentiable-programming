# How to run this?
# python jaxnn.py
#
# Medium-term goal: python jaxnn.py --input=[10.5,12,13,5,2] --labels=[1,1,0,0] --num_passes=10
# The above command will take the input tensor, run 10 passes on a 2x2 FC
# neural network optimizing for a binary classification on the labels given.
import jax.numpy as np

class Neuron(object):
    weights = [];
    bias = 0;

    def __init__(self, num_inputs):
        self.weights = np.ones(num_inputs)

    def forward(self, inputs):
        linear = np.dot(self.weights, inputs) + self.bias
        relu = lambda x: x if x > 1 else 0
        return relu(linear)
