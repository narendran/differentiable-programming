# How to run this?
# python bread_toaster.py
#
# Given the number of breads, return the expected toasting time.
import jax.numpy as np
from jax.nn import relu
from jax import grad, jit, vmap, ops

FIXED_TOASTING_DURATION_SECONDS = 30

class BreadToaster(object):

    def toast_time_fixed(self, num_breads):
        return num_breads * FIXED_TOASTING_DURATION_SECONDS

class LearningBreadToaster(object):
    toasting_duration_seconds = 0.0
    num_breads_input = 0.0
    learning_rate = 0.001

    def __init__(self, toasting_duration_seconds):
        self.toasting_duration_seconds = toasting_duration_seconds

    def toast_time_variable_forward(self, num_breads):
        self.num_breads_input = float(num_breads) # stored for local derivative calculation
        return num_breads * self.toasting_duration_seconds

    def toast_time_variable_backward(self, loss):
        # print(self.num_breads_input)
        local_derivative = float(grad(self.toast_time_variable_forward)(self.num_breads_input))
        global_derivative = loss * local_derivative
        self.toasting_duration_seconds = self.toasting_duration_seconds + (self.learning_rate * global_derivative) # directionality is part of loss function here
        print("Loss: ", loss)
        # print("LD: ", local_derivative)
        # print("GD: ", global_derivative)
        print(self.toasting_duration_seconds)
        return

class BiasableBreadToaster(LearningBreadToaster):
    bias_fn = lambda(x): x
    learning_rate = 0.0001

    def __init__(self, toasting_duration_seconds, bias_fn):
        self.toasting_duration_seconds = toasting_duration_seconds
        self.bias_fn = bias_fn

    def toast_time_variable_forward(self, num_breads):
        self.num_breads_input = float(num_breads) # stored for local derivative calculation
        # print("forward")
        # print(self.bias_fn(num_breads))
        # print(self.toasting_duration_seconds)
        return self.bias_fn(num_breads) * self.toasting_duration_seconds
