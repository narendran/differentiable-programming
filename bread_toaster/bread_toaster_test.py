# Run this test with `python -m unittest bread_toaster.bread_toaster_test`

import unittest
import random
import jax.numpy as np
from numpy import testing as np_testing

from bread_toaster import BreadToaster, LearningBreadToaster, BiasableBreadToaster

class TestBreadToaster(unittest.TestCase):

    # def test_fixed_toasting_time(self):
    #     bread_toaster = BreadToaster()
    #     expected_duration_seconds = 60
    #
    #     self.assertEqual(bread_toaster.toast_time_fixed(2), expected_duration_seconds)

    # def test_variable_toasting_time(self):
    #     actual_toasting_duration_seconds = 20
    #     actual_toasting_time_fn = lambda(num_breads): num_breads * actual_toasting_duration_seconds # my toaster should correctly figure this out.
    #
    #     bread_toaster = LearningBreadToaster(100.0)
    #
    #     for i in range(0, 100):
    #         chosen_bread_count = random.randint(1, 10)
    #         predicted_toasting_seconds = bread_toaster.toast_time_variable_forward(chosen_bread_count)
    #         bread_toaster.toast_time_variable_backward(float(actual_toasting_time_fn(float(chosen_bread_count)) - predicted_toasting_seconds))
    #
    #     # Toaster learns toasting duration in ~10 steps
    #     self.assertEqual(int(bread_toaster.toasting_duration_seconds), actual_toasting_duration_seconds)

    # def test_biased_toasting_time_unbiased_learner(self):
    #     actual_toasting_duration_seconds = 20
    #     actual_toasting_time_fn = lambda(num_breads): math.sin(num_breads) * actual_toasting_duration_seconds # my toaster should correctly figure this out.
    #
    #     bread_toaster = LearningBreadToaster(100.0)
    #
    #     for i in range(0, 1000):
    #         chosen_bread_count = random.randint(1, 10)
    #         predicted_toasting_seconds = bread_toaster.toast_time_variable_forward(chosen_bread_count)
    #         print(chosen_bread_count, actual_toasting_time_fn(float(chosen_bread_count)))
    #         bread_toaster.toast_time_variable_backward(float(actual_toasting_time_fn(float(chosen_bread_count)) - predicted_toasting_seconds))
    #
    #     # Toaster learns a value close to 0 to approximate the sine curve
    #     self.assertLess(int(bread_toaster.toasting_duration_seconds), 1)
    #     self.assertGreater(int(bread_toaster.toasting_duration_seconds), -1)



    def test_biased_toasting_time_biased_learner(self):
        actual_toasting_duration_seconds = 20
        actual_toasting_time_fn = lambda(num_breads): (num_breads - (np.power(num_breads, 3)/6) + (np.power(num_breads, 5)/120)) * actual_toasting_duration_seconds # my toaster should correctly figure this out.

        # My bread toaster is proven to have trigonometric properties, but I don't know what.
        bread_toaster = BiasableBreadToaster(100.0, lambda(x): np.sin(x))

        for i in range(0, 1000):
            chosen_bread_count = random.randint(1, 10)
            predicted_toasting_seconds = bread_toaster.toast_time_variable_forward(chosen_bread_count)
            print(chosen_bread_count, actual_toasting_time_fn(float(chosen_bread_count)))
            bread_toaster.toast_time_variable_backward(float(actual_toasting_time_fn(float(chosen_bread_count)) - predicted_toasting_seconds))

        # Toaster learns a value close to 0 to approximate the sine curve
        # self.assertLess(int(bread_toaster.toasting_duration_seconds), 1)
        # self.assertGreater(int(bread_toaster.toasting_duration_seconds), -1)


if __name__ == '__main__':
    unittest.main()
