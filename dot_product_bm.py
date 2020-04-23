import jax.numpy as np
from jax import grad, jit, vmap
from jax import random
from timeit import default_timer as timer
import csv

# random number generation to test out matrix multiplication (SIMD speed up with jit)
key = random.PRNGKey(0)
mat1 = random.normal(key, (10,10))
key = random.PRNGKey(100)
mat2 = random.normal(key, (10,10))

def mydot(mat1):
    prod = mat1
    for i in range(0, OPERATION_SEQUENCE_LENGTH):
        prod = np.dot(mat1, prod)
    return prod
jitted_mydot = jit(mydot)

PROFILE_ITERATIONS = 1000

mydot_times_ms = {}
jitted_mydot_times_ms = {}

for OPERATION_SEQUENCE_LENGTH in range(10, 101, 10):
    print(OPERATION_SEQUENCE_LENGTH)

    start = timer()
    for i in range(0,PROFILE_ITERATIONS):
        prod = mydot(mat1)
    mydot_times_ms[OPERATION_SEQUENCE_LENGTH] = ((timer() - start) * 1000000) / (PROFILE_ITERATIONS)

    start = timer()
    for i in range(0,PROFILE_ITERATIONS):
        prod = jitted_mydot(mat1)
    jitted_mydot_times_ms[OPERATION_SEQUENCE_LENGTH] = ((timer() - start) * 1000000) / (PROFILE_ITERATIONS)

a_file = open("mydot.csv", "w")
writer = csv.writer(a_file)
for key, value in mydot_times_ms.items():
    writer.writerow([key, value])
a_file.close()

b_file = open("jitted_mydot.csv", "w")
writer = csv.writer(b_file)
for key, value in jitted_mydot_times_ms.items():
    writer.writerow([key, value])
b_file.close()
