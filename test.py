import numpy as np
import random
from MDPCGEnv import Env
from FW_PE import *

try:
    a=np.load("aaa.npy")
except IOError:
    a = np.arange(12).reshape(3,2,2)
    np.save("aaa",a)

print(a)

env = Env()
N = env.N
m = np.ones((N)) / N  # the initial distribution

# FW_PE(env,200,m)

pi = np.zeros(5)

for i in range(5):
    pi[i] = np.random.random(1)

print(pi)