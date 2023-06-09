from MDPCGEnv import Env
import numpy as np
from FW_baseline import *
from FW_RM import *
from FW_PE import *
import matplotlib.pyplot as plt

K_standard = 1000  # the episode amount to reach accurate Wardrop Equilibrium
K = 800  # the episode amound in online MDPCG environment

env = Env()
N = env.N
m = np.ones((N)) / N  # the initial distribution

try:
    x_star=np.load("x_star.npy")
except IOError:
    x, _ = FW_baseline(env, K_standard, m)
    x_star = x[-1]
    np.save("x_star",x_star)

x_baseline_list, _ = FW_baseline(env, K, m, noise=True)
x_RM_list, _ = FW_RM(env, K, m)
x_PE_list, _ = FW_PE(env, K, m)
x_baseline_norm = []
x_RM_norm = []
x_PE_norm = []

for i in range(len(x_baseline_list)):
    x_delta = x_baseline_list[i]-x_star
    x_baseline_norm.append(np.linalg.norm(x_delta.flatten(), ord=2))

for i in range(len(x_RM_list)):
    x_delta = x_RM_list[i]-x_star
    x_RM_norm.append(np.linalg.norm(x_delta.flatten(), ord=2))

for i in range(len(x_PE_list)):
    x_delta = x_PE_list[i]-x_star
    x_PE_norm.append(np.linalg.norm(x_delta.flatten(), ord=2))

l_baseline = plt.plot(range(K), x_baseline_norm, label = "baseline")
l_RM = plt.plot(range(K), x_RM_norm, label = "RM")
l_PE = plt.plot(range(K), x_PE_norm[:K], label= "PE_Expolarion")
l_PE_last = plt.plot(range(K, len(x_PE_norm)), x_PE_norm[K:],label="PE_Exploitation")

plt.legend(loc = 1)
plt.show()