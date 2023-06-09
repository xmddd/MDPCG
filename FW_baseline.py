# from MDPCGEnv import Env
import numpy as np
import matplotlib.pyplot as plt
from subroutines import *

# env = Env()
def FW_baseline(env, K, m, noise= False, ifplot = False):
    N = env.N
    T = env.T
    g = env.g
    # K = 100
    # m = np.ones((N)) / N

    x = init_flow(env, g, m)
    # print(env.delay(0,0,1,x[0,0,1]))
    # raise Exception("End")
    # print(f"cost={calculate(x)}")
    # cost = [env.calculate(x)]
    cost = []
    x_list=[]

    for k in range(K):
        D = env.Response(x, noise=noise)
        pi = ValueIteration(env, D)
        x_tilde = RetrieveDensity(env, pi, m)
        alpha = 2/(k+2)
        x = alpha*x_tilde + (1-alpha) * x
        x_list.append(x)
        cost.append(env.calculate(x))
    # print(x)
    # calculate(x,debug = True)
    if ifplot:
        fig = plt.figure(num=1, figsize=(4, 4))
        plt.plot(range(K), cost)
        plt.show()

    return x_list, cost

