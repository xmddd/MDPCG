from MDPCGEnv import Env
import numpy as np
import matplotlib.pyplot as plt
from subroutines import *


def FW_RM(env, K,m, ifplot = False):
    N = env.N
    T = env.T
    g = env.g
    p = env.p

    alpha = np.sqrt(len(g.es) / (T * K))
    # K = 100
    # m = np.ones((N)) / N

    x = init_flow(env, g, m)
    # print(env.delay(0,0,1,x[0,0,1]))
    # raise Exception("End")
    # print(f"cost={calculate(x)}")
    # cost = [env.calculate(x)]
    cost = []
    x_list= []

    U = np.zeros((T, N, N, p+1, 1))  # represent \sum_i D_iX_i
    V = np.zeros((T, N, N, p+1, p+1))  # represent \lambda I + \sum_i X_iX_i^T
    V = V + np.identity(p+1)

    for k in range(K):
        D = env.Response(x, noise=True)
        theta, V, U = Regression(env, x, V, U, D)
        d_estimate = Estimate(env, theta, x)
        pi = ValueIteration(env, d_estimate)
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