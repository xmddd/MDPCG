from MDPCGEnv import Env
import numpy as np
import matplotlib.pyplot as plt
from subroutines import *

def FW_PE(env, K, m, ifplot = False):
    N = env.N
    T = env.T
    g = env.g
    p = env.p
    K = int(K / (p+1))
    # print(K)
    # m = np.ones((N)) / N

    cost = []
    x_list = []

    # Only Regression

    U = np.zeros((T, N, N, p+1, 1))  # represent \sum_i D_iX_i
    V = np.zeros((T, N, N, p+1, p+1))  # represent \lambda I + \sum_i X_iX_i^T
    V = V + np.identity(p+1)
    for i in range(p+1):
        x = init_flow(env, g, m)
        for k in range(K):
            D = env.Response(x, noise=True)
            theta, V, U = Regression(env, x, V, U, D)
            x_list.append(x)
            cost.append(env.calculate(x))

    for k in range(int(np.sqrt(K))):
        x = init_flow(env, g, m)
        d_estimate = Estimate(env,theta, x)
        pi = ValueIteration(env, d_estimate)
        x_tilde = RetrieveDensity(env, pi, m)
        alpha = 2/(k+2)
        x = alpha*x_tilde + (1-alpha) * x
        x_list.append(x)
        cost.append(env.calculate(x))
    # print(env.delay(0,0,1,x[0,0,1]))
    # raise Exception("End")
    # print(f"cost={calculate(x)}")

    if ifplot:
        fig = plt.figure(num=1, figsize=(4, 4))
        plt.plot(range((p+1)*K+int(np.sqrt(K))), cost)
        plt.show()

    return x_list, cost