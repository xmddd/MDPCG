from MDPCGEnv import Env
import numpy as np
import matplotlib.pyplot as plt
from subroutines import ValueIteration, RetrieveDensity

def init_flow(g, m):
    """
    Init an flow allocation x for graph g.
    """
    # pi = np.zeros((N, N, T))
    x = np.zeros((T, N, N))
    y = np.zeros((T, N))
    y[0] = m
    for t in range(T):
        for j in range(N):
            outd = g.vs[j].outdegree()
            list = [i.index for i in g.vs[j].successors()]
            for i in list:
                x[t, j, i] = 1/outd * y[t, j]
        if t < T-1:
            for j in range(N):
                y[t+1, j] = np.sum(x[t, :, j])

    return x




env = Env()
N = env.N
T = env.T
g = env.g
K = 100
m = np.ones((N)) / N

x = init_flow(g, m)
# print(env.delay(0,0,1,x[0,0,1]))
# raise Exception("End")
# print(f"cost={calculate(x)}")
cost = [env.calculate(x)]

for k in range(K):
    D = env.Response(x, noise=False)
    pi = ValueIteration(env, D)
    x_tilde = RetrieveDensity(env, pi, m)
    alpha = 2/(k+2)
    x = alpha*x_tilde + (1-alpha) * x
    cost.append(env.calculate(x))
# print(x)
# calculate(x,debug = True)
fig = plt.figure(num=1, figsize=(4, 4))
plt.plot(range(K-10), cost[10:-1])
plt.show()
