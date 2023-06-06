from graph import GraphEnv
import numpy as np
import matplotlib.pyplot as plt


def init_flow(g, m):
    """
    init an flow allocation x for graph g,
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


def ValueIteration(x, debug=False):
    MAX = 1000
    l = np.ones((T, N, N)) * MAX
    v = np.zeros((T, N))
    pi = np.zeros((T, N), dtype=int)
    for t in reversed(range(T)):
        for j in range(N):
            list = [i.index for i in g.vs[j].successors()]
            for i in list:
                eid = g.get_eid(j, i)
                if t == T-1:
                    l[t, j, i] = g.es[eid]["latency"][t](x[t, j, i])
                else:
                    l[t, j, i] = g.es[eid]["latency"][t](
                        x[t, j, i]) + v[t+1, i]
            v[t, j] = np.min(l[t, j, :])
            pi[t, j] = np.argmin(l[t, j, :])
    return pi


def RetrieveDensity(pi, m):
    """
    Retrieve a flow from policy pi
    """
    x = np.zeros((T, N, N))
    for t in range(T):
        for i in range(N):
            k = pi[t, i]
            if t == 0:
                x[t, i, k] = m[i]
            else:
                x[t, i, k] = np.sum(x[t-1, :, i])
    return x


env = GraphEnv()
N = env.N
T = env.T
g = env.g
K = 100
m = np.zeros((N))
m[0] = 1

x = init_flow(g, m)
# print(x)
# print(f"cost={calculate(x)}")
cost = [env.calculate(x)]

for k in range(K):
    pi = ValueIteration(x)
    x_tilde = RetrieveDensity(pi, m)
    alpha = 2/(k+2)
    x = alpha*x_tilde + (1-alpha) * x
    cost.append(env.calculate(x))
# print(x)
# calculate(x,debug = True)
# fig = plt.figure(num=1, figsize=(4, 4))
# plt.plot(range(K), cost[:-1])
# plt.show()
