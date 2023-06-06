from graph import GraphEnv
import numpy as np
import matplotlib.pyplot as plt
import random
import copy


def init_I(m):
    """
    init start position of each population with distribution m
    """
    I = np.zeros((U, N), dtype=int)
    start = np.zeros((U), dtype=int)
    for u in range(U):
        j = random.choices(range(N), weights=m, k=1)[0]
        start[u] = j
        I[u, j] = 1
    return I, start


def init_pi(g):
    """
    Each population u choose policy pi(u) arbitrarily
    """
    pi = np.zeros((T, N, N))

    for t in range(T):
        for j in range(N):
            outd = g.vs[j].outdegree()
            list = [i.index for i in g.vs[j].successors()]
            for i in list:
                pi[t, j, i] = 1/outd
    return pi


def Policy2Prob(I, pi):
    """
    Each population can generate the probabilty distribution p from its own policy pi
    """
    p = np.zeros((T, N, N))
    q = np.zeros((T, N))
    for j in range(N):
        q[0, j] = I[j]
    for t in range(T):
        for j in range(N):
            for i in range(N):
                p[t, j, i] = q[t, j] * pi[t, j, i]
        if t < T-1:
            for j in range(N):
                q[t+1, j] = np.sum(p[t, :, j])
    return p


def Prob2Policy(p):
    """
    Each population can recover its policy from probabilty distribution p
    """
    pi = np.zeros((T, N, N))
    for t in range(T):
        for j in range(N):
            q = np.sum(p[t, j, :])
            for i in range(N):
                if q > 0:
                    pi[t, j, i] = p[t, j, i] / q
    return pi


def Prob2Flow(p):
    x = np.average(p, axis=0)
    print(np.around(x, 6))


def BestResponse(d, debug=False):
    MAX = 1000
    l = np.ones((T, N, N)) * MAX
    v = np.zeros((T, N))
    pi = np.zeros((T, N, N))
    for t in reversed(range(T)):
        for j in range(N):
            list = [i.index for i in g.vs[j].successors()]
            for i in list:
                if t == T-1:
                    l[t, j, i] = d[t, j, i]
                else:
                    l[t, j, i] = d[t, j, i] + v[t+1, i]
            v[t, j] = np.min(l[t, j, :])
            k = np.argmin(l[t, j, :])
            pi[t, j, k] = 1
    return pi


def PlayWithEnv(pi):
    sum = 0
    d = copy.deepcopy(d_init)
    current = copy.deepcopy(start)
    for t in range(T):
        env.init_counter()
        for u in range(U):
            next[u] = env.Act(current[u], pi[u, t, current[u]])
        for u in range(U):
            cost = env.Response(t, current[u], next[u], U)
            sum += cost / U
            d[t, current[u], next[u]] = cost
        current = copy.deepcopy(next)
    return d, sum


# if __name__ == '__main__':
env = GraphEnv()
g = env.g
N = env.N
T = env.T
U = 100
K = 100

m = np.zeros((N))
m[0] = 1
I, start = init_I(m)

pi = np.zeros((U, T, N, N))
p = np.zeros((U, T, N, N))
for u in range(U):
    pi[u] = init_pi(g)

for u in range(U):
    p[u] = Policy2Prob(I[u], pi[u])

    # Prob2Flow(p)

d_init = env.zero_load()

next = np.zeros((U), dtype=int)
pi_tilde = np.zeros((U, T, N, N))
p_tilde = np.zeros((U, T, N, N))
cost = []
for k in range(K):
    d, latency = PlayWithEnv(pi)
    # if k==1:
    #     Prob2Flow(p)
    #     print(d)
    BR = BestResponse(d)
    for u in range(U):
        pi_tilde[u] = BR
    for u in range(U):
        p_tilde[u] = Policy2Prob(I[u], pi_tilde[u])
    alpha = 2/(k+2)
    p = alpha*p_tilde + (1-alpha) * p
    for u in range(U):
        pi[u] = Prob2Policy(p[u])
    # if k==0:
    #     print(pi[0])
        # Prob2Flow(p)
    cost.append(latency)

# fig = plt.figure(num=1, figsize=(4, 4))
# plt.plot(range(K), cost)
# plt.show()
