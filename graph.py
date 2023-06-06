import igraph as ig
import numpy as np
import random


def fx(x):
    return x


def f1(x):
    return 1


def f0(x):
    return 0


class GraphEnv:
    def __init__(self):
        self.T = 4
        self.g = ig.Graph(n=4, edges=[[0, 0], [1, 1], [2, 2], [3, 3], [0, 1], [
            0, 2], [1, 3], [2, 3], [1, 2], [2, 1]], directed=True)
        self.g.vs["name"] = ['0', '1', '2', '3']
        e1 = [f1 for t in range(self.T)]
        e0 = [f0 for t in range(self.T)]
        ex = [fx for t in range(self.T)]
        self.g.es["latency"] = [e1, e1, e1, e0, ex, e1, ex, e1, e1, e1]
        for e in self.g.es:
            e["name"] = f"{e.source}-{e.target}"
        self.N = self.g.vcount()
        self.cnt = np.zeros((self.N, self.N), dtype=int)

    def calculate(self, x, debug=False):
        sum = 0
        for t in range(self.T):
            for e in self.g.es:
                sum += e["latency"][t](x[t, e.source, e.target]) * \
                    x[t, e.source, e.target]
                if debug:
                    print(e["latency"][t](x[t, e.source, e.target])
                          * x[t, e.source, e.target])
        return sum

    def init_counter(self):
        self.cnt = np.zeros((self.N, self.N), dtype=int)

    def Act(self, curr, pi):
        next = random.choices(range(self.N), weights=pi, k=1)[0]
        self.cnt[curr, next] += 1
        return next

    def Response(self, t, curr, next,U):
        eid = self.g.get_eid(curr, next)
        return self.g.es[eid]["latency"][t](self.cnt[curr, next]/U)

    def zero_load(self):
        d = np.zeros((self.T, self.N, self.N))
        for t in range(self.T):
            for e in self.g.es:
                d[t, e.source, e.target] = e["latency"][t](0)
        return d
# print(g.vcount())
# print(g.es[4]["name"])
# print(g)
# print(g.vs[0].successors()[0].index)
# eid = g.get_eid(1,3)
# print(g.es[eid]["latency"](10))
