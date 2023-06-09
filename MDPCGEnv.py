import igraph as ig
import numpy as np
import pandas as pd
import random

np.random.seed(1)

p=1
Rate = 6
Velocity = 8
FuelPrice = 2.5
FuelEff = 20
TimeStep = 18
TimeInterval = 1/3
# TimeStep times TimeInterval is the total Time the game continues
TotalPoPulation = 2000

nodes_info = pd.read_excel(r'./SanFrancisco_graph.xlsx', sheet_name='nodes')
edges_info = pd.read_excel(r'./SanFrancisco_graph.xlsx', sheet_name='edges')
CustomerDemandRate_info = pd.read_excel(
    r'./SanFrancisco_graph.xlsx', sheet_name='CustomerDemandRate')

D_avg = np.mean(np.array(edges_info.values[:, 2])) 
tau = Rate * D_avg * TimeInterval
# print(D_avg)

# print(edges)


class Env:
    def __init__(self):
        self.T = TimeStep
        self.N = len(nodes_info.values)
        self.Total = TotalPoPulation
        self.g = ig.Graph(n=self.N, directed=True)
        self.p = p
        for edge_info in edges_info.values:
            # print(edge_info)
            source = int(edge_info[0])
            dest = int(edge_info[1])
            self.g.add_edge(source, dest, dist=edge_info[2])
        self.g.vs["name"] = nodes_info.values[:, 1]
        self.g.vs["type"] = nodes_info.values[:, 2]

    def cost(self, t, j, i, x):
        """
        Calculate the cost when driver routes from j to i at time t
        """
        eid = self.g.get_eid(j, i)
        Dist = self.g.es[eid]["dist"]
        if self.g.vs[j]["type"] == "Resident" and self.g.vs[i]["type"] == "Downtown":
            CustomerDemandRate = CustomerDemandRate_info.values[t][1]
        elif self.g.vs[j]["type"] == "Downtown" and self.g.vs[i]["type"] == "Downtown":
            CustomerDemandRate = CustomerDemandRate_info.values[t][2]
        elif self.g.vs[j]["type"] == "Downtown" and self.g.vs[i]["type"] == "Resident":
            CustomerDemandRate = CustomerDemandRate_info.values[t][3]
        elif self.g.vs[j]["type"] == "Resident" and self.g.vs[i]["type"] == "Resident":
            CustomerDemandRate = CustomerDemandRate_info.values[t][4]
        else:
            raise Exception("Wrong Node Type!")
        m = Rate * Dist  # monetary cost
        c_trav = tau * Dist / Velocity + FuelPrice / FuelEff * Dist
        c_wait = tau / CustomerDemandRate
        # print(f"Rate={Rate}",f"Dist={Dist}",f"m={m}")
        # print(f"tau={tau}",f"c_trac={c_trav}",f"c_wait={c_wait}")
        # print(f"x={x}")
        return -m + c_trav + c_wait * TotalPoPulation * x

    def Response(self, x, noise = True):
        d = np.zeros((self.T, self.N, self.N))
        for t in range(self.T):
            for e in self.g.es:
                j = e.source
                i = e.target
                d[t, j, i] = self.cost(t, j, i, x[t, j, i])
        if noise:
            eta = np.random.normal(loc=0, scale=10,size=(self.T,self.N,self.N))
            d += eta
        return d

    def calculate(self, x):
        sum = 0
        for t in range(self.T):
            for e in self.g.es:
                j = e.source
                i = e.target
                sum += self.cost(t, j, i, x[t, j, i]) * x[t, j, i]
        return sum

# env = Env()
# g = env.g
# print(g.es["dist"])
