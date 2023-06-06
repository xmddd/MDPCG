from FW_node_global import cost as cost_node
from FW_population import cost as cost_population
import matplotlib.pyplot as plt

K= 100
# fig = plt.figure()
l1 = plt.plot(range(K),cost_node[:-1],label="node")
l2 = plt.plot(range(K),cost_population,label="population")
plt.legend()

plt.xlabel("Episode")
plt.ylabel("cost")
plt.show()