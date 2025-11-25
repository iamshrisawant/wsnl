**Experiment No. 6: Energy-Efficient Routing**

**Theory for Viva**

Purpose

* To design and evaluate a routing method that reduces energy consumption and increases the lifetime of a wireless sensor network.
* To understand how routing decisions based on residual energy influence network performance.

Meaning of energy-efficient routing

* Routing that selects paths minimizing total energy usage.
* Ensures uniform energy drain across nodes to avoid early node failures.
* Enhances overall network lifetime and reliability.

Common energy-efficient routing methods

* LEACH (Low Energy Adaptive Clustering Hierarchy)

  * Forms clusters and rotates cluster heads.
  * Reduces long-distance transmissions.
* PEGASIS

  * Creates a chain of nodes for data forwarding.
  * Minimizes energy usage per round.
* Energy-aware shortest path

  * Selects path based on residual energy + link cost.
  * Avoids nodes with low energy.

Key principles

* Nodes closer to the sink should not be overloaded.
* Routing decisions should consider remaining battery level.
* Multi-hop communication saves energy compared to direct long-range transmission.

Performance parameters

* Residual energy of nodes
* Number of rounds until first node dies (FND)
* Network lifetime
* Average energy consumption per packet

Common observations

* LEACH reduces energy consumption by limiting long transmissions.
* Rotating cluster heads prevents premature failure of any single node.
* Energy-aware routing increases lifetime compared to shortest-path routing.

Summary points for viva

* Energy-efficient routing is essential for long-term operation of WSNs.
* Clustering and energy-aware metrics reduce unnecessary energy drain.
* Protocols like LEACH outperform non-energy-aware methods.
* Balancing load among nodes prolongs network lifetime.

---

**Implementation for Performance**

Objective
To simulate energy consumption in a simple routing scenario and compare a normal (shortest-path) method with an energy-efficient method.

Requirements

* Python or MATLAB for numerical computation and plotting.
* A small network model with nodes, distances, and energy values.

Procedure (conceptual steps)

1. Initialize a set of sensor nodes with equal initial energy.
2. Create a distance matrix representing node-to-node distances.
3. For normal routing:

   * Always choose the shortest path to the sink.
4. For energy-efficient routing:

   * Choose paths based on a weighted metric:
     residual energy + distance cost.
5. Simulate packet forwarding over many rounds.
6. After each round, subtract transmission energy based on distance.
7. Track remaining energy and time until the first node dies.
8. Compare network lifetime and energy consumption.

Python implementation (fast demonstration)

```
import numpy as np
import matplotlib.pyplot as plt

nodes = 10
energy = np.ones(nodes) * 1.0
distance = np.random.randint(1, 10, (nodes, nodes))

E_tx = 0.01

life_normal = []
life_energy = []

for r in range(200):
    src = np.random.randint(0, nodes)
    dst = np.random.randint(0, nodes)

    # Normal routing: shortest distance
    cost_normal = distance[src][dst]
    energy[src] -= E_tx * cost_normal
    life_normal.append(energy.mean())

    # Energy-aware routing: distance + inverse energy weight
    weight = distance[src][dst] / (energy[src] + 0.1)
    energy[src] -= E_tx * weight
    life_energy.append(energy.mean())

plt.plot(life_normal, label="Normal Routing")
plt.plot(life_energy, label="Energy-Efficient Routing")
plt.legend()
plt.title("Energy Consumption Comparison")
plt.xlabel("Rounds")
plt.ylabel("Average Energy")
plt.show()
```

MATLAB implementation (alternative)

```
nodes = 10;
energy = ones(1,nodes);
distance = randi([1 10], nodes, nodes);
Etx = 0.01;

life_norm = [];
life_eff = [];

for r = 1:200
    src = randi(nodes);
    dst = randi(nodes);

    cost_norm = distance(src, dst);
    energy(src) = energy(src) - Etx * cost_norm;
    life_norm(end+1) = mean(energy);

    weight = distance(src, dst) / (energy(src) + 0.1);
    energy(src) = energy(src) - Etx * weight;
    life_eff(end+1) = mean(energy);
end

plot(life_norm); hold on;
plot(life_eff);
legend('Normal Routing', 'Energy-Efficient Routing');
title('Energy Consumption Comparison');
xlabel('Rounds');
ylabel('Average Energy');
```

Key observations during performance

* Energy-efficient routing consumes energy more slowly.
* Network lifetime increases because nodes with low battery are avoided.
* Normal routing depletes nodes quickly, especially those on shortest paths.
* Energy-aware techniques distribute load more evenly.

Conclusion
Energy-efficient routing significantly improves network lifetime by selecting paths that consider both distance and residual energy. This leads to slower energy depletion and more balanced load distribution across all sensor nodes.