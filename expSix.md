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

Alternate implementation

```
# Energy-Efficient Routing Simulation (simple, Colab-ready)
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
np.random.seed(42)

# ------- Parameters -------
N = 25                     # number of sensor nodes (excluding sink)
area = 100                 # area side length
comm_range = 30.0          # communication range
initial_energy = 2.0       # initial energy units per node
packet_bits = 2000         # packet size in bits
E_elec = 50e-9             # J/bit (kept for formula; scaled later)
E_amp = 100e-12            # J/bit/m^2 (kept for formula; scaled later)

# scale energies so values are visible in plots (only scaling, doesn't change relative behavior)
scale = 1e9
E_elec *= scale
E_amp *= scale

rounds = 300               # max rounds to simulate

# ------- Create nodes and sink -------
positions = np.random.rand(N, 2) * area
sink_pos = np.array([area/2, area/2])
all_positions = np.vstack([positions, sink_pos])  # last index is sink
sink_idx = N

# ------- Build connectivity graph -------
def build_graph(pos, r):
    G = nx.Graph()
    for i in range(len(pos)):
        G.add_node(i, pos=pos[i])
    for i in range(len(pos)):
        for j in range(i+1, len(pos)):
            d = np.linalg.norm(pos[i]-pos[j])
            if d <= r:
                G.add_edge(i, j, weight=d)
    return G

G = build_graph(all_positions, comm_range)

# ensure graph connectivity to sink by re-drawing until connected (small N)
if not nx.is_connected(G):
    # add edges to connect isolated nodes to sink if out of range (for demo simplicity)
    for i in range(N):
        if not nx.has_path(G, i, sink_idx):
            d = np.linalg.norm(all_positions[i] - all_positions[sink_idx])
            G.add_edge(i, sink_idx, weight=d)

# ------- Energy model functions -------
def energy_tx(d):
    """Energy to transmit one packet over distance d."""
    return E_elec * packet_bits + E_amp * packet_bits * (d**2)

def energy_rx():
    """Energy to receive one packet."""
    return E_elec * packet_bits

# ------- Routing strategies -------
def shortest_path_route(G, src, dst):
    """Return node list for shortest-path (distance) routing."""
    try:
        return nx.shortest_path(G, src, dst, weight='weight')
    except nx.NetworkXNoPath:
        return None

def energy_aware_route(G, src, dst, residual_energy):
    """
    Greedy energy-aware multi-hop:
    from current node, choose neighbor that maximizes (residual_energy / distance)
    while approaching the sink. Stops if stuck.
    """
    path = [src]
    cur = src
    visited = set([src])
    for _ in range(len(G.nodes())+5):
        if cur == dst:
            return path
        neighbors = [n for n in G.neighbors(cur) if n not in visited]
        if not neighbors:
            return None
        # score = residual_energy[n] / distance(cur,n)  (higher better), also prefer closer to sink
        scores = []
        for n in neighbors:
            d = G[cur][n]['weight']
            # small term to encourage progress toward sink
            d_to_sink_cur = np.linalg.norm(all_positions[cur]-all_positions[dst])
            d_to_sink_n = np.linalg.norm(all_positions[n]-all_positions[dst])
            progress = (d_to_sink_cur - d_to_sink_n)
            score = (residual_energy[n] + 1e-6) / (d + 1e-6) + 0.5 * progress
            scores.append((score, n))
        scores.sort(reverse=True)
        next_hop = scores[0][1]
        path.append(next_hop)
        visited.add(next_hop)
        cur = next_hop
    return None

# ------- Simulation loop -------
def simulate(strategy='shortest'):
    residual = np.ones(N+1) * initial_energy
    residual[sink_idx] = 1e9  # sink is assumed powered (infinite)
    avg_energy = []
    first_dead_round = None

    for r in range(rounds):
        # pick a random source sensor (not sink) to send to sink
        src = np.random.randint(0, N)
        dst = sink_idx

        if strategy == 'shortest':
            path = shortest_path_route(G, src, dst)
        else:
            path = energy_aware_route(G, src, dst, residual)

        if path is None or len(path) < 2:
            # no route, skip this round
            avg_energy.append(residual[:N].mean())
            continue

        # forward packet hop-by-hop
        for i in range(len(path)-1):
            tx = path[i]
            rx = path[i+1]
            d = np.linalg.norm(all_positions[tx] - all_positions[rx])
            etx = energy_tx(d)
            erx = energy_rx()
            # subtract energy if node still alive
            if residual[tx] > 0:
                residual[tx] -= etx
            if residual[rx] > 0 and rx != sink_idx:
                residual[rx] -= erx

        avg_energy.append(residual[:N].mean())

        # record first node death
        if first_dead_round is None and np.any(residual[:N] <= 0):
            first_dead_round = r

    return {
        'avg_energy': np.array(avg_energy),
        'first_dead_round': first_dead_round if first_dead_round is not None else rounds,
        'residual': residual.copy()
    }

# ------- Run both strategies -------
res_short = simulate('shortest')
res_energy = simulate('energy')

print("First node death (shortest-path):", res_short['first_dead_round'])
print("First node death (energy-aware):", res_energy['first_dead_round'])

# ------- Plot results -------
plt.figure(figsize=(10,5))
plt.plot(res_short['avg_energy'], label='Shortest-path')
plt.plot(res_energy['avg_energy'], label='Energy-aware')
plt.xlabel('Rounds')
plt.ylabel('Average Residual Energy (per node)')
plt.title('Average Node Energy over Rounds')
plt.legend()
plt.grid(True)
plt.show()

# Visualize final residual energy (node color)
plt.figure(figsize=(6,6))
pos_dict = {i: all_positions[i] for i in range(N+1)}
node_colors = [max(0, min(1, res_energy['residual'][i]/initial_energy)) for i in range(N)]
nx.draw(G, pos=pos_dict, node_color=node_colors + [0.0], cmap=plt.cm.viridis, with_labels=True)
plt.title('Network (node color ∝ residual energy, sink marked last)')
plt.show()
 
```

Key observations during performance

* Energy-efficient routing consumes energy more slowly.
* Network lifetime increases because nodes with low battery are avoided.
* Normal routing depletes nodes quickly, especially those on shortest paths.
* Energy-aware techniques distribute load more evenly.

Conclusion
Energy-efficient routing significantly improves network lifetime by selecting paths that consider both distance and residual energy. This leads to slower energy depletion and more balanced load distribution across all sensor nodes.