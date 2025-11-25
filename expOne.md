**Experiment No. 1: Node Deployment Strategies**

**Theory for Viva**

Purpose of deployment

* To ensure good coverage, connectivity, network lifetime, and sensing reliability.
* Deployment determines how well the WSN monitors an area.

Types of deployment strategies

* Random deployment

  * Nodes are scattered arbitrarily.
  * Used when terrain is inaccessible (forests, war zones).
  * Coverage is unpredictable; may create coverage holes.
* Uniform deployment

  * Distance between nodes is nearly equal.
  * Provides balanced coverage and moderate connectivity.
  * Easier to analyze but needs controlled placement.
* Grid-based deployment

  * Nodes arranged in rows and columns.
  * Provides the most structured coverage and predictable connectivity patterns.
  * Ideal for environments where planned installation is possible.

Performance factors

* Coverage: How well the monitored area is sensed.
* Connectivity: Ability of nodes to communicate with neighbors or sink.
* Energy efficiency: Better-arranged nodes reduce retransmissions and energy waste.
* Fault tolerance: Dense or structured placement reduces the impact of node failures.

Observations typically seen

* Random placement creates uneven node density.
* Uniform placement maintains balanced spacing.
* Grid placement maximizes area coverage with minimal overlapping sensing regions.

Summary points for viva

* Deployment strategy directly influences the quality and efficiency of a WSN.
* Grid deployment is generally optimal when structured placement is possible.
* Random deployment is used where manual placement is not feasible.
* Coverage holes and connectivity gaps occur mostly in random deployments.

---

**Implementation for Performance**

Objective
To generate, visualize, and compare node positions for random, uniform, and grid-based deployment within a defined area.

Requirements

* Python with NumPy and Matplotlib or MATLAB (any one).
* System capable of plotting scatter graphs.

Procedure (conceptual steps)

1. Define sensing area dimensions, e.g., 100 × 100 units.
2. Select number of nodes, e.g., 50 nodes.
3. Generate node coordinates using three strategies:

   * Random: randomly generated (x, y) positions.
   * Uniform: evenly spaced positions using linear spacing.
   * Grid: matrix-style placement using meshgrid.
4. Plot all three deployments separately to compare coverage visually.
5. Observe node density, spacing, and presence of coverage holes.
6. Interpret the deployment patterns based on coverage and connectivity expectations.

Python implementation (fast demonstration)

```
import numpy as np
import matplotlib.pyplot as plt

N = 50
area = 100

x_rand = area * np.random.rand(N)
y_rand = area * np.random.rand(N)

steps = int(np.sqrt(N))
x_uni = np.linspace(0, area, steps)
y_uni = np.linspace(0, area, steps)
xu, yu = np.meshgrid(x_uni, y_uni)

plt.figure(figsize=(12,4))

plt.subplot(1,3,1)
plt.scatter(x_rand, y_rand)
plt.title("Random Deployment")

plt.subplot(1,3,2)
plt.scatter(xu, yu)
plt.title("Uniform Deployment")

plt.subplot(1,3,3)
plt.scatter(xu, yu)
plt.title("Grid Deployment")

plt.show()
```

Alternate implementation

```
# Experiment 1 — realistic demo (Colab-ready)
# Generates Random / Uniform / Grid deployments,
# estimates area coverage by Monte-Carlo sampling,
# computes connectivity stats (graph) and a simple Tx energy cost model.

import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
import math
from tqdm import trange

# --- parameters (changeable during demo) ---
AREA = 100.0               # side of square area in meters
N = 49                     # number of nodes (choose a perfect square for grid)
sensing_r = 12.0           # sensing radius (meters)
comm_r = 25.0              # communication range (meters)
mc_samples = 40000         # Monte Carlo samples to estimate coverage (higher -> better estimate)
np.random.seed(42)

# --- helper functions ---
def deploy_random(n, area):
    return np.random.rand(n, 2) * area

def deploy_uniform(n, area):
    # try to create near-uniform spacing by jittering grid points
    s = int(round(math.sqrt(n)))
    xs = np.linspace(area/(2*s), area-area/(2*s), s)
    grid = np.array(np.meshgrid(xs, xs)).T.reshape(-1,2)
    # if n < s^2, trim; if n > s^2, pad with random
    pts = grid[:n]
    jitter = (np.random.rand(len(pts),2)-0.5) * (area/s*0.1)  # small jitter
    return pts + jitter

def deploy_grid(n, area):
    s = int(round(math.sqrt(n)))
    xs = np.linspace(0, area, s)
    xv, yv = np.meshgrid(xs, xs)
    pts = np.vstack((xv.flatten(), yv.flatten())).T
    return pts[:n]

def estimate_coverage(nodes, sensing_r, area, samples=20000):
    # Monte Carlo: fraction of random points within any sensor circle
    pts = np.random.rand(samples,2) * area
    # vectorized distance check
    dx = pts[:,0][:,None] - nodes[:,0][None,:]
    dy = pts[:,1][:,None] - nodes[:,1][None,:]
    d2 = dx*dx + dy*dy
    covered = (d2 <= sensing_r**2).any(axis=1)
    return covered.mean()  # fraction covered

def connectivity_stats(nodes, comm_r):
    n = len(nodes)
    G = nx.Graph()
    G.add_nodes_from(range(n))
    # add edges for pairs within comm_r
    for i in range(n):
        for j in range(i+1,n):
            if np.linalg.norm(nodes[i]-nodes[j]) <= comm_r:
                G.add_edge(i,j)
    comps = nx.number_connected_components(G)
    avg_degree = np.mean([d for _, d in G.degree()]) if n>0 else 0
    # average shortest path length from node 0 to others (if connected)
    if nx.is_connected(G):
        avg_sp = nx.average_shortest_path_length(G)
    else:
        avg_sp = None
    return {'graph': G, 'components': comps, 'avg_degree': avg_degree, 'avg_shortest_path': avg_sp}

def tx_energy_cost(nodes, comm_r, E_tx_per_meter=0.0001):
    # simple model: sending a packet cost proportional to distance^2 (free-space)
    # compute average transmission energy for each node to its nearest neighbor toward sink (node 0 acting as sink)
    n = len(nodes)
    sink = nodes[0]
    costs = []
    for i in range(n):
        d = np.linalg.norm(nodes[i]-sink)
        if d <= comm_r:
            costs.append(E_tx_per_meter * d**2)  # direct
        else:
            # multi-hop via nearest neighbor within comm range
            # find nearest neighbor within comm range; if none, assume very high cost
            dists = np.linalg.norm(nodes - nodes[i], axis=1)
            candidates = dists[(dists>0) & (dists<=comm_r)]
            if len(candidates)==0:
                costs.append(E_tx_per_meter * (d**2) * 2) # penalize disconnected
            else:
                # assume one hop to neighbor + neighbor to sink (approx)
                costs.append(E_tx_per_meter * ( (candidates.min())**2 + (d - candidates.min())**2 ))
    return np.mean(costs)

# --- run deployments and collect metrics ---
deployments = {
    'Random' : deploy_random(N, AREA),
    'Uniform' : deploy_uniform(N, AREA),
    'Grid' : deploy_grid(N, AREA)
}

results = {}
for name, nodes in deployments.items():
    cov = estimate_coverage(nodes, sensing_r, AREA, samples=mc_samples)
    conn = connectivity_stats(nodes, comm_r)
    etx = tx_energy_cost(nodes, comm_r)
    results[name] = {'coverage': cov, 'components': conn['components'],
                     'avg_degree': conn['avg_degree'], 'avg_sp': conn['avg_shortest_path'],
                     'avg_tx_energy': etx, 'nodes': nodes, 'graph': conn['graph']}

# --- print concise summary ---
for name, r in results.items():
    print(f"{name:8s} | Coverage: {r['coverage']*100:5.1f}% | Components: {r['components']:2d} | "
          f"AvgDeg: {r['avg_degree']:.2f} | AvgSP: {str(r['avg_sp']):8s} | AvgTxE: {r['avg_tx_energy']:.6f}")

# --- plotting for demo ---
fig, axs = plt.subplots(1,3, figsize=(15,4))
for ax, (name, r) in zip(axs, results.items()):
    nodes = r['nodes']
    ax.scatter(nodes[:,0], nodes[:,1], s=40)
    # draw sensing circles (semi-transparent)
    for xy in nodes:
        circ = plt.Circle((xy[0], xy[1]), sensing_r, color='C0', alpha=0.06)
        ax.add_artist(circ)
    ax.set_title(f"{name}\nCov={r['coverage']*100:.1f}% Comp={r['components']}")
    ax.set_xlim(0, AREA); ax.set_ylim(0, AREA); ax.set_aspect('equal')
plt.tight_layout()
plt.show()
```

Key observations during performance

* Random deployment shows irregular distribution and uneven coverage.
* Uniform deployment shows evenly spaced nodes but still not perfectly structured.
* Grid deployment shows a clear, organized pattern with predictable spacing.

Conclusion
Grid deployment provides the most efficient and reliable coverage, uniform deployment gives balanced spacing, and random deployment results in unpredictable coverage patterns.