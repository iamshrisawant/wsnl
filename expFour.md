**Experiment No. 4: Routing Protocol Simulation**

**Theory for Viva**

Purpose

* To compare the performance of table-driven (proactive) and on-demand (reactive) routing protocols in a wireless sensor network.
* To analyze metrics like latency, packet delivery ratio, and throughput.

Types of routing protocols

* Table-driven (proactive)

  * Examples: DSDV, OLSR
  * Maintains routing tables continuously.
  * Low latency because routes already exist.
  * High control overhead due to periodic updates.
* On-demand (reactive)

  * Examples: AODV, DSR
  * Discovers routes only when needed.
  * Reduces overhead but increases initial delay.
  * Suitable for dynamic or energy-constrained networks.

Key performance parameters

* Latency: time taken from source to destination.
* Packet Delivery Ratio (PDR): ratio of successfully received packets.
* Throughput: successful data transmitted per unit time.
* Control overhead: extra routing messages required.

Common observations

* Proactive routing provides lower latency but consumes more energy due to constant updates.
* Reactive routing saves energy but shows initial delay during route discovery.
* In small or stable networks, proactive is more efficient.
* In large or dynamic networks, reactive is preferred.

Summary points for viva

* Proactive protocols maintain up-to-date routes; reactive protocols create routes when needed.
* Latency is lowest in proactive and highest in reactive (at first transmission).
* Reactive protocols generally consume less energy.
* Routing choice affects network lifetime and performance significantly.

---

**Implementation for Performance**

Objective
To simulate and compare table-driven and on-demand routing using latency, throughput, and packet delivery ratio as performance indicators.

Requirements

* Python or MATLAB for conceptual simulation.
* Graph-based model of nodes and packet transmission.

Procedure (conceptual steps)

1. Create a graph representing sensor nodes and links.
2. For table-driven protocol:

   * Precompute routing tables using algorithms like Dijkstra.
   * Send packets directly using stored routes.
3. For on-demand protocol:

   * Perform route discovery when a packet needs to be sent (e.g., BFS).
   * Cache the route after discovery.
4. Generate random packet transmissions between nodes.
5. Measure latency, delivery ratio, and throughput for each protocol.
6. Plot comparison graphs and interpret the differences.

Python implementation (fast demonstration)

```
import numpy as np
import networkx as nx
import time

G = nx.random_geometric_graph(20, 0.4)

src, dst = 1, 15

start = time.time()
proactive_path = nx.shortest_path(G, src, dst)
latency_pro = time.time() - start

start = time.time()
reactive_path = nx.shortest_path(G, src, dst)
latency_re = time.time() - start + 0.002   # added delay for discovery

print("Proactive Latency:", latency_pro)
print("Reactive Latency:", latency_re)
print("Proactive Path:", proactive_path)
print("Reactive Path:", reactive_path)
```

Alternate implementation

```
# Experiment 4: Routing Protocol Simulation (fixed, runnable in Colab)
# Models: proactive (table-driven) vs reactive (on-demand)
# Metrics: latency, PDR (packet delivery ratio), throughput, control overhead
# Requirements: pip install networkx matplotlib (Colab usually has them)

import random
import math
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt

random.seed(42)
np.random.seed(42)

# ---------- Parameters (tweakable for demonstration) ----------
NUM_NODES = 30
AREA_SIZE = 100            # square area in arbitrary units
RADIO_RANGE = 25           # nodes within this distance have a link
SIM_ROUNDS = 300           # number of packet-generation events
FLOW_PROB = 0.6            # probability that a round generates a packet
TX_TIME_PER_HOP = 0.01     # seconds per hop transmission (for latency calc)
LINK_RELIABILITY = 0.95    # probability that a packet successfully traverses one link
PROACTIVE_UPDATE_INTERVAL = 50   # rounds between proactive routing table updates
REACTIVE_CACHE_TTL = 100          # how long a discovered route stays in cache (rounds)
DISCOVERY_MSG_COST = 1.0    # cost multiplier (approx number of control messages per flooded node)
DISCOVERY_DELAY_PER_HOP = 0.008 # additional delay per hop during route discovery (seconds)
# ---------------------------------------------------------------

# Helper: create random geometric graph with positions
positions = {i: (random.uniform(0, AREA_SIZE), random.uniform(0, AREA_SIZE)) for i in range(NUM_NODES)}
G = nx.Graph()
G.add_nodes_from(range(NUM_NODES))
for i in range(NUM_NODES):
    xi, yi = positions[i]
    for j in range(i+1, NUM_NODES):
        xj, yj = positions[j]
        dist = math.hypot(xi-xj, yi-yj)
        if dist <= RADIO_RANGE:
            G.add_edge(i, j, weight=dist)

# if graph disconnected, we keep it (real networks can be partitioned) but warn
num_components = nx.number_connected_components(G)
print(f"Nodes: {NUM_NODES}, Edges: {G.number_of_edges()}, Connected components: {num_components}")

# Precompute all-pairs shortest paths for proactive routing (paths are lists of nodes)
all_pairs_paths = dict(nx.all_pairs_shortest_path(G))

# State containers for metrics
metrics = {
    'proactive': {'sent':0, 'delivered':0, 'latencies':[], 'control_msgs':0},
    'reactive':  {'sent':0, 'delivered':0, 'latencies':[], 'control_msgs':0}
}

# Reactive route cache: maps (src,dst) -> (path, last_used_round)
reactive_cache = {}

# Function: attempt to deliver packet along path; returns (delivered_bool, latency_seconds, control_msgs_used)
def transmit_along_path(path, include_discovery_delay=0.0):
    hops = max(len(path)-1, 0)
    # simulate per-hop reliability
    for hop in range(hops):
        if random.random() > LINK_RELIABILITY:
            # packet lost on this hop
            # latency still accumulates for attempts until loss
            latency = (hop+1) * TX_TIME_PER_HOP + include_discovery_delay
            return False, latency, 0
    # if all hops succeed
    latency = hops * TX_TIME_PER_HOP + include_discovery_delay
    return True, latency, 0

# Function: reactive route discovery (flooding simulation)
def reactive_discover(src, dst, current_round):
    # simple model: BFS shortest path found by flooding, cost = all nodes visited (approx)
    try:
        path = nx.shortest_path(G, src, dst)
    except nx.NetworkXNoPath:
        return None, 0, 0.0  # no path exists
    # flooding cost ~ number of nodes in connected component (approx nodes that hear discovery)
    comp_nodes = len(max(nx.connected_components(G), key=len))
    control_msgs = int(DISCOVERY_MSG_COST * comp_nodes)
    discovery_delay = (len(path)-1) * DISCOVERY_DELAY_PER_HOP
    # cache route
    reactive_cache[(src,dst)] = (path, current_round)
    return path, control_msgs, discovery_delay

# Function: proactive periodic update cost
def proactive_periodic_control(round_idx):
    # assume each proactive update incurs one control message per edge (simplified)
    return G.number_of_edges()

# Simulation loop
for r in range(1, SIM_ROUNDS+1):
    # 1) Proactive periodic update
    if r % PROACTIVE_UPDATE_INTERVAL == 0:
        metrics['proactive']['control_msgs'] += proactive_periodic_control(r)

    # 2) Randomly decide to generate a packet (flow) this round
    if random.random() > FLOW_PROB:
        continue

    # choose random source and destination from same component if possible
    src = random.randrange(NUM_NODES)
    dst = random.randrange(NUM_NODES)
    while dst == src:
        dst = random.randrange(NUM_NODES)

    # PROACTIVE: route exists if precomputed shortest path available
    metrics['proactive']['sent'] += 1
    if (src in all_pairs_paths) and (dst in all_pairs_paths[src]):
        path = all_pairs_paths[src][dst]
        delivered, latency, _ = transmit_along_path(path, include_discovery_delay=0.0)
        if delivered:
            metrics['proactive']['delivered'] += 1
            metrics['proactive']['latencies'].append(latency)
        # proactive has no per-packet control cost here (cost is in periodic updates)
    else:
        # no route (partition) -> packet lost, count latency zero in that case (or skip)
        pass

    # REACTIVE
    metrics['reactive']['sent'] += 1
    cached = reactive_cache.get((src,dst), None)
    if cached:
        path, last_used = cached
        # check TTL
        if r - last_used > REACTIVE_CACHE_TTL:
            cached = None
            reactive_cache.pop((src,dst), None)
    if not cached:
        # discover route by flooding (if possible)
        path, cmsgs, disc_delay = reactive_discover(src, dst, r)
        metrics['reactive']['control_msgs'] += cmsgs
        if path is None:
            # no path exists -> cannot deliver
            continue
        delivered, latency, _ = transmit_along_path(path, include_discovery_delay=disc_delay)
        if delivered:
            metrics['reactive']['delivered'] += 1
            metrics['reactive']['latencies'].append(latency)
    else:
        path, last_used = cached
        # use cached route (no discovery cost)
        delivered, latency, _ = transmit_along_path(path, include_discovery_delay=0.0)
        if delivered:
            metrics['reactive']['delivered'] += 1
            metrics['reactive']['latencies'].append(latency)
        # update cache usage time
        reactive_cache[(src,dst)] = (path, r)

# After simulation: compute metrics
def summarize(name, data):
    sent = data['sent']
    delivered = data['delivered']
    pdr = delivered / sent if sent>0 else 0.0
    avg_latency = np.mean(data['latencies']) if data['latencies'] else float('nan')
    throughput = delivered / SIM_ROUNDS  # delivered per round (relative throughput)
    ctrl = data['control_msgs']
    return {'sent':sent, 'delivered':delivered, 'pdr':pdr, 'avg_latency':avg_latency, 'throughput':throughput, 'control_msgs':ctrl}

pro = summarize('proactive', metrics['proactive'])
rea = summarize('reactive', metrics['reactive'])

print("\nSimulation summary (NUM_NODES={}, RADIO_RANGE={}, LINK_RELIABILITY={})".format(NUM_NODES, RADIO_RANGE, LINK_RELIABILITY))
print("Proactive:", pro)
print("Reactive :", rea)

# Simple visual comparison
labels = ['PDR', 'Avg Latency (s)', 'Throughput (per round)', 'Control Msgs']
pro_values = [pro['pdr'], pro['avg_latency'], pro['throughput'], pro['control_msgs']]
rea_values = [rea['pdr'], rea['avg_latency'], rea['throughput'], rea['control_msgs']]

x = np.arange(len(labels))
width = 0.35

fig, ax = plt.subplots(figsize=(9,4))
ax.bar(x - width/2, pro_values, width, label='Proactive')
ax.bar(x + width/2, rea_values, width, label='Reactive')
ax.set_xticks(x)
ax.set_xticklabels(labels, rotation=20)
ax.legend()
plt.title("Proactive vs Reactive (simple model)")
plt.tight_layout()
plt.show()
```

Key observations during performance

* Proactive protocol shows minimal delay since the route is precomputed.
* Reactive protocol shows higher latency due to route discovery overhead.
* Routes may be the same, but performance differs based on control operations.
* PDR and throughput vary depending on the number of route discoveries required.

Conclusion
Proactive routing offers low latency at the cost of high overhead, whereas reactive routing reduces overhead but increases delay due to on-demand route discovery. Routing protocol selection must balance energy, latency, and network conditions for optimal performance in WSNs.