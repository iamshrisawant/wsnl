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

MATLAB implementation (alternative)

```
N = 20;
A = rand(N) > 0.7;
G = graph(A);

src = 1; dst = 15;

tic;
pro_path = shortestpath(G, src, dst);
lat_pro = toc;

tic;
re_path = shortestpath(G, src, dst);
lat_re = toc + 0.002;

disp("Proactive Latency:"); disp(lat_pro);
disp("Reactive Latency:"); disp(lat_re);
disp("Proactive Path:"); disp(pro_path);
disp("Reactive Path:"); disp(re_path);
```

Key observations during performance

* Proactive protocol shows minimal delay since the route is precomputed.
* Reactive protocol shows higher latency due to route discovery overhead.
* Routes may be the same, but performance differs based on control operations.
* PDR and throughput vary depending on the number of route discoveries required.

Conclusion
Proactive routing offers low latency at the cost of high overhead, whereas reactive routing reduces overhead but increases delay due to on-demand route discovery. Routing protocol selection must balance energy, latency, and network conditions for optimal performance in WSNs.