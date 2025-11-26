**Experiment No. 7: Data Aggregation Techniques**

**Theory for Viva**

Purpose

* To implement and compare different data aggregation techniques that reduce redundant data transmission in a wireless sensor network.
* To improve energy efficiency and bandwidth utilization by minimizing unnecessary communication.

Meaning of data aggregation

* Process of combining sensor readings to reduce the volume of transmitted data.
* Helps lower energy consumption because communication is the most expensive operation in WSNs.

Common aggregation techniques

* Averaging

  * Computes the mean of sensor readings.
  * Useful when individual values are not required.
* Summation

  * Adds all readings, used for total or cumulative measurements.
* Minimum and Maximum

  * Extracts extreme values from sensor data.
  * Used for threshold-based monitoring and alerts.
* Data fusion

  * Uses statistical rules or machine learning to combine multiple readings.
* Compression aggregation

  * Removes redundancy by encoding or compressing data.

Advantages

* Reduces communication energy drastically.
* Minimizes congestion and collisions in the network.
* Extends network lifetime by lowering load on nodes.
* Improves scalability in dense networks.

Challenges

* Aggregation may cause loss of raw data accuracy.
* Wrong aggregation may eliminate outliers or important anomalies.
* Requires synchronization or cluster-based architectures.

Common observations

* Aggregation reduces the number of packets transmitted.
* Cluster-based aggregation improves efficiency since cluster heads perform the computation.
* Energy saved increases with network density.

Summary points for viva

* Data aggregation reduces redundancy and increases energy efficiency.
* Average, sum, min–max, and fusion are common methods.
* Aggregation reduces communication but may reduce precision.
* Cluster heads typically perform the aggregation task.

---

**Implementation for Performance**

Objective
To simulate multiple sensor readings, apply different aggregation techniques, and compare the reduction in transmitted data.

Requirements

* Python or MATLAB for data processing and visualization.
* A list of simulated sensor readings representing multiple nodes.

Procedure (conceptual steps)

1. Generate a set of sensor readings (temperature, humidity, or random values).
2. Apply aggregation methods:

   * Average
   * Sum
   * Min and Max
   * Compression or fusion (optional)
3. Count the number of original data points and number of aggregated outputs.
4. Calculate percentage reduction in transmitted data.
5. Compare aggregation method results for efficiency and accuracy.
6. Display output numerically or through a small plot.

Python implementation (fast demonstration)

```
import numpy as np

data = np.random.randint(20, 40, 50)

avg_val = np.mean(data)
sum_val = np.sum(data)
min_val = np.min(data)
max_val = np.max(data)

print("Original Data Count:", len(data))
print("Transmitted After Aggregation: 1")

print("Average:", avg_val)
print("Sum:", sum_val)
print("Min:", min_val)
print("Max:", max_val)

reduction = (1 / len(data)) * 100
print("Data Reduction (%):", reduction)
```

Alternate Implementation
```
# Experiment 7: Data Aggregation Techniques (Colab-ready)
# Demonstrates cluster-based aggregation vs raw forwarding.
# Author: ChatGPT (adapted for exam/demo use)

import numpy as np
import matplotlib.pyplot as plt

np.random.seed(42)

# PARAMETERS
AREA = 100                 # square area (0..AREA)
N = 100                    # number of sensors
Kx, Ky = 4, 4              # grid clusters (Kx * Ky clusters)
CH_PER_CLUSTER = 1         # cluster head count (simple: 1 per cell)
SINK = np.array([AREA/2, AREA+20])  # sink placed outside area (to simulate real-world)
E_TX_BASE = 50e-6          # energy per packet (Joule) for electronics (base)
E_TX_DIST = 1e-6           # additional energy factor per m^2
E_RX = 50e-6               # energy to receive one packet (Joule)
E_PROC = 5e-6              # energy to aggregate/process one packet (Joule)
PKT_SIZE = 1               # normalized packet size (unit)

# 1. Deploy nodes randomly (or use poisson - keep random for realism)
nodes_xy = AREA * np.random.rand(N, 2)

# 2. Create a spatially correlated field (smooth gaussian field)
#    Base temperature field: linear gradient + smooth noise
x = np.linspace(0, AREA, 50)
y = np.linspace(0, AREA, 50)
X, Y = np.meshgrid(x, y)
base_field = 20 + (X + Y) * 0.05  # slight gradient
# Add smooth noise by convolving white noise (approx) - emulate with low-freq components
noise = 2 * np.sin(0.1 * X) * np.cos(0.12 * Y)
field = base_field + noise

# sample field values at node positions (bilinear interpolation approximation)
def sample_field(xy):
    x_idx = np.clip((xy[:,0] / AREA * (x.size-1)).astype(int), 0, x.size-2)
    y_idx = np.clip((xy[:,1] / AREA * (y.size-1)).astype(int), 0, y.size-2)
    return field[y_idx, x_idx] + np.random.normal(0, 0.5, len(x_idx))  # measurement noise

readings = sample_field(nodes_xy)

# 3. Partition nodes into grid clusters (Kx x Ky)
def assign_cluster(xy, Kx, Ky, area):
    # integer coords 0..Kx-1, 0..Ky-1
    ix = np.minimum((xy[:,0] / area * Kx).astype(int), Kx-1)
    iy = np.minimum((xy[:,1] / area * Ky).astype(int), Ky-1)
    return ix + iy * Kx

clusters = assign_cluster(nodes_xy, Kx, Ky, AREA)
num_clusters = Kx * Ky

# pick cluster heads: choose node nearest to cluster centroid
cluster_heads = np.full(num_clusters, -1, dtype=int)
for c in range(num_clusters):
    members = np.where(clusters == c)[0]
    if members.size == 0:
        continue
    # cluster centroid
    cx = ( (c % Kx) + 0.5 ) * (AREA / Kx)
    cy = ( (c // Kx) + 0.5 ) * (AREA / Ky)
    dists = np.linalg.norm(nodes_xy[members] - np.array([cx, cy]), axis=1)
    cluster_heads[c] = members[np.argmin(dists)]

# ensure cluster_heads unique and valid
cluster_heads = cluster_heads[cluster_heads >= 0]

# 4. Utility functions for energy and packet counting
def tx_energy(distance_m, pkt_size=PKT_SIZE):
    # simple energy model: E = E_TX_BASE * pkt_size + E_TX_DIST * distance^2 * pkt_size
    return (E_TX_BASE * pkt_size) + (E_TX_DIST * (distance_m**2) * pkt_size)

def rx_energy(pkt_size=PKT_SIZE):
    return E_RX * pkt_size

# 5. Scenario A: Raw forwarding (every node -> sink)
packets_raw = N
total_E_raw = 0.0
for i in range(N):
    d = np.linalg.norm(nodes_xy[i] - SINK)
    total_E_raw += tx_energy(d)
# no Rx counted at sink for energy accounting (sink is mains-powered usually)

# 6. Scenario B: Cluster aggregation (node -> CH; CH aggregates; CH -> sink)
packets_to_CH = 0
packets_CH_to_sink = 0
total_E_agg = 0.0

for c in range(num_clusters):
    members = np.where(clusters == c)[0]
    if members.size == 0:
        continue
    ch = cluster_heads[c]
    # sensors send to CH
    for m in members:
        if m == ch:
            continue
        d = np.linalg.norm(nodes_xy[m] - nodes_xy[ch])
        total_E_agg += tx_energy(d)         # node tx
        total_E_agg += rx_energy()         # CH rx
        packets_to_CH += 1
    # CH processes/aggregates data
    # cost: process N_members packets into 1 (processing cost)
    total_E_agg += E_PROC * members.size
    # CH sends 1 aggregated packet to sink
    d_ch_sink = np.linalg.norm(nodes_xy[ch] - SINK)
    total_E_agg += tx_energy(d_ch_sink)
    packets_CH_to_sink += 1

# 7. Compute aggregation accuracy for mean: compare true mean vs aggregated mean
true_mean = readings.mean()
# aggregated mean reconstructed at sink: weighted by cluster sizes using cluster-head averages
agg_values = []
cluster_sizes = []
for c in range(num_clusters):
    members = np.where(clusters == c)[0]
    if members.size == 0: continue
    ch = cluster_heads[c]
    cluster_mean = readings[members].mean()
    agg_values.append(cluster_mean)
    cluster_sizes.append(members.size)
# reconstructed global mean using cluster means and sizes
recon_mean = np.average(agg_values, weights=cluster_sizes)
mean_error = abs(recon_mean - true_mean)

# 8. Print and plot results
print("Experiment 7: Data Aggregation (cluster-based)")
print(f"Number of nodes: {N}")
print(f"Clusters (grid): {Kx} x {Ky} = {num_clusters}")
print(f"Raw forwarding packets (to sink): {packets_raw}")
print(f"Aggregation: packets to CH (intermediate): {packets_to_CH}, CH->sink packets: {packets_CH_to_sink}")
print(f"Total TX energy (raw forwarding): {total_E_raw:.6f} J")
print(f"Total TX+RX+proc energy (aggregation): {total_E_agg:.6f} J")
print(f"Energy saved by aggregation: {100*(1 - total_E_agg/total_E_raw):.2f}%")
print(f"True global mean: {true_mean:.3f}, Reconstructed mean at sink: {recon_mean:.3f}")
print(f"Absolute mean error due to aggregation: {mean_error:.4f}")

# Visualizations
plt.figure(figsize=(14,5))

plt.subplot(1,3,1)
plt.scatter(nodes_xy[:,0], nodes_xy[:,1], c=readings, cmap='coolwarm', s=30, edgecolor='k')
for ch in cluster_heads:
    plt.scatter(nodes_xy[ch,0], nodes_xy[ch,1], marker='*', s=180, c='gold', edgecolor='k')
plt.scatter(SINK[0], SINK[1], marker='D', s=120, c='green', label='Sink', edgecolor='k')
plt.title('Node Locations & Cluster Heads (stars)')
plt.xlabel('X'); plt.ylabel('Y'); plt.colorbar(label='Reading')

plt.subplot(1,3,2)
plt.bar(['Raw packets','Packets to CH','Packets CH->Sink'], [packets_raw, packets_to_CH, packets_CH_to_sink])
plt.title('Packet Counts')

plt.subplot(1,3,3)
plt.bar(['Raw Energy (J)','Aggregated Energy (J)'], [total_E_raw, total_E_agg])
plt.title('Energy Comparison')

plt.tight_layout()
plt.show()
```

Key observations during performance

* A large set of 50 values reduces to a single representative value after aggregation.
* Data transmission reduces by more than 98%.
* Aggregation maintains general information while reducing communication cost.
* Accuracy trade-off depends on the aggregation method used.

Conclusion
Data aggregation techniques significantly reduce redundant information and lower communication energy in wireless sensor networks. While aggregation improves efficiency and extends network lifetime, it may introduce a slight loss of precision compared to raw data transmission.