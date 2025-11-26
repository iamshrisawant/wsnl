**Experiment No. 7: Data Aggregation Techniques**

**Theory for Viva**

Purpose

* To implement and compare different data aggregation methods used to reduce redundant data transmission in wireless sensor networks.
* To minimize energy consumption and improve efficiency by combining data before forwarding it.

Meaning of data aggregation

* Process of collecting and combining data from multiple sensor nodes.
* Reduces number of transmitted packets and lowers communication overhead.
* Extends network lifetime since transmission is the most energy-consuming task.

Common aggregation techniques

* Average aggregation

  * Computes mean of sensor readings.
  * Suitable for temperature or humidity monitoring.
* Minimum/Maximum aggregation

  * Selects the lowest or highest value.
  * Used for threshold-based event detection.
* Sum aggregation

  * Adds sensor readings; useful for counting applications.
* Redundancy elimination

  * Removes repeated values from nearby nodes sensing similar environment.
* Weighted fusion

  * Assigns weights to nodes based on reliability or location.

Advantages

* Reduces energy consumption by lowering transmission load.
* Minimizes congestion and collision in the network.
* Improves scalability for large WSN deployments.

Limitations

* Aggregation may introduce loss of accuracy.
* Requires trust in aggregator nodes; vulnerable to faults.
* Delay increases due to data processing.

Summary points for viva

* Aggregation reduces network traffic and extends network lifetime.
* Different techniques suit different sensing applications.
* Aggregation trades accuracy for efficiency.
* Cluster heads or intermediate nodes usually perform aggregation.

---

**Implementation for Performance**

Objective
To simulate data aggregation methods such as average, minimum, maximum, and redundancy removal using synthetic sensor readings.

Requirements

* Python or MATLAB for computation and visualization.
* A dataset of simulated readings from multiple nodes.

Procedure (conceptual steps)

1. Generate sensor readings for a group of nodes.
2. Apply multiple aggregation techniques:

   * Average, min, max, sum, redundancy elimination.
3. Count the number of packets before and after aggregation.
4. Evaluate reduction in communication overhead.
5. Interpret the effectiveness of each technique.
6. Present aggregated outputs and energy savings.

Python implementation (fast demonstration)

```
import numpy as np

readings = np.random.randint(20, 40, 20)

avg = np.mean(readings)
mn = np.min(readings)
mx = np.max(readings)
summ = np.sum(readings)

unique_values = np.unique(readings)

print("Original Readings:", readings)
print("Average:", avg)
print("Minimum:", mn)
print("Maximum:", mx)
print("Sum:", summ)
print("After Redundancy Removal:", unique_values)
print("Reduction in packets:", len(readings) - len(unique_values))
```

Alternate implementation

```
# Experiment 8: Localization in WSN (Colab-ready)
# Demonstrates: centroid method and RSSI-based multilateration (linearized least-squares)
# Author: ChatGPT
# Run in one cell. Requires: numpy, matplotlib

import numpy as np
import matplotlib.pyplot as plt

np.random.seed(42)

# --- Simulation parameters ---
area_size = 100                 # area: 0..area_size (meters)
num_anchors = 5                 # number of anchor nodes (known positions)
num_unknown = 10                # number of nodes to localize
P_tx_dbm = 0.0                  # transmit power at reference distance (d0)
d0 = 1.0                        # reference distance (m)
path_loss_exp = 3.0             # realistic indoor/outdoor exponent (2-4)
rss_noise_std_db = 2.5          # RSSI measurement noise (dB)
min_anchor_dist = 5.0           # avoid placing anchors too close to unknown nodes

# --- Helper functions ---
def rssi_from_dist(d, P0_dbm=P_tx_dbm, n=path_loss_exp, d0=d0):
    """Log-distance path loss model: returns RSSI (dBm) at distance d."""
    # Avoid log(0)
    d = np.maximum(d, 1e-3)
    return P0_dbm - 10*n*np.log10(d / d0)

def dist_from_rssi(rssi_dbm, P0_dbm=P_tx_dbm, n=path_loss_exp, d0=d0):
    """Invert log-distance model: estimate distance from RSSI (dBm)."""
    return d0 * 10 ** ((P0_dbm - rssi_dbm) / (10 * n))

def multilateration_least_squares(anchor_pos, ranges):
    """
    Linearized multilateration using least squares.
    anchor_pos: array shape (m,2)
    ranges: array shape (m,) estimated distances
    Returns estimated (x,y).
    Uses first anchor as reference to form linear system.
    """
    A = []
    b = []
    x1, y1 = anchor_pos[0]
    r1 = ranges[0]
    for i in range(1, len(anchor_pos)):
        xi, yi = anchor_pos[i]
        ri = ranges[i]
        A.append([2*(xi - x1), 2*(yi - y1)])
        b.append((ri**2 - r1**2) - (xi**2 + yi**2) + (x1**2 + y1**2))
    A = np.array(A)
    b = np.array(b)
    # Solve A * [x,y]^T = b in least squares sense
    sol, *_ = np.linalg.lstsq(A, b, rcond=None)
    return sol[0], sol[1]

# --- Generate anchors and unknown nodes ---
# Place anchors near edges for realistic geometry
anchors = np.column_stack((
    np.random.uniform(5, area_size-5, num_anchors),
    np.random.uniform(5, area_size-5, num_anchors)
))

unknowns = np.column_stack((
    np.random.uniform(0, area_size, num_unknown),
    np.random.uniform(0, area_size, num_unknown)
))

# Ensure anchors are not too close to unknowns (minor relocation if needed)
for i,u in enumerate(unknowns):
    dists = np.linalg.norm(anchors - u, axis=1)
    if np.any(dists < min_anchor_dist):
        # shift the anchor slightly away
        idx = np.argmin(dists)
        anchors[idx] += np.sign(anchors[idx] - u) * (min_anchor_dist - dists[idx] + 1.0)

# --- Simulate RSSI measurements with noise and estimate distances ---
true_positions = unknowns.copy()
est_positions_centroid = []
est_positions_multi = []
errors_centroid = []
errors_multi = []

for u in true_positions:
    # true distances from each anchor
    dists = np.linalg.norm(anchors - u, axis=1)

    # measured RSSI with gaussian noise (dB)
    rssi_meas = rssi_from_dist(dists) + np.random.normal(0, rss_noise_std_db, size=dists.shape)

    # estimate distances from noisy RSSI
    est_dists = dist_from_rssi(rssi_meas)
    
    # centroid method: simple average of anchors that "hear" it (all anchors here)
    centroid = np.mean(anchors, axis=0)
    est_positions_centroid.append(centroid)
    errors_centroid.append(np.linalg.norm(centroid - u))
    
    # multilateration using linearized least squares
    try:
        x_est, y_est = multilateration_least_squares(anchors, est_dists)
    except Exception as e:
        # fallback: centroid if solver fails
        x_est, y_est = centroid
    est_positions_multi.append([x_est, y_est])
    errors_multi.append(np.linalg.norm([x_est, y_est] - u))

est_positions_centroid = np.array(est_positions_centroid)
est_positions_multi = np.array(est_positions_multi)

# --- Metrics ---
rmse_centroid = np.sqrt(np.mean(np.array(errors_centroid)**2))
rmse_multi = np.sqrt(np.mean(np.array(errors_multi)**2))

print(f"Localization results (area {area_size}m x {area_size}m)")
print(f"Anchors: {num_anchors}, Unknowns: {num_unknown}")
print(f"Path-loss exponent: {path_loss_exp}, RSS noise std (dB): {rss_noise_std_db}")
print(f"Centroid RMSE: {rmse_centroid:.2f} m")
print(f"Multilateration RMSE: {rmse_multi:.2f} m")

# --- Plot results ---
plt.figure(figsize=(8,8))
plt.scatter(anchors[:,0], anchors[:,1], marker='^', s=120, label='Anchors (known)', edgecolor='k')
plt.scatter(true_positions[:,0], true_positions[:,1], marker='o', s=60, label='True positions', c='g')
plt.scatter(est_positions_centroid[:,0], est_positions_centroid[:,1], marker='x', s=60, label='Centroid est', c='orange')
plt.scatter(est_positions_multi[:,0], est_positions_multi[:,1], marker='s', s=60, label='Multilateration est', c='red')

# draw lines from true to estimated (multilateration)
for tp, me in zip(true_positions, est_positions_multi):
    plt.plot([tp[0], me[0]], [tp[1], me[1]], 'r--', linewidth=0.7, alpha=0.7)

plt.xlim(-5, area_size+5)
plt.ylim(-5, area_size+5)
plt.legend()
plt.title('WSN Localization: Anchors, True positions, Estimates')
plt.xlabel('X (m)')
plt.ylabel('Y (m)')
plt.grid(True)
plt.show()

# --- Per-node printout (first few) ---
print("\nSample per-node errors (first 6 nodes):")
for i in range(min(6, num_unknown)):
    print(f"Node {i+1}: True({true_positions[i,0]:.1f},{true_positions[i,1]:.1f}) "
          f"CentroidErr={errors_centroid[i]:.2f}m  MultiErr={errors_multi[i]:.2f}m")
```

Key observations during performance

* Redundancy elimination significantly reduces the number of packets to be transmitted.
* Simple aggregation techniques like average and sum yield meaningful summaries of the sensed environment.
* Aggregation reduces energy consumption by lowering communication load.
* Accuracy decreases slightly but efficiency increases greatly.

Conclusion
Data aggregation effectively minimizes redundant transmissions and enhances energy efficiency in wireless sensor networks. Techniques such as averaging, minimum/maximum extraction, and redundancy elimination provide compact and meaningful representations of sensor data, leading to longer network lifetime.