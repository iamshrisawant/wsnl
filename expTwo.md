**Experiment No. 2: Energy Consumption Analysis**

**Theory for Viva**

Purpose

* To study how different MAC protocols influence energy consumption in wireless sensor nodes.
* Energy is a critical resource in WSNs because nodes are battery-powered and often deployed in inaccessible environments.

Need for energy-efficient MAC protocols

* Communication (TX/RX) consumes more energy than sensing or processing.
* MAC protocols reduce unnecessary listening, collisions, idle time, and retransmissions.
* Energy efficiency directly increases network lifetime.

Protocols involved

* S-MAC

  * Uses periodic sleep and listen cycles (duty cycling).
  * Nodes sleep for a fixed interval to save energy.
  * Reduces idle listening but increases latency.
* IEEE 802.15.4

  * Standard low-power MAC for WSNs.
  * Uses CSMA-CA and optional beacon-enabled mode.
  * Provides better throughput but may use more energy under heavy load.

Energy model used for analysis

* Transmission power consumption (Ptx)
* Reception power consumption (Prx)
* Idle state power consumption (Pidle)
* Sleep state power consumption (Psleep)

Common observations

* S-MAC consumes less overall energy because nodes spend most time sleeping.
* IEEE 802.15.4 performs better in throughput but wastes more energy when the channel is busy.
* Energy consumption increases with higher traffic load or lower duty cycle.

Summary points for viva

* S-MAC is an energy-saving protocol due to periodic sleep scheduling.
* IEEE 802.15.4 provides better network performance at the cost of more energy.
* Energy consumption depends on duty cycle, traffic rate, and collision probability.
* Effective MAC protocol selection increases the network lifetime significantly.

---

**Implementation for Performance**

Objective
To simulate and analyze the energy consumption of sensor nodes under two MAC protocols: S-MAC and IEEE 802.15.4.

Requirements

* Python with NumPy and Matplotlib or MATLAB.
* Simple energy model (TX, RX, Idle, Sleep power levels).
* Traffic generation logic (packet send/receive events).

Procedure (conceptual steps)

1. Define energy parameters for each protocol (TX, RX, Idle, Sleep).
2. Define traffic load and time duration for simulation.
3. For each time slot:

   * Evaluate node state (TX, RX, Idle, Sleep).
   * Deduct corresponding energy from total energy.
4. Run simulation separately for:

   * S-MAC with duty cycling
   * IEEE 802.15.4 without duty cycling
5. Plot energy consumption vs time.
6. Compare results and interpret protocol efficiency.

Python implementation (fast demonstration)

```
import numpy as np
import matplotlib.pyplot as plt

time = 1000
energy_smac = 100
energy_ieee = 100

P_tx = 0.02
P_rx = 0.01
P_idle = 0.005
P_sleep = 0.001

duty_cycle = 0.3
traffic = np.random.rand(time)

E_smac = []
E_ieee = []

for t in range(time):
    if traffic[t] > 0.7:
        energy_smac -= P_tx
        energy_ieee -= P_tx
    elif traffic[t] > 0.4:
        energy_smac -= P_rx
        energy_ieee -= P_rx
    else:
        if t % int(1/duty_cycle) == 0:
            energy_smac -= P_sleep
        else:
            energy_smac -= P_idle
        energy_ieee -= P_idle

    E_smac.append(energy_smac)
    E_ieee.append(energy_ieee)

plt.plot(E_smac, label="S-MAC")
plt.plot(E_ieee, label="IEEE 802.15.4")
plt.legend()
plt.title("Energy Consumption Comparison")
plt.xlabel("Time")
plt.ylabel("Remaining Energy")
plt.show()
```

Alternate implementation

```
# Colab-ready simulation: Energy comparison S-MAC vs IEEE-style
# Install networkx if missing: uncomment the following line in Colab
# !pip install networkx

import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
from collections import deque, defaultdict
import random

# ---------- Simulation parameters ----------
SEED = 42
np.random.seed(SEED)
random.seed(SEED)

N = 40                    # number of sensor nodes
area = 100                # square area (0..area)
comm_range = 25.0         # communication range (units)
initial_energy = 2.0      # energy units per node
E_tx_per_unit = 0.06      # energy cost per TX (base)
E_rx_per_unit = 0.03      # energy cost per RX (base)
E_idle = 0.005            # per-slot idle listening cost
E_sleep = 0.0005          # per-slot sleep cost
slot_time = 1.0           # abstract time per slot
p_gen = 0.02              # per node packet generation probability per slot
max_slots = 1200          # simulation duration in slots

duty_cycle = 0.25         # for S-MAC (awake fraction)
# ------------------------------------------------

# create random node positions, sink at center
positions = {i: (np.random.uniform(0, area), np.random.uniform(0, area)) for i in range(N)}
# add sink as node N at center
sink_id = N
positions[sink_id] = (area/2, area/2)

# build connectivity graph (undirected) based on comm_range
G = nx.Graph()
G.add_nodes_from(list(positions.keys()))
for i in positions:
    for j in positions:
        if i >= j: continue
        xi, yi = positions[i]; xj, yj = positions[j]
        if np.hypot(xi-xj, yi-yj) <= comm_range:
            G.add_edge(i, j)

# precompute next-hop (shortest path) toward sink for each node
shortest_paths = {}
for i in range(N):
    try:
        path = nx.shortest_path(G, i, sink_id)
        # path[0] is i, path[1] is next hop (if direct path exists)
        shortest_paths[i] = path
    except (nx.NetworkXNoPath, KeyError):
        shortest_paths[i] = None  # disconnected

def run_simulation(mac_type='SMAC'):
    # mac_type: 'SMAC' or 'IEEE'
    energy = np.ones(N) * initial_energy
    node_alive = np.ones(N, dtype=bool)
    # duty schedule for S-MAC: simple synchronous-ish pattern (each node cycles)
    # To keep it simple yet realistic, assign each node a random phase in duty cycle
    phases = np.random.randint(0, int(1/duty_cycle)) if duty_cycle>0 else np.zeros(N, dtype=int)
    node_phase = {i: np.random.randint(0, int(1/duty_cycle)) if duty_cycle>0 else 0 for i in range(N)}
    # queues for packets at each node (packets are dicts with id for stats)
    queues = {i: deque() for i in range(N)}
    delivered = 0
    generated = 0
    delivered_history = []
    avg_energy_history = []
    first_dead_slot = None

    # unique packet id counter
    pkt_counter = 0

    for slot in range(max_slots):
        # 1) new packet generation
        for i in range(N):
            if not node_alive[i]:
                continue
            if np.random.rand() < p_gen:
                # create packet with route path
                if shortest_paths[i] is None:
                    # disconnected -> cannot be delivered; still counts as generated
                    generated += 1
                    continue
                pkt = {'id': pkt_counter, 'path': list(shortest_paths[i]), 'cur_idx': 0}
                pkt_counter += 1
                queues[i].append(pkt)
                generated += 1

        # 2) determine awake/sleep state this slot
        awake = np.ones(N, dtype=bool)
        if mac_type == 'SMAC':
            cycle_len = int(1/duty_cycle)
            for i in range(N):
                awake[i] = ((slot + node_phase[i]) % cycle_len) < max(1, int(cycle_len * duty_cycle))
        elif mac_type == 'IEEE':
            awake[:] = True

        # 3) collect transmission attempts: each node with queue and awake attempts to transmit one packet to next hop
        tx_attempts = defaultdict(list)  # receiver -> list of (sender, pkt)
        for i in range(N):
            if not node_alive[i] or not awake[i]:
                # consume sleep or dead energy
                energy[i] -= E_sleep if mac_type == 'SMAC' else E_idle*0.0
                continue
            # node awake: idle listening cost (unless transmitting)
            if queues[i]:
                pkt = queues[i][0]  # peek
                path = pkt['path']
                cur = pkt['cur_idx']
                if cur + 1 < len(path):
                    next_hop = path[cur+1]
                    # if next_hop is sink, treat as receiver N (sink is awake)
                    tx_attempts[next_hop].append((i, pkt))
                else:
                    # already at sink (shouldn't happen), mark delivered
                    queues[i].popleft()
                    delivered += 1
            else:
                energy[i] -= E_idle  # idle listening cost when awake and no TX/RX

        # 4) resolve transmissions per receiver (including sink)
        successes = []
        for recv, senders in tx_attempts.items():
            # if receiver is sink (sink_id), it's always awake and has infinite energy (abstract)
            if len(senders) == 1:
                sender, pkt = senders[0]
                # successful transmission
                successes.append((sender, recv, pkt))
            else:
                # collision: all packets lost at this hop (simple model)
                # they still consume TX energy
                for sender, pkt in senders:
                    # consume TX energy even on collision
                    if node_alive[sender]:
                        energy[sender] -= E_tx_per_unit
                # no receiver energy consumed because collided
                continue

        # 5) apply successful transmissions: TX energy at sender, RX at receiver; advance pkt to next node
        for sender, recv, pkt in successes:
            if sender >= N or not node_alive[sender]:
                continue
            # if receiver is sink -> delivered
            if recv == sink_id:
                energy[sender] -= E_tx_per_unit
                # deliver and remove packet from sender queue
                if queues[sender] and queues[sender][0]['id'] == pkt['id']:
                    queues[sender].popleft()
                    delivered += 1
            else:
                # ensure receiver is awake and alive to receive
                if recv < N and node_alive[recv]:
                    energy[sender] -= E_tx_per_unit
                    energy[recv] -= E_rx_per_unit
                    # move packet along: remove from sender queue and append to receiver queue with cur_idx+1
                    if queues[sender] and queues[sender][0]['id'] == pkt['id']:
                        queues[sender].popleft()
                        new_pkt = {'id': pkt['id'], 'path': pkt['path'], 'cur_idx': pkt['cur_idx'] + 1}
                        queues[recv].append(new_pkt)
                else:
                    # receiver dead or nonexistent: tx consumed energy but packet lost
                    energy[sender] -= E_tx_per_unit
                    if queues[sender] and queues[sender][0]['id'] == pkt['id']:
                        queues[sender].popleft()

        # 6) check node energy and mark dead if <=0
        for i in range(N):
            if node_alive[i] and energy[i] <= 0:
                node_alive[i] = False
                if first_dead_slot is None:
                    first_dead_slot = slot

        # 7) record statistics
        avg_energy_history.append(np.mean(np.clip(energy, 0, None)))
        delivered_history.append(delivered / generated if generated>0 else 0.0)

        # early stop if all nodes dead
        if not node_alive.any():
            break

    # wrap up metrics
    metrics = {
        'avg_energy_history': np.array(avg_energy_history),
        'delivered_history': np.array(delivered_history),
        'generated': generated,
        'delivered': delivered,
        'pdr': delivered / generated if generated>0 else 0.0,
        'first_dead_slot': first_dead_slot if first_dead_slot is not None else max_slots,
        'slots_simulated': slot+1
    }
    return metrics, energy, node_alive

# run both MACs
metrics_smac, energy_smac, alive_smac = run_simulation('SMAC')
metrics_ieee, energy_ieee, alive_ieee = run_simulation('IEEE')

# plotting results
plt.figure(figsize=(12,5))
plt.subplot(1,2,1)
plt.plot(metrics_smac['avg_energy_history'], label='S-MAC (duty {:.0%})'.format(duty_cycle))
plt.plot(metrics_ieee['avg_energy_history'], label='IEEE (always-on)')
plt.xlabel('Slot')
plt.ylabel('Avg remaining energy (per node)')
plt.legend()
plt.title('Average Energy vs Time')

plt.subplot(1,2,2)
# cumulative PDR over time (delivered_history)
plt.plot(metrics_smac['delivered_history'], label='S-MAC PDR')
plt.plot(metrics_ieee['delivered_history'], label='IEEE PDR')
plt.xlabel('Slot')
plt.ylabel('Cumulative Packet Delivery Ratio (delivered/generated)')
plt.legend()
plt.title('Delivery Ratio vs Time')
plt.tight_layout()
plt.show()

# textual summary
print("S-MAC: generated {}, delivered {}, PDR {:.3f}, first_dead_slot {}"
      .format(metrics_smac['generated'], metrics_smac['delivered'], metrics_smac['pdr'], metrics_smac['first_dead_slot']))
print("IEEE : generated {}, delivered {}, PDR {:.3f}, first_dead_slot {}"
      .format(metrics_ieee['generated'], metrics_ieee['delivered'], metrics_ieee['pdr'], metrics_ieee['first_dead_slot']))
```

Key observations during performance

* S-MAC energy decreases slowly due to sleep periods.
* IEEE 802.15.4 energy drops faster due to continuous idle listening.
* Higher traffic increases energy usage in both protocols.
* Duty cycle strongly impacts S-MAC efficiency.

Conclusion
S-MAC is more energy-efficient due to duty cycling, while IEEE 802.15.4 provides better throughput at the cost of higher energy consumption. Energy-saving MAC protocols significantly extend the lifetime of wireless sensor networks.