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

MATLAB implementation (alternative)

```
time = 1000;
energy_smac = 100;
energy_ieee = 100;

P_tx = 0.02;
P_rx = 0.01;
P_idle = 0.005;
P_sleep = 0.001;

duty_cycle = 0.3;
traffic = rand(1,time);

E_smac = zeros(1,time);
E_ieee = zeros(1,time);

for t = 1:time
    if traffic(t) > 0.7
        energy_smac = energy_smac - P_tx;
        energy_ieee = energy_ieee - P_tx;
    elseif traffic(t) > 0.4
        energy_smac = energy_smac - P_rx;
        energy_ieee = energy_ieee - P_rx;
    else
        if mod(t, round(1/duty_cycle)) == 0
            energy_smac = energy_smac - P_sleep;
        else
            energy_smac = energy_smac - P_idle;
        end
        energy_ieee = energy_ieee - P_idle;
    end
    E_smac(t) = energy_smac;
    E_ieee(t) = energy_ieee;
end

plot(E_smac); hold on;
plot(E_ieee);
legend('S-MAC','IEEE 802.15.4');
title('Energy Consumption Comparison');
xlabel('Time');
ylabel('Remaining Energy');
```

Key observations during performance

* S-MAC energy decreases slowly due to sleep periods.
* IEEE 802.15.4 energy drops faster due to continuous idle listening.
* Higher traffic increases energy usage in both protocols.
* Duty cycle strongly impacts S-MAC efficiency.

Conclusion
S-MAC is more energy-efficient due to duty cycling, while IEEE 802.15.4 provides better throughput at the cost of higher energy consumption. Energy-saving MAC protocols significantly extend the lifetime of wireless sensor networks.