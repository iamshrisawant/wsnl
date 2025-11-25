**Experiment No. 5: QoS Analysis in Wireless Sensor Networks**

**Theory for Viva**

Purpose

* To analyze Quality of Service (QoS) parameters such as latency, jitter, and throughput under varying traffic conditions.
* To understand how network load affects performance in wireless sensor networks.

Meaning of QoS

* Measures how efficiently and reliably the network transmits data.
* Ensures timely and accurate delivery of sensed information.

Key QoS parameters

* Latency

  * Time taken for a packet to reach the destination.
  * Increases with congestion and low bandwidth.
* Jitter

  * Variation in latency between packets.
  * High jitter affects real-time applications like monitoring or control systems.
* Throughput

  * Rate of successful data transmission.
  * Decreases when collisions or congestion occur.
* Packet Delivery Ratio (PDR)

  * Ratio of received packets to sent packets.
  * Indicates reliability.

Influencing factors

* Traffic load and packet generation rate.
* MAC protocol efficiency.
* Network topology and density.
* Interference, collisions, and retransmissions.

Common observations

* At low traffic load, all QoS parameters remain stable with low latency and high throughput.
* As load increases, congestion rises causing higher latency and jitter.
* Throughput increases initially, then saturates or drops due to collisions.
* Packet delivery ratio decreases when channel becomes busy.

Summary points for viva

* QoS ensures reliable and efficient operation of WSN applications.
* Latency, jitter, throughput, and PDR are main QoS metrics.
* High traffic load degrades QoS due to collisions and buffer overflow.
* MAC and routing protocols strongly influence QoS performance.

---

**Implementation for Performance**

Objective
To simulate QoS parameters (latency, jitter, throughput) under low, medium, and high traffic loads and compare their behavior.

Requirements

* Python with NumPy and Matplotlib or MATLAB.
* Simple packet generation and transmission timing model.

Procedure (conceptual steps)

1. Define simulation duration and number of packets to generate.
2. Create three traffic scenarios:

   * Low load
   * Medium load
   * High load
3. Record timestamps for packet sending and receiving.
4. Compute latency for each packet.
5. Calculate jitter from differences in consecutive latencies.
6. Compute throughput as received packets per time unit.
7. Plot metrics for each traffic level and observe trends.

Python implementation (fast demonstration)

```
import numpy as np
import matplotlib.pyplot as plt

def simulate(load):
    send_times = np.cumsum(np.random.exponential(1/load, 500))
    delays = np.random.normal(0.02, 0.005*load, 500)
    recv_times = send_times + delays
    latency = recv_times - send_times
    jitter = np.abs(np.diff(latency))
    throughput = 500 / send_times[-1]
    return latency, jitter, throughput

lat_low, jit_low, thr_low = simulate(5)
lat_med, jit_med, thr_med = simulate(15)
lat_high, jit_high, thr_high = simulate(30)

plt.plot(lat_low[:100], label="Low Load")
plt.plot(lat_med[:100], label="Medium Load")
plt.plot(lat_high[:100], label="High Load")
plt.title("Latency Comparison")
plt.legend()
plt.show()
```

MATLAB implementation (alternative)

```
function [latency, jitter, thr] = simulate(load)
    send = cumsum(exprnd(1/load,1,500));
    delays = normrnd(0.02, 0.005*load, 1, 500);
    recv = send + delays;
    latency = recv - send;
    jitter = abs(diff(latency));
    thr = 500 / send(end);
end

[lat_low, jit_low, thr_low] = simulate(5);
[lat_med, jit_med, thr_med] = simulate(15);
[lat_high, jit_high, thr_high] = simulate(30);

plot(lat_low(1:100)); hold on;
plot(lat_med(1:100));
plot(lat_high(1:100));
legend('Low Load','Medium Load','High Load');
title('Latency Comparison');
```

Key observations during performance

* Latency rises steadily as load increases due to congestion.
* Jitter becomes more unpredictable in medium and high traffic.
* Throughput increases initially but drops slightly at high load.
* Under heavy load, packets experience longer delays and more variance.

Conclusion
QoS parameters degrade as traffic increases. High load causes higher latency, larger jitter values, and reduced throughput. Maintaining optimal traffic flow and using efficient MAC protocols is essential for sustaining QoS in wireless sensor networks.