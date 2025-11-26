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

Alternate implementation
```
# QoS Simulation (M/M/1 queue with finite buffer)
# Simulates three traffic loads and reports latency, jitter, throughput, PDR.
# Run in Colab / local Python (requires numpy, matplotlib)

import numpy as np
import matplotlib.pyplot as plt

np.random.seed(42)

def simulate_qos(arrival_rate, service_rate, buffer_size, sim_time):
    """
    arrival_rate: lambda (packets per second)
    service_rate: mu (packets per second)
    buffer_size: max packets in queue (excluding server). If full, arriving packet is dropped.
    sim_time: total simulated time (seconds)
    Returns: dict with lists: arrival_times, depart_times, latencies, dropped_count
    """
    t = 0.0
    next_arrival = np.random.exponential(1/arrival_rate) if arrival_rate>0 else np.inf
    next_departure = np.inf  # no packet being served initially
    queue = []  # stores arrival times of waiting packets
    in_service_arrival_time = None
    arrivals = 0
    departures = 0
    dropped = 0

    arrival_times = []
    depart_times = []
    latencies = []

    while t < sim_time:
        if next_arrival <= next_departure:
            # process arrival event
            t = next_arrival
            arrivals += 1
            arrival_times.append(t)

            if in_service_arrival_time is None:
                # server idle -> start service immediately
                in_service_arrival_time = t
                service_time = np.random.exponential(1/service_rate)
                next_departure = t + service_time
            else:
                # server busy -> check buffer
                if len(queue) < buffer_size:
                    queue.append(t)
                else:
                    dropped += 1

            # schedule next arrival
            inter = np.random.exponential(1/arrival_rate)
            next_arrival = t + inter
        else:
            # process departure event
            t = next_departure
            departures += 1
            depart_times.append(t)
            latency = t - in_service_arrival_time
            latencies.append(latency)

            # serve next from queue if any
            if queue:
                in_service_arrival_time = queue.pop(0)
                service_time = np.random.exponential(1/service_rate)
                next_departure = t + service_time
            else:
                in_service_arrival_time = None
                next_departure = np.inf

        # safety break if extremely long
        if arrivals > 10_000_000:
            break

    results = {
        "arrivals": arrivals,
        "departures": departures,
        "dropped": dropped,
        "arrival_times": np.array(arrival_times),
        "depart_times": np.array(depart_times),
        "latencies": np.array(latencies),
    }
    return results

def analyze_and_plot(scenarios, sim_time=200, service_rate=5.0, buffer_size=10):
    """
    scenarios: dict{name: arrival_rate}
    sim_time: seconds per scenario
    service_rate: server processing rate (packets/sec)
    buffer_size: queue capacity
    """
    summary = {}
    plt.figure(figsize=(12,8))

    # Latency time-series subplot
    plt.subplot(2,2,1)
    for name, ar in scenarios.items():
        res = simulate_qos(ar, service_rate, buffer_size, sim_time)
        lat = res['latencies']
        times = res['depart_times']
        summary[name] = res

        # plot last 100 latencies vs departure time for visual clarity
        idx = slice(max(0, len(lat)-100), len(lat))
        plt.plot(times[idx], lat[idx], label=f"{name} (λ={ar})")
    plt.xlabel("Departure time (s)")
    plt.ylabel("Latency (s)")
    plt.title("Latency over time (sampled)")
    plt.legend()

    # Latency CDF subplot
    plt.subplot(2,2,2)
    for name in scenarios:
        lat = summary[name]['latencies']
        if len(lat)==0:
            continue
        sorted_lat = np.sort(lat)
        cdf = np.arange(1, len(sorted_lat)+1) / len(sorted_lat)
        plt.plot(sorted_lat, cdf, label=name)
    plt.xlabel("Latency (s)")
    plt.ylabel("CDF")
    plt.title("Latency CDF")
    plt.legend()

    # Throughput and PDR bar subplot
    plt.subplot(2,2,3)
    names = []
    throughputs = []
    pdrs = []
    drops = []
    for name in scenarios:
        res = summary[name]
        names.append(name)
        throughput = res['departures'] / sim_time
        pdr = res['departures'] / max(1, res['arrivals'])
        throughputs.append(throughput)
        pdrs.append(pdr)
        drops.append(res['dropped'])
    x = np.arange(len(names))
    width = 0.35
    plt.bar(x - width/2, throughputs, width, label='Throughput (pkt/s)')
    plt.bar(x + width/2, pdrs, width, label='PDR (ratio)')
    plt.xticks(x, names)
    plt.title("Throughput and PDR")
    plt.legend()

    # Jitter histogram subplot
    plt.subplot(2,2,4)
    for name in scenarios:
        lat = summary[name]['latencies']
        if len(lat) < 2:
            continue
        jitter = np.abs(np.diff(lat))
        # plot KDE-like histogram (normalized) for comparison
        plt.hist(jitter, bins=30, alpha=0.4, density=True, label=name)
    plt.xlabel("Jitter (s, |Δlatency|)")
    plt.ylabel("Density")
    plt.title("Jitter distribution")
    plt.legend()

    plt.tight_layout()
    plt.show()

    # Print numeric summary
    print("Numeric summary (sim_time = {:.1f}s, service_rate = {:.1f} pkt/s, buffer_size = {})".format(
        sim_time, service_rate, buffer_size))
    print("{:>12} {:>10} {:>10} {:>10} {:>10}".format("Scenario","Arrivals","Depart","Dropped","Throughput"))
    for name in names:
        r = summary[name]
        print("{:>12} {:10d} {:10d} {:10d} {:10.3f}".format(
            name, r['arrivals'], r['departures'], r['dropped'], r['departures']/sim_time
        ))

# Example scenarios: low, medium, high load
scenarios = {
    "Low": 1.0,    # λ = 1 pkt/s, service_rate=5 -> lightly loaded
    "Medium": 4.0, # λ = 4 pkt/s, near server capacity
    "High": 8.0    # λ = 8 pkt/s, overloaded -> drops, long queues
}

analyze_and_plot(scenarios, sim_time=300, service_rate=5.0, buffer_size=8)
```

Key observations during performance

* Latency rises steadily as load increases due to congestion.
* Jitter becomes more unpredictable in medium and high traffic.
* Throughput increases initially but drops slightly at high load.
* Under heavy load, packets experience longer delays and more variance.

Conclusion
QoS parameters degrade as traffic increases. High load causes higher latency, larger jitter values, and reduced throughput. Maintaining optimal traffic flow and using efficient MAC protocols is essential for sustaining QoS in wireless sensor networks.