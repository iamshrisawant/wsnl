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

MATLAB implementation (alternative)

```
readings = randi([20 40], 1, 20);

avg = mean(readings);
mn = min(readings);
mx = max(readings);
summ = sum(readings);
unique_vals = unique(readings);

disp("Original Readings:"); disp(readings);
disp("Average:"); disp(avg);
disp("Minimum:"); disp(mn);
disp("Maximum:"); disp(mx);
disp("Sum:"); disp(summ);
disp("After Redundancy Removal:"); disp(unique_vals);
disp("Reduction in packets:"); disp(length(readings) - length(unique_vals));
```

Key observations during performance

* Redundancy elimination significantly reduces the number of packets to be transmitted.
* Simple aggregation techniques like average and sum yield meaningful summaries of the sensed environment.
* Aggregation reduces energy consumption by lowering communication load.
* Accuracy decreases slightly but efficiency increases greatly.

Conclusion
Data aggregation effectively minimizes redundant transmissions and enhances energy efficiency in wireless sensor networks. Techniques such as averaging, minimum/maximum extraction, and redundancy elimination provide compact and meaningful representations of sensor data, leading to longer network lifetime.