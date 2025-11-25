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

MATLAB implementation (alternative)

```
data = randi([20 40], 1, 50);

avg_val = mean(data);
sum_val = sum(data);
min_val = min(data);
max_val = max(data);

disp("Original Data Count:");
disp(length(data));

disp("Transmitted After Aggregation: 1");

disp("Average:"); disp(avg_val);
disp("Sum:"); disp(sum_val);
disp("Min:"); disp(min_val);
disp("Max:"); disp(max_val);

reduction = (1 / length(data)) * 100;
disp("Data Reduction (%):"); disp(reduction);
```

Key observations during performance

* A large set of 50 values reduces to a single representative value after aggregation.
* Data transmission reduces by more than 98%.
* Aggregation maintains general information while reducing communication cost.
* Accuracy trade-off depends on the aggregation method used.

Conclusion
Data aggregation techniques significantly reduce redundant information and lower communication energy in wireless sensor networks. While aggregation improves efficiency and extends network lifetime, it may introduce a slight loss of precision compared to raw data transmission.