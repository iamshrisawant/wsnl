**Experiment No. 1: Node Deployment Strategies**

**Theory for Viva**

Purpose of deployment

* To ensure good coverage, connectivity, network lifetime, and sensing reliability.
* Deployment determines how well the WSN monitors an area.

Types of deployment strategies

* Random deployment

  * Nodes are scattered arbitrarily.
  * Used when terrain is inaccessible (forests, war zones).
  * Coverage is unpredictable; may create coverage holes.
* Uniform deployment

  * Distance between nodes is nearly equal.
  * Provides balanced coverage and moderate connectivity.
  * Easier to analyze but needs controlled placement.
* Grid-based deployment

  * Nodes arranged in rows and columns.
  * Provides the most structured coverage and predictable connectivity patterns.
  * Ideal for environments where planned installation is possible.

Performance factors

* Coverage: How well the monitored area is sensed.
* Connectivity: Ability of nodes to communicate with neighbors or sink.
* Energy efficiency: Better-arranged nodes reduce retransmissions and energy waste.
* Fault tolerance: Dense or structured placement reduces the impact of node failures.

Observations typically seen

* Random placement creates uneven node density.
* Uniform placement maintains balanced spacing.
* Grid placement maximizes area coverage with minimal overlapping sensing regions.

Summary points for viva

* Deployment strategy directly influences the quality and efficiency of a WSN.
* Grid deployment is generally optimal when structured placement is possible.
* Random deployment is used where manual placement is not feasible.
* Coverage holes and connectivity gaps occur mostly in random deployments.

---

**Implementation for Performance**

Objective
To generate, visualize, and compare node positions for random, uniform, and grid-based deployment within a defined area.

Requirements

* Python with NumPy and Matplotlib or MATLAB (any one).
* System capable of plotting scatter graphs.

Procedure (conceptual steps)

1. Define sensing area dimensions, e.g., 100 × 100 units.
2. Select number of nodes, e.g., 50 nodes.
3. Generate node coordinates using three strategies:

   * Random: randomly generated (x, y) positions.
   * Uniform: evenly spaced positions using linear spacing.
   * Grid: matrix-style placement using meshgrid.
4. Plot all three deployments separately to compare coverage visually.
5. Observe node density, spacing, and presence of coverage holes.
6. Interpret the deployment patterns based on coverage and connectivity expectations.

Python implementation (fast demonstration)

```
import numpy as np
import matplotlib.pyplot as plt

N = 50
area = 100

x_rand = area * np.random.rand(N)
y_rand = area * np.random.rand(N)

steps = int(np.sqrt(N))
x_uni = np.linspace(0, area, steps)
y_uni = np.linspace(0, area, steps)
xu, yu = np.meshgrid(x_uni, y_uni)

plt.figure(figsize=(12,4))

plt.subplot(1,3,1)
plt.scatter(x_rand, y_rand)
plt.title("Random Deployment")

plt.subplot(1,3,2)
plt.scatter(xu, yu)
plt.title("Uniform Deployment")

plt.subplot(1,3,3)
plt.scatter(xu, yu)
plt.title("Grid Deployment")

plt.show()
```

MATLAB implementation (alternative)

```
N = 50;
area = 100;

xr = area*rand(1,N);
yr = area*rand(1,N);

n = sqrt(N);
[xg, yg] = meshgrid(linspace(0,area,n), linspace(0,area,n));

subplot(1,3,1); scatter(xr, yr); title("Random Deployment");
subplot(1,3,2); scatter(xg(:), yg(:)); title("Uniform Deployment");
subplot(1,3,3); scatter(xg, yg); title("Grid Deployment");
```

Key observations during performance

* Random deployment shows irregular distribution and uneven coverage.
* Uniform deployment shows evenly spaced nodes but still not perfectly structured.
* Grid deployment shows a clear, organized pattern with predictable spacing.

Conclusion
Grid deployment provides the most efficient and reliable coverage, uniform deployment gives balanced spacing, and random deployment results in unpredictable coverage patterns.