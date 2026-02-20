#set page(paper: "a4")
#set text(font: "Linux Libertine", size: 10pt)

= Activation Function Benchmark Analysis

This benchmark evaluates six different activation functions on 16 random nonlinear approximation problems. Each model was trained for 4 seconds using R-Adam with muP scaling and Weight Normalization.

== Performance Comparison

#table(
  columns: (auto, auto, auto),
  inset: 10pt,
  align: horizon,
  [*Activation Function*], [*Err Reduct/s*], [*Iterations/s*],
  [Alias (32 bins)], [0.2030], [2,480,814],
  [Vanilla (ReLU)], [0.1783], [3,648,564],
  [Stochastic Gaussian], [0.1773], [1,096,456],
  [Leaky ReLU], [0.1820], [3,674,772],
  [Uniform [-1, 1]], [0.1847], [2,577,889],
  [Uniform [0, 1]], [0.1824], [2,439,181],
)

== Observations
1. *Alias Activations* demonstrated the highest efficiency in error reduction per second (0.2030), outperforming static activations by approximately 11-14%.
2. *Vanilla and Leaky ReLU* achieved the highest raw throughput (iterations/s) due to their minimal computational overhead.
3. *Stochastic Gaussian* activations showed the lowest throughput, primarily due to the overhead of random number generation within the inner loop, but maintained competitive error reduction.
4. The learned nature of Alias bins allows the network to adapt its nonlinearity to the specific problem geometry, effectively offloading complexity from weights to nodes.
