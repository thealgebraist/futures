#set page(margin: 1in)
#set text(size: 11pt)

= 20D Non-linear Manifold Approximation: Neuron Scaling Analysis

== Overview
This study identifies the generalization "sweet spot" across 32 unique 20-dimensional non-linear approximation problems. Each problem explores a different functional regime (Interactions, High-Frequency, Rational, and Mixed non-linearities).

== Methodology
- *Dimensionality*: 20D input manifold.
- *Scaling*: #emph[muP] initialization + Weight Normalization.
- *Optimizer*: R-Adam with Gradient Clipping (5.0).
- *Experiments*: 512 independent trials (32 problems $times$ 16 widths).
- *Intensity*: 8 seconds of high-throughput C++23/Accelerate training per trial.

== Results: Aggregated Generalization
The following table summarizes the average normalized Mean Squared Error (MSE) across all 32 problems for each width $n$.

#table(
  columns: (1fr, 1fr),
  inset: 10pt,
  align: horizon,
  [*Neurons (n)*], [*Avg Norm MSE*],
  [RESULTS_PLACEHOLDER]
)

== Analysis
The aggregate data reveals a characteristic U-shaped generalization curve. 

== Conclusion
[CONCLUSION_PLACEHOLDER]
