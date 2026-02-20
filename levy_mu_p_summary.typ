#set page(margin: 1in)
#set text(size: 11pt)

= Levy Stable FFNN with muP and Weight Normalization

== Overview
This report analyzes the performance of a 32-neuron Feed-Forward Neural Network implementing #emph[muP] (Maximal Update Parameterization) and Weight Normalization. Stochastic activations are drawn from a symmetric Levy Stable distribution.

== Methodology
- *Architecture*: 16 Input -> 32 Hidden -> 32 Hidden -> 1 Output.
- *Scaling*: muP initialization (hidden $1/sqrt("width")$, output $1/"width"$).
- *Optimization*: R-Adam with Weight Normalization (decoupled magnitude and direction).
- *Hardware*: C++23 with Apple Accelerate framework.

== Results: Validation MSE
#table(
  columns: (1fr, 1fr),
  inset: 10pt,
  align: horizon,
  [*Asset*], [*Final Val MSE*],
  [AAPL], [0.616],
  [GOOGL], [0.709],
  [MSFT], [51820.3 (Diverged)],
  [NVDA], [37596.4 (Diverged)],
)

== Analysis
The muP scaling and Weight Normalization successfully stabilized the training for AAPL and GOOGL, yielding very low validation errors. However, the divergence in MSFT and NVDA indicates that even with these advanced techniques, high-frequency stochastic activations require robust Lipschitz-aware clipping or lower learning rates to ensure global convergence.

== Theoretical Bounds
The generalization gap is bounded by the Lipschitz constant of the network, which is regularized by the Levy noise scale.
