#set page(paper: "a4")
#set text(font: "Linux Libertine", size: 10pt)

= Alias Activation Analysis

== 32-Neuron Basis Comparison
The alias activation model was re-evaluated across three weight bases: Random, FFT, and Haar. Each neuron learned a 512-bin piecewise linear activation function.

#grid(
  columns: (1fr, 1fr),
  gutter: 10pt,
  image("alias_activation_rand.png", width: 100%),
  image("alias_activation_fft.png", width: 100%)
)
#image("alias_activation_haar.png", width: 50%)

== 2-Neuron 20D Sphere Approximation
A specialized run with 2 neurons and 64-bin alias activations was trained for 120s to approximate a 20D sphere boundary ($||x||^2 < 0.25$).

=== Activation Curves
#grid(
  columns: (1fr, 1fr),
  gutter: 10pt,
  image("sphere_curve_neuron_1.png", width: 100%),
  image("sphere_curve_neuron_2.png", width: 100%)
)

=== Activation Map (2D Slice)
The following heatmap shows the network output over a 2D slice ($x_1, x_2$) of the 20D space, with other dimensions set to zero.

#image("sphere_activation_map.png", width: 60%)

The model successfully captured the circular boundary in the 2D projection, demonstrating the capacity of learned alias activations to represent non-linear manifolds with extremely low neuron counts.
