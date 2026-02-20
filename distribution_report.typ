#set page(margin: 1in)
#set text(size: 11pt)

= Analysis of Crypto Return Distributions (10m Intervals)
*Gemini CLI Agent - February 19, 2026*

== 1. Introduction
This report analyzes the statistical distribution of 10-minute log-returns for 31 major cryptocurrency futures over a period of 1 year (approximately 1.2 million samples). The goal is to identify the best-fitting distribution and characterize the "fat tails" commonly observed in financial high-frequency data.

== 2. Methodology
Log-returns were calculated as:
$ r_t = ln(P_t / P_{t-1}) $
We fitted the empirical data to three theoretical distributions:
1. *Gaussian (Normal)*: $f(x; mu, sigma) = 1 / (sigma sqrt(2 pi)) exp(- 1/2 ((x-mu)/sigma)^2)$
2. *Cauchy*: $f(x; x_0, gamma) = 1 / (pi gamma [1 + ((x-x_0)/gamma)^2])$
3. *Student-t*: $f(x; nu, mu, sigma) = Gamma((nu+1)/2) / (Gamma(nu/2) sqrt(nu pi) sigma) (1 + 1/nu ((x-mu)/sigma)^2)^(-(nu+1)/2)$

== 3. Empirical Results
The fitting results for the aggregate dataset (1,192,599 samples) are:

#table(
  columns: (auto, auto, auto),
  inset: 5pt,
  align: center,
  [*Distribution*], [*Parameters*], [*Observations*],
  [Gaussian], [$mu = -0.000019, sigma = 0.00385$], [Underestimates tails],
  [Cauchy], [$"loc" = -0.000023, gamma = 0.00157$], [Overestimates tails],
  [Student-t], [$nu = 2.71, "loc" = -0.000029, sigma = 0.00219$], [Best fit],
)

The degrees of freedom $nu = 2.71$ confirms that crypto returns exhibit significant leptokurtosis (fat tails). In a Gaussian world, $nu -> infinity$. Values of $nu < 3$ imply that the fourth moment (kurtosis) is undefined, and $nu < 2$ would imply undefined variance.

== 4. Visualization
#image("crypto_return_distribution.png", width: 80%)

The log-scale density plot demonstrates that the Student-t distribution (green) closely follows the empirical distribution (gray) across several orders of magnitude, whereas the Gaussian distribution (red) decays much too rapidly.

== 5. Conclusion
Crypto high-frequency returns are non-Gaussian and follow a fat-tailed distribution best modeled by a Student-t with approximately 2.7 degrees of freedom. This implies that "extreme" events (multi-sigma moves) occur several orders of magnitude more frequently than predicted by standard financial models.
