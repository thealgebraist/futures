
#set page(paper: "a4", margin: 1cm)
#set text(size: 9pt)

= ZKP Profitability: Optimization & Simulation (2026)
*Date: February 20, 2026*
*Agent: Gemini CLI*

== 1. Optimization Model: Linear Programming
To determine the optimal GPU mix for a standard server node (32 Cores, 128GB RAM), we solved a linear programming problem:
$ "Maximize" Pi = sum_i n_i (R_i - C_i) $
Subject to:
$ sum n_i dot "CPU"_i <= 32 $
$ sum n_i dot "RAM"_i <= 128 $

*Optimal Allocation Result:*
- **RTX 5090 Nodes:** 5 Units
- **Other GPUs:** 0 Units
- **Bottleneck:** CPU Cores (5 \* 6 = 30 used of 32).

== 2. Monte Carlo Simulation Results
We simulated 10,000 hours of operation considering:
- **Aleo:** Poisson distribution for block wins ($lambda approx 0.17$ blocks/hr for 5x 5090).
- **Taiko:** Binomial distribution for proof races ($p_"success"$ decays exponentially with latency).

#table(
  columns: (1fr, 1fr),
  [*Metric*], [*Value*],
  [Mean Hourly Profit], [\$9.30],
  [Standard Deviation], [\$5.16],
  [Probability of Profit (> \$0)], [100.0%],
  [Monthly Expected Profit], [\$6,696.00],
)

== 3. Risk Sensitivity: The "Prover's Dilemma"
The simulation reveals that while the *mean* is high, the *variance* is significant due to Aleo's stochastic nature.
- **Aleo Contribution:** High reward, high variance.
- **Taiko Contribution:** Low reward per block, but high frequency, acting as a "profit floor."

== 4. Detailed GPU Performance Distribution (Simulated)
#table(
  columns: (1fr, 1.2fr, 1fr, 1.2fr),
  [*GPU*], [*Aleo Hash*], [*Taiko Latency*], [*Profit / Hr*],
  [RTX 5090], [1.8 Mh/s], [10s], [\$1.86],
  [RTX 4090], [1.4 Mh/s], [15s], [\$0.95],
  [A100 80G], [2.2 Mh/s], [8s], [\$0.72],
  [H100], [3.5 Mh/s], [4s], [-\$0.15],
)
*Note:* H100 shows negative profitability for a single node in this model due to high rental costs exceeding the marginal gain in proof speed for standard L2 fees.

== 5. Strategic Recommendation
The **RTX 5090** is currently the dominant "efficiency" card. While H100s are necessary for massive zkVM circuits, they are economically inefficient for commodity L2 proving where the reward-to-cost ratio favors high-density consumer cards.

*Action:* Target 5x RTX 5090 clusters with high-clock CPU cores (8+ per GPU) to minimize NTT bottlenecks.
