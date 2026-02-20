#set page(paper: "a4", margin: 1cm)
#set text(size: 9pt)

= ZKP Proving Profitability: GPU Tier Analysis (2026)
*Date: February 20, 2026*
*Agent: Gemini CLI*

== 1. High-End Data Center Tier (H200, H100, A100)
For serious ZKP operations (zkVMs like SP1, RISC Zero, or L2 Sequencers), data center GPUs offer the high VRAM and memory bandwidth required for large circuits.

#table(
  columns: (1fr, 1.2fr, 1fr, 1fr, 1.2fr),
  [*GPU*], [*VRAM / Bandwidth*], [*Monthly Cost*], [*ZKP Target*], [*Reliability*],
  [H200], [141GB / 4.8 TB/s], [\$2,200 - \$2,700], [Large zkVMs], [99.9% (Tier 3)],
  [H100], [80GB / 3.4 TB/s], [\$1,100 - \$1,900], [L2 Batching], [99.9% (Reserved)],
  [A100], [80GB / 2.0 TB/s], [\$850 - \$950], [Standard PoSW], [High (Stable)],
)

== 2. Consumer Monthly Rental Tier (4090, 5090)
Renting consumer GPUs by the month on decentralized marketplaces (Vast.ai, RunPod) is the cheapest path to "small profit."

=== 2.1 The "Cheap Monthly" Strategy
- *RTX 4090:* ~\$201.60 / month (\$0.28 / hr). Best for Aleo mining or low-stakes L2 pools.
- *RTX 5090:* ~\$266.40 / month (\$0.37 / hr). 30% faster than 4090 in ZK tasks; 32GB VRAM allows larger batch sizes.
- *Risk Factor:* Cheap monthly rentals on Vast.ai often lack reliability. A 5% downtime can wipe out monthly profits in competitive protocols.

== 3. Reliability vs. Profitability Trade-off
We define the *Risk-Adjusted Profit* $Pi_r$ as:
$ Pi_r = (Pi times R) - (C_f times (1 - R)) $
Where $R$ is the reliability coefficient ($0.0$ to $1.0$) and $C_f$ is the cost of failure.

- *High Reliability (H100):* $R=0.99$. Capture 99% of block rewards.
- *Low Reliability (Cheap 4090):* $R=0.85$. Frequent interruptions lead to "orphaned proofs."

== 4. Conclusion: Which to Rent?
1. *For Aleo:* Rent an *RTX 5090* on a monthly spot basis with a private solver.
2. *For zkSync/Taiko:* Rent an *A100 (80GB)* for high VRAM-to-price efficiency.
3. *For Whale Proving:* **H200** instances are the only viable path for high-throughput zkVM tasks.

== 5. Strategic Comparison Table
#table(
  columns: (1fr, 1fr, 1.5fr, 1fr),
  [*Strategy*], [*Setup*], [*Cost / Month*], [*Risk Level*],
  [Bottom-Feeding], [RTX 4090 (Vast)], [\$200], [Critical],
  [Efficient Proving], [RTX 5090 (Vast)], [\$266], [High],
  [Stability Tier], [A100 (RunPod)], [\$850], [Low],
  [Whale Prover], [H200 (Genesis)], [\$2,500+], [None (SLA)],
)
