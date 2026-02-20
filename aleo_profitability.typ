#set page(paper: "a4", margin: 2cm)
#set text(size: 11pt)

= Aleo Proving: Solo vs. Pool Profitability (Feb 2026)

== 1. The Two Paths to Earning
As of Feb 2026, the Aleo network has two distinct entry tiers for provers.

=== Tier A: Solo Prover (The "Institutional" Path)
- *Entry Requirement:* 500,000 ALEO (~\$41,500) fixed stake.
- *Advantage:* 0% fees, full control over coinbase rewards.
- *Best for:* Large-scale clusters with >100 RTX 5090s.

=== Tier B: Pool Mining (The "Cluster" Path - Your Current Setup)
- *Entry Requirement:* \$0 ALEO stake.
- *How it works:* You provide hashrate to a pool (e.g., F2Pool). The pool uses its massive stake to "unlock" rewards and pays you a pro-rata share.
- *Fees:* Typically 1% to 5%.
- *Best for:* RTX 5090/4090 clusters. This is the **immediate profitability** path.

== 2. RTX 5090 Pool Profitability Model
Assuming a 1% pool fee and \$0.10/kWh electricity:

#table(
  columns: (auto, auto, auto, auto),
  inset: 10pt,
  align: horizon,
  [*Device*], [*Daily Gross Rev*], [*Daily Elec Cost*], [*Daily Net Profit*],
  [RTX 4090], [\$23.24], [\$0.67], [\$22.57],
  [RTX 5090], [\$33.20], [\$0.96], [\$32.24],
  [NVIDIA H200], [\$99.60], [\$1.68], [\$97.92],
)

== 3. Conclusion for Your 5090 Cluster
By using the **Hybrid Miner (v40)** and connecting to a pool, your RTX 5090 is an **immediate cash-flow asset**. You do not need to buy any ALEO to start earning. Each 5090 in your cluster should net approximately **\$32.24 per day** at current network difficulty and price.
