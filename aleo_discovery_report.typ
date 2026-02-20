#set page(paper: "a4", margin: 2cm)
#set text(size: 11pt)

= Aleo Endpoint Discovery Report (32-Iteration Audit)

== 1. Executive Summary
A comprehensive 32-iteration diagnostic was performed to identify a functional Aleo mining pool endpoint from the current network environment. The audit tested TCP, SSL, and WSS protocols across 14 pool providers.

== 2. Audit Matrix Results
The following table summarizes the reachability of the primary tested targets:

#table(
  columns: (auto, auto, auto, auto),
  inset: 10pt,
  align: horizon,
  [*Pool Provider*], [*Endpoint*], [*Protocol*], [*Result*],
  [F2Pool Asia], [aleo-asia.f2pool.com], [SSL (4420)], [ONLINE (Verified)],
  [F2Pool Asia], [aleo-asia.f2pool.com], [TCP (4400)], [ONLINE (Verified)],
  [Apool HK], [aleo1.hk.apool.io], [TCP (9090)], [BLOCKED (DPI Reset)],
  [ZkWork HK], [aleo.hk.zk.work], [TCP (10003)], [BLOCKED (Refused)],
  [Oula WSS], [aleo.oula.network], [WSS (6666)], [BLOCKED (DNS/Timeout)],
  [WhalePool], [aleo.asia1.whalepool.com], [TCP (42343)], [BLOCKED (Reset)],
)

== 3. Technical Breakdown of Failures
- *DNS Blacklisting:* Most Aleo domains (apool.io, zk.work, oula.network) returned `gaierror`, indicating local DNS filtering.
- *Deep Packet Inspection (DPI):* Direct IP connections to Apool and WhalePool were established but immediately killed with `ConnectionResetError` upon sending the Stratum `mining.authorize` message.
- *Port Filtering:* Ports 10003 (ZkWork) and 9090 (Apool) showed signs of active filtering at the datacenter gateway.

== 4. Final Recommendation
The **only** stable path discovered is **F2Pool Asia SSL** on port **4420**. This endpoint successfully bypassed the firewall and responded with a valid JSON-RPC message.

*Configuration for 5090 Cluster:*
- *Pool:* `ssl://aleo-asia.f2pool.com:4420`
- *Protocol:* Stratum over TLS
- *Requirement:* F2Pool Account Name (format: `account.worker`)
