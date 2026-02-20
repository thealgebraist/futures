#set page(paper: "a4", margin: 2cm)
#set text(font: "Inter", size: 10pt)

#align(center)[
  #text(size: 20pt, weight: "bold")[Zenith Audit: Distressed Ecoin Recovery Analysis] \
  #text(size: 12pt, fill: gray)[Analysis of Top Recovery Candidates for Undervalued Assets]
]

#v(1cm)

== Executive Summary
This report analyzes 32 "ecoins" that have experienced a >10x decline from their All-Time High (ATH). Assets are ranked by a *Recovery Score*, which combines stability (consolidation tightness), momentum (recent breakout logic), volume accumulation, and alpha capture capacity.

== Top Recovery Candidates

#let data = (
  ("IOSTUSDT", "79.16", "0.33%", "0.36"),
  ("GRTUSDT", "73.77", "1.48%", "0.81"),
  ("QTUMUSDT", "71.62", "1.83%", "0.26"),
  ("ANKRUSDT", "70.56", "-0.94%", "1.00"),
  ("IOTAUSDT", "68.82", "1.68%", "0.83"),
  ("ALGOUSDT", "68.58", "0.22%", "0.82"),
  ("VETUSDT", "66.57", "1.50%", "0.94"),
  ("THETAUSDT", "65.60", "-4.63%", "0.71"),
  ("FILUSDT", "64.97", "3.93%", "0.84"),
  ("KNCUSDT", "64.79", "-9.78%", "1.25"),
)

#table(
  columns: (1fr, 1fr, 1fr, 1fr),
  inset: 10pt,
  align: horizon,
  [*Ticker*], [*Recovery Score*], [*Momentum*], [*Vol Ratio*],
  ..data.flatten()
)

== Analysis Methodology
1. *Stability*: Measured as the inverse of return volatility over the last 1000 candles. High stability indicates a mature accumulation floor.
2. *Momentum*: Absolute price return over the last 1000 10-minute intervals.
3. *Vol Ratio*: Ratio of recent volume to the 6-month average.

== Conclusion
Candidates such as *IOST*, *GRT*, and *QTUM* show the most significant accumulation patterns coupled with early momentum signals. These assets are prioritized for the next phase of automated trade execution.
