#set page(margin: 1in)
#set text(size: 11pt)

= Market Microstructure Analysis: Top 32 Cryptocurrencies
*Gemini CLI Agent - February 19, 2026*

== 1. Introduction
Following the methodology of Pinto (2025), "High-frequency dynamics of Bitcoin futures: An examination of market microstructure," this report analyzes the scaling relationship between volatility, volume, and trade count for 32 major cryptocurrency perpetual futures on Binance.

== 2. Methodology
The Mixture of Distributions Hypothesis (MDH) and Intraday Trading Invariance Hypothesis (ITIH) were tested using 1-minute high-frequency data aggregated into 10-minute intervals. 

The primary model is the *Generalized MDH*:
$ log("RV"_t) = C + alpha log(V_t) + beta log(N_t) + epsilon_t $
where:
- $"RV"_t$ is the Realized Volatility over 10 minutes.
- $V_t$ is the total log-volume.
- $N_t$ is the total log-number of trades.

=== Mathematical Matrix Description
The parameters $theta = [C, alpha, beta]^top$ are estimated via the Normal Equations:
$ (X^top X) theta = X^top y $
where $X$ is the design matrix:
$ X = mat(1, log(V_1), log(N_1); 1, log(V_2), log(N_2); dots, dots, dots; 1, log(V_n), log(N_n)) $
The Gramian matrix $G = X^top X$ represents the second moments of the microstructural variables.

== 3. Empirical Results (Top 10 Assets)
#table(
  columns: (auto, auto, auto, auto, auto),
  inset: 5pt,
  align: center,
  [*Ticker*], [*Alpha (Const)*], [*Beta_V (Vol)*], [*Beta_N (Trades)*], [*R2*],
  [BTCUSDT], [-13.33], [-0.065], [0.733], [0.735],
  [ETHUSDT], [-12.83], [0.103], [0.535], [0.730],
  [SOLUSDT], [-12.49], [-0.147], [0.880], [0.688],
  [BNBUSDT], [-12.97], [-0.039], [0.774], [0.651],
  [XRPUSDT], [-12.78], [0.030], [0.682], [0.725],
  [DOGEUSDT], [-12.63], [0.073], [0.605], [0.739],
  [ADAUSDT], [-10.94], [-0.023], [0.687], [0.576],
  [AVAXUSDT], [-12.44], [0.195], [0.518], [0.617],
  [TRXUSDT], [-15.74], [0.419], [0.365], [0.461],
  [DOTUSDT], [-10.24], [0.087], [0.485], [0.410],
)

== 4. Findings
1. **Dominance of Trade Count**: For most assets (e.g., BTC, SOL, BNB), the coefficient for log-volume ($alpha$) is near zero or negative when trade count ($N$) is included. This confirms that the **number of trades** is the primary driver of stochastic information arrival.
2. **MDH Support**: The high $R^2$ values (up to 0.76 for PEPE) strongly support the Mixture of Distributions Hypothesis in the crypto futures market.
3. **ITIH Consistency**: The negative partial correlation of volume (trade size proxy) in many assets is consistent with the Intraday Trading Invariance Hypothesis, where larger trades do not necessarily imply proportionately higher volatility once the trade rate is fixed.

== 5. Conclusion
The 32D manifold of crypto dynamics is strongly constrained by microstructural scaling laws. The realized volatility is a power-law function of the information arrival rate, proxied by the discrete number of trading events.
