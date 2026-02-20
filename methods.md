# Mathematical Methods Summary

## 1. Neural Ordinary Differential Equations (Neural ODEs)
Treats neural network depth as a continuous trajectory.
**Equation:**
$$\frac{dh(t)}{dt} = f(h(t), t, 	heta)$$
**Solution:**
$$h(T) = h(0) + \int_0^T f(h(t), t, 	heta) dt$$
Implemented using Euler discretization in C++23.

## 2. Liquid Time-Constant (LTC) Networks
Inspired by biological neural circuits, using input-dependent time constants.
**Equation:**
$$\frac{dh}{dt} = - \left[ \frac{1}{	au} + f(x, t) ight] h + f(x, t) A$$
where $	au$ is the time constant and $A$ is the bias/leakage parameter.

## 3. Discontinuous Piecewise Linear (PWL) Models
Allows for jumps in both the value (intercept) and the derivative (slope).
**Equation:**
$$y = \beta_0 + \beta_1 x + \sum_{i=1}^N \left[ \gamma_{0,i} \mathbb{I}(x > c_i) + \gamma_{1,i} (x-c_i) \mathbb{I}(x > c_i) ight]$$
where $\mathbb{I}$ is the indicator function and $c_i$ are knot points.

## 4. Merton Jump-Diffusion (SDE)
Stochastic model for asset prices with discrete jumps.
**Equation:**
$$\frac{dS_t}{S_t} = \mu dt + \sigma dW_t + d\left( \sum_{i=1}^{N_t} (Y_i - 1) ight)$$
- $\mu$: Drift
- $\sigma$: Volatility
- $W_t$: Standard Brownian Motion
- $N_t$: Poisson process with intensity $\lambda$
- $Y_i$: Jump size (log-normal)

## 5. Optimization: R-Adam
A variant of Adam that rectifies the variance of the adaptive learning rate.
**Gradient Clipping:**
$$	ext{grad} = 	ext{grad} 	imes \min\left(1, \frac{	ext{threshold}}{\|	ext{grad}\|}ight)$$
Used to ensure numerical stability in high-frequency financial data.
