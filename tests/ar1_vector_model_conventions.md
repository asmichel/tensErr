# Stable Stationary Reflection-Invariant AR(1) Vector Model

Assume the vector AR(1) model

```math
x_t=b+Ax_{t-1}+\varepsilon_t,\qquad x_t\in\mathbb{R}^d,
```

with second-order white noise satisfying

```math
\mathbb{E}[\varepsilon_t]=0,\qquad
\mathrm{Cov}(\varepsilon_t)\succ0,\qquad
\mathrm{Cov}(\varepsilon_t,\varepsilon_s)=0\quad(t\ne s),
```

and with $`\varepsilon_t`$ uncorrelated with the past. Since
$`\mathrm{Cov}(\varepsilon_t)`$ is positive definite, rescale

```math
x_t\rightarrow \mathrm{Cov}(\varepsilon_t)^{-1/2}x_t.
```

and instead consider the simplified model with

```math
\mathrm{Cov}(\varepsilon_t)=I\,.
```

Assume stability and
reflection invariance in the normalized coordinates:

```math
\rho(A)\lt1,\qquad A=A^\top.
```

Because $`\rho(A)\lt1`$, the needed resolvents are

```math
(I-A)^{-1}=\sum_{j=0}^{\infty}A^j,\qquad
(I-A^2)^{-1}=\sum_{j=0}^{\infty}A^{2j}.
```

The stationary mean is

```math
\bar m=(I-A)^{-1}b.
```

Assume the process is initialized in its stationary second-order distribution:

```math
x_t-\bar m=\sum_{j=0}^{\infty}A^j\varepsilon_{t-j}.
```

The stationary covariance is

```math
\bar P=(I-A^2)^{-1}.
```

For every $`h\in\mathbb{Z}`$, the stationary autocovariance is

```math
\Gamma(h)=\mathrm{Cov}(x_{t+h},x_t)
=A^{|h|}(I-A^2)^{-1}
=\Gamma(-h)=\Gamma(h)^\top.
```

Define the summed autocovariance matrix $`C`$ by

```math
C:=\sum_{h\in\mathbb{Z}}\Gamma(h)
=\left[\sum_{h\in\mathbb{Z}}A^{|h|}\right](I-A^2)^{-1}
=(I-A)^{-2}.
```
