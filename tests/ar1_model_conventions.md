# Stable Stationary Gaussian AR(1) Conventions

All scalar and vector AR(1) test fixtures use unit innovation covariance and
are initialized exactly in their stationary distributions. A seeded CPU-local
Torch generator supplies the Gaussian draws, so fixture construction neither
requires burn-in nor mutates Torch's global random-number state.

## Symmetric vector model

Assume the vector AR(1) model

```math
x_t=b+Ax_{t-1}+\varepsilon_t,\qquad x_t\in\mathbb{R}^d,
```

with independent Gaussian noise satisfying

```math
\varepsilon_t\sim N(0,I),\qquad
\mathrm{Cov}(\varepsilon_t,\varepsilon_s)=0\quad(t\ne s),
```

and with $`\varepsilon_t`$ independent of the past.

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

## Scalar specialization

The scalar fixtures are the $`d=1`$ specialization with $`A=\rho``,
$`|\rho|<1`$, stationary mean $`\mu`$, and $`b=(1-\rho)\mu`$:

```math
x_t=\mu+\rho(x_{t-1}-\mu)+\varepsilon_t,
\qquad \varepsilon_t\sim N(0,1).
```

They are initialized as

```math
x_0\sim N\!\left(\mu,\frac{1}{1-\rho^2}\right).
```

Consequently,

```math
\Gamma(h)=\frac{\rho^{|h|}}{1-\rho^2},
\qquad
C=\sum_{h\in\mathbb Z}\Gamma(h)=\frac{1}{(1-\rho)^2}.
```
