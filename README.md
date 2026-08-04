# tensErr

`tensErr` provides Torch-native gamma-method uncertainty estimates for
Monte Carlo histories.

## Scalar means

```python
import torch

from tensErr import gamma_method

estimate = gamma_method(
    samples,
    gamma_method_s=2.0,
    accumulation_dtype=torch.float64,
    return_autocorrelation=False,
)
```

`samples` may have shape $`(N,)`$ for one chain, $`(R,N)`$ for $`R`$ chains,
or $`(\ast B,R,N)`$ for batched observables. Every chain must have the same
retained length $`N\geq2`$. Chains are centered separately before their
autocovariances are pooled.

The default `gamma_method_s=2.0` selects the autocorrelation window
automatically. Setting `gamma_method_s=0.0` disables autocorrelation analysis
and reports the IID variance of the supplied entries. To estimate uncertainty
across independent complete chains, reduce each chain to its mean and pass the
resulting length-$`R`$ vector; passing the original $`(R,N)`$ histories instead
measures within-chain variation.

`gamma_method` returns a `GammaMethodEstimate`. The `+` operator combines
independent estimates with identical batch shapes; it does not add their
reported values. The result is weighted by each operand's total sample count,
with independent errors combined using the same weights. `sum(estimates)`
applies the same operation to a nonempty iterable.

| Field | Meaning |
| --- | --- |
| `value` | Mean of the supplied samples |
| `stderr` | Standard error of `value` |
| `snr` | $`\lvert\mathrm{value}\rvert/\mathrm{stderr}`$ if $`\mathrm{stderr}\gt0`$; $`+\infty`$ if $`\mathrm{stderr}=0`$ |
| `tau_int` | Integrated autocorrelation time $`\tau_{\mathrm{int}}`$ |
| `C_f` | Summed autocovariance $`C_f=2\tau_{\mathrm{int}}v_f`$, where $`v_f=\Gamma_f(0)`$ |
| `stderr_of_stderr` | Estimated uncertainty of `stderr`; `None` on a combined estimate |
| `window` | Selected maximum lag; `None` on a combined estimate |
| `sample_shapes` | Retained length of every replica |
| `autocovariance` | $`\Gamma_f(t)`$ with shape $`(\ast B,\lfloor N/2\rfloor)`$ when requested; otherwise `None` |
| `autocorrelation` | $`\rho_f(t)`$ with shape $`(\ast B,\lfloor N/2\rfloor)`$ when requested; otherwise `None` |

Combining estimates concatenates `sample_shapes`; `stderr_of_stderr`, `window`,
`autocovariance`, and `autocorrelation` are `None` on the result.

## Efficient norm and error for autocorrelated vectors

The standard interface is `VectorNormGammaMethodHelper`. Let $`X_i^r`$ be
sample $`i`$ from replica $`r`$, where each sample is a $`d`$-dimensional
vector. The $`R`$ replicas are independent, each replica may be autocorrelated
in $`i`$, all have length $`N`$, and $`x=\mathbb{E}[X_i^r]`$ is their common
mean. The helper estimates $`\lVert x\rVert`$ and its standard error.

Split the replicas into a first (widehat) block of size $`M`$ and a second
(widecheck) block of size $`K=R-M`$. The helper accumulates

```math
\bar X^r = \frac{1}{N}\sum_{i=1}^{N}X_i^r,
\qquad
\widehat{\bar X}
= \frac{1}{M}\sum_{r=1}^{M}\bar X^r,
\qquad
q_i^r = \widehat{\bar X}\mathbin{\cdot}X_i^r,
\qquad
Q_i = \frac{1}{K}\sum_{r=M+1}^{R}q_i^r.
```

With the replica counts known at construction and these reduced terms
streamed, the helper uses $`O(N+d)`$ working memory
([implementation note](notes/theory/gradient-diagnostic.md#implementation)),
compared with $`O(NRd)`$ for a naive retained-history analysis
([problem statement](notes/theory/gradient-diagnostic.md#problem-statement)).

In the example below, `widehat_replica_means` yields the $`M`$ vectors
$`\bar X^r`$, and the application-provided
`widecheck_projection_histories(widehat_mean)` yields the $`K`$ length-$`N`$
vectors $`q^r`$.

```python
from tensErr import VectorNormGammaMethodHelper

helper = VectorNormGammaMethodHelper(
    replica_count=R,
    sample_count=N,
    widehat_count=M,
)

for replica_mean in widehat_replica_means:
    helper.accumulate_widehat_term(replica_mean)

widehat_mean = helper.finalize_widehat()

for projection_history in widecheck_projection_histories(widehat_mean):
    helper.accumulate_widecheck_projection_term(projection_history)

estimate = helper.compute(
    gamma_method_s=2.0,
    return_autocorrelation=False,
)
```

Omit `widehat_count` to use the default $`M=\lfloor R/2\rfloor`$. The helper
enforces the order of these operations but does not count contributions, so
the caller must supply exactly $`M`$ widehat and $`K`$ widecheck replicas.

Let $`\bar Q`$ be the mean of the $`N`$ values $`Q_i`$, and let $`\delta_Q`$
be the standard error returned by `gamma_method` for that scalar history.
The reported value and standard error are

```math
\bar Q = \frac{1}{N}\sum_{i=1}^{N}Q_i,
\qquad
\widehat{\lVert x\rVert}
= \mathrm{sign}(\bar Q)\sqrt{|\bar Q|},
\qquad
\mathrm{stderr}(\widehat{\lVert x\rVert})
= \frac{\sqrt{K/R}\,\delta_Q}{|\widehat{\lVert x\rVert}|}.
```

`compute()` returns a `VectorNormGammaMethodEstimate`:

| Field | Meaning |
| --- | --- |
| `value` | $`\widehat{\lVert x\rVert}`$ |
| `stderr` | $`\mathrm{stderr}(\widehat{\lVert x\rVert})`$ |
| `snr` | $`\lvert\mathrm{value}\rvert/\mathrm{stderr}`$ if $`\mathrm{stderr}\gt0`$; $`+\infty`$ if $`\mathrm{stderr}=0`$ |
| `stderr_of_stderr` | Estimated uncertainty of `stderr` |
| `tau_int` | Integrated autocorrelation time $`\tau_Q`$, which is unchanged by propagation to the norm |
| `C_f` | Propagated summed autocovariance $`C_{\lVert x\rVert}=K C_Q/(R\lvert\bar Q\rvert)`$ |
| `window` | Selected maximum lag; `None` on a combined estimate |
| `sample_shapes` | Length of every original vector replica |
| `replica_count` | Total replica count $`R`$ |
| `widehat_count` | Widehat replica count $`M`$ |
| `widecheck_count` | Widecheck replica count $`K=R-M`$ |
| `autocovariance` | Propagated $`\Gamma_{\lVert x\rVert}(t)=K\Gamma_Q(t)/(R\lvert\bar Q\rvert)`$ when requested; otherwise `None` |
| `autocorrelation` | $`\rho_Q(t)`$, which is unchanged by propagation to the norm, when requested; otherwise `None` |
| `Q_bar_C_f` | Unpropagated $`C_Q`$ for a direct estimate; `None` on a combined estimate |
| `Q_bar_autocovariance` | Unpropagated $`\Gamma_Q(t)`$ when requested; otherwise `None` |

Error propagation through the square root is singular when `value` is zero.

`VectorNormGammaMethodEstimate` inherits `GammaMethodEstimate`. Adding two
vector-norm estimates therefore returns another vector-norm estimate, weighted
by their total original sample counts $`RN`$. Replica-block counts are added,
`sample_shapes` are concatenated, and dense-history and raw $`Q`$ diagnostics
are `None` on the combined result.

`vector_norm_gamma_method` is the lower-level entrypoint for callers that
already have the $`Q_i`$ history. It accepts `Q_history` with shape
$`(\ast B,N)`$ plus `replica_count` $`=R`$ and `widehat_count` $`=M`$, and
returns the same estimate. Use $`1\leq M\lt R`$.

Computations remain on the input device and use `accumulation_dtype`.
