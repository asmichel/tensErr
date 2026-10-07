[//]:<This document targets Github's markdown renderer; do not remove this comment.>
# tensErr

tensErr computes uncertainties for tensor-valued Markov chain Monte Carlo
ensembles using the gamma method
([Wolff, 2004](https://arxiv.org/abs/hep-lat/0306017),
[Wolff, 2009](https://arxiv.org/pdf/0812.0677#page=19)).
It works directly with PyTorch tensors, using the same interface on CPUs
and accelerators.

## Gamma method

```python
def gamma_method(
    samples: torch.Tensor,
    *,
    gamma_method_s: float = 2.0,
    accumulation_dtype: torch.dtype = torch.float64,
    return_autocorrelation: bool = False,
) -> GammaMethodEstimate: ...
```

`samples` may have shape $`(N,)`$ for one chain, $`(R,N)`$ for $`R`$ chains,
or $`(\ast B,R,N)`$ for batched observables. Every chain must have the same
retained length $`N\geq2`$.

The default `gamma_method_s=2.0` selects the autocorrelation window
automatically. Setting `gamma_method_s=0.0` disables autocorrelation analysis
and reports the IID variance of the supplied entries.

`gamma_method` returns a `GammaMethodEstimate` with the following fields:

| Field | Meaning |
| --- | --- |
| `value` | Mean of the supplied samples |
| `stderr` | Standard error of `value` |
| `snr` | $`\lvert\mathrm{value}\rvert/\mathrm{stderr}`$ if $`\mathrm{stderr}\gt0`$; $`+\infty`$ if $`\mathrm{stderr}=0`$ |
| `tau_int` | Integrated autocorrelation time $`\tau_{\mathrm{int}}`$ |
| `C_f` | Summed autocovariance $`C_f=2\tau_{\mathrm{int}}v_f`$, where $`v_f=\Gamma_f(0)`$ |
| `stderr_of_stderr` | Estimated uncertainty of `stderr` |
| `window` | Selected maximum lag |
| `sample_shapes` | Retained length of every replica |
| `autocovariance` | $`\Gamma_f(t)`$ with shape $`(\ast B,\lfloor N/2\rfloor)`$ when requested; otherwise `None` |
| `autocorrelation` | $`\rho_f(t)`$ with shape $`(\ast B,\lfloor N/2\rfloor)`$ when requested; otherwise `None` |

`GammaMethodEstimate` objects representing the same observable but arising from independent samples can be averaged to produce one `GammaMethodEstimate` using the overloaded `+` and `sum` operators. Error analysis can thus be performed using chunking, without loading all the samples simultaneously into memory. Combining estimates concatenates `sample_shapes`; `stderr_of_stderr`, `window`,
`autocovariance`, and `autocorrelation` are `None` on the result.

## Efficient norm and error for autocorrelated vectors

Documentation under construction.
