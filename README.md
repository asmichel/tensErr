# torch-uwerr

`torch-uwerr` provides a Torch-native implementation of the gamma-method
uncertainty estimate for dense Monte Carlo histories.

The public estimate types and scalar entry point are:

```python
from torch_uwerr import Estimate, GammaMethodEstimate, gamma_method_mean
```

`Estimate` is the shared base for all reported estimates. It stores `value`
and `stderr`; its derived `snr` is `abs(value) / stderr` for positive standard
errors and positive infinity for zero standard errors, independently of
`value`.

`gamma_method_mean(history)` accepts histories shaped `(..., chains, samples)`.
A one-dimensional `(N,)` tensor is treated as one unbatched chain, `(R, N)`
contains `R` independent chains for one observable, and `(*B, R, N)` returns
batch-shaped estimates with shape `B`.

The implementation centers each chain along the sample axis and computes the
dense autocovariance with `torch.fft.rfft` and `torch.fft.irfft`. It keeps the
history on its original device and only changes dtype according to the
`accumulation_dtype` argument.

## Version 0.2 vector-result API

`VectorNormGammaMethodEstimate` now inherits `Estimate` and uses the common
estimate field names. The version 0.1 names have been replaced without aliases:

| Version 0.1 | Version 0.2 |
| --- | --- |
| `norm` | `value` |
| `norm_stderr` | `stderr` |
| `norm_snr` | `snr` |
| `norm_stderr_of_stderr` | `stderr_of_stderr` |

Replica counts and the `Q_bar_*` diagnostics retain their existing names.
