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
`accumulation_dtype` argument. The returned `C_f` is the windowed summed
autocovariance `2 * tau_int * v_f`, where `v_f` is the lag-zero
autocovariance; it remains available when the full autocorrelation history is
not requested.
