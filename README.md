# torch-uwerr

`torch-uwerr` provides a Torch-native implementation of the gamma-method
uncertainty estimate for dense Monte Carlo histories.

The public entry point is:

```python
from torch_uwerr import GammaMethodEstimate, gamma_method_mean
```

`gamma_method_mean(history)` accepts histories shaped `(..., chains, samples)`.
A one-dimensional `(N,)` tensor is treated as one unbatched chain, `(R, N)`
contains `R` independent chains for one observable, and `(*B, R, N)` returns
batch-shaped estimates with shape `B`.

The implementation centers each chain along the sample axis and computes the
dense autocovariance with `torch.fft.rfft` and `torch.fft.irfft`. It keeps the
history on its original device and only changes dtype according to the
`accumulation_dtype` argument.
