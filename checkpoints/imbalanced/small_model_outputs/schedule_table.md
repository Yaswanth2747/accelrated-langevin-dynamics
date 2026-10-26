| schedule | method | TV ↓ | FID ↓ | near-dup |
|---|---|---|---|---|
| uniform (10 each) | baseline | 0.321 | 449.32 | 0.000 |
| uniform (10 each) | BD (classifier) | 0.212 | 406.86 | 0.002 |
| uniform (10 each) | BD (kNN, denoised) | 0.266 | 367.09 | 0.005 |
| fewer at high noise | baseline | 0.359 | 610.41 | 0.000 |
| fewer at high noise | BD (classifier) | 0.265 | 561.35 | 0.004 |
| fewer at high noise | BD (kNN, denoised) | 0.337 | 533.18 | 0.006 |
| more at high noise | baseline | 0.328 | 473.88 | 0.000 |
| more at high noise | BD (classifier) | 0.199 | 492.05 | 0.003 |
| more at high noise | BD (kNN, denoised) | 0.268 | 475.85 | 0.005 |