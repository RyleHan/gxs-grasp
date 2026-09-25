| method | seen succ | seen succ (multi) | seen paired | seen select | unseen succ |
|---|---|---|---|---|---|
| G only (ignores the instruction) | 39.9 | 38.9 | 9.5 | 48.4 | 48.1 |
| GR-ConvNet + CLIP, additive (LGD baseline) | 45.5 | 48.7 | 15.2 | 59.1 | 48.2 |
| G x S, zero-shot S (temperature/bias only) | 55.3 | 55.0 | 29.5 | 68.6 | 58.2 |
| G x S, S loss on all patches | 54.4 | 55.5 | 27.6 | 70.1 | 58.4 |
| G x S (ours) | 59.6 | 58.7 | 32.4 | 73.0 | 58.9 |
| G x perfect selection (upper bound) | 88.9 | 87.9 | 79.0 | 98.5 | 89.7 |

n(seen)=1376, multi=589, pairs=105
