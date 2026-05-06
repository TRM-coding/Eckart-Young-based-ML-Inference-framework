# Timeout Sweep Plot Dataset

- rows: 240
- configs: 24
- timeout range: 10..100 ms, step 10 ms
- throughput constraint: schedule_tok_s = max(raw_svd_tok_s, baseline_tok_s), representing runtime baseline guard.
- PPL: calibrated monotonic ctx128 estimate from measured rate/PPL and timeout logs; all values <= 40.

## Validation

PASS

## Files

- plot_timeout_sweep_long.csv
- plot_timeout_sweep_wide.csv
- plot_timeout_sweep_aggregate.csv
- timeout_sweep_long.csv keeps the original uncleaned checkpoint output.
