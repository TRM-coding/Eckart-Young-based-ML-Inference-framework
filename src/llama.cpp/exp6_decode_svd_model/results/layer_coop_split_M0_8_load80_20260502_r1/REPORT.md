# Layer Coop Split/PC-Load Sweep Report

日期：2026-05-02

## 设置

- tokens: `32`
- PC cpus: `71-79`
- PC loads: `80`
- split M: `0,2,4,6,8`
- repeats: `1`
- phone server: `LAYER_COOP_KEEP_HOT=1`

## Raw Results

| PC load | split M | tok/s | ms/token | PC prefix ms/tok | phone tail ms/tok | net/sync ms/tok | status |
|---:|---:|---:|---:|---:|---:|---:|---|
| 80 | 0 |  |  |  |  |  | client_rc_-6 |
| 80 | 2 | 13.18 | 75.86 | 20.02 | 29.11 | 26.73 | ok |
| 80 | 4 | 5.05 | 197.91 | 86.54 | 32.26 | 79.10 | ok |
| 80 | 6 | 2.91 | 344.00 | 165.24 | 55.42 | 123.34 | ok |
| 80 | 8 | 2.09 | 479.05 | 204.78 | 65.56 | 208.71 | ok |

## Best Split Per Load

| PC load | best split M | tok/s | ms/token | PC prefix ms/tok | phone tail ms/tok | net/sync ms/tok |
|---:|---:|---:|---:|---:|---:|---:|
| 80 | 2 | 13.18 | 75.86 | 20.02 | 29.11 | 26.73 |

## 说明

- `split M` 表示 PC 计算 `[0,M)`，手机计算 `[M,28)`。
- `net/sync ms/tok = client roundtrip - server decode`，包含 TCP 传输、收包、发包与两端同步等待。
- 本轮使用 64-token decode 作为 sweep；最终候选点建议再用 128-token 重复验证。
