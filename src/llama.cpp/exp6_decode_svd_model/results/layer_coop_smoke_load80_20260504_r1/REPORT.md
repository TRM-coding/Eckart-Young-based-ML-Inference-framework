# Layer Coop Split/PC-Load Sweep Report

日期：2026-05-02

## 设置

- tokens: `32`
- PC cpus: `71-79`
- PC loads: `80`
- split M: `2,4,6,8`
- repeats: `1`
- phone server: `LAYER_COOP_KEEP_HOT=1`

## Raw Results

| PC load | split M | tok/s | ms/token | PC prefix ms/tok | phone tail ms/tok | net/sync ms/tok | status |
|---:|---:|---:|---:|---:|---:|---:|---|
| 80 | 2 | 0.49 | 2059.61 | 353.60 | 679.74 | 1026.26 | ok |
| 80 | 4 | 1.78 | 561.91 | 390.44 | 57.85 | 113.61 | ok |
| 80 | 6 | 0.62 | 1624.87 | 1025.43 | 112.09 | 487.34 | ok |
| 80 | 8 | 0.53 | 1891.39 | 1270.70 | 111.04 | 509.65 | ok |

## Best Split Per Load

| PC load | best split M | tok/s | ms/token | PC prefix ms/tok | phone tail ms/tok | net/sync ms/tok |
|---:|---:|---:|---:|---:|---:|---:|
| 80 | 4 | 1.78 | 561.91 | 390.44 | 57.85 | 113.61 |

## 说明

- `split M` 表示 PC 计算 `[0,M)`，手机计算 `[M,28)`。
- `net/sync ms/tok = client roundtrip - server decode`，包含 TCP 传输、收包、发包与两端同步等待。
- 本轮使用 64-token decode 作为 sweep；最终候选点建议再用 128-token 重复验证。
