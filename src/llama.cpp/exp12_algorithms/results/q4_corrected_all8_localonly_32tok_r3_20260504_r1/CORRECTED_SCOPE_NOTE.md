# Corrected 8-core Scenario Scope Note

这轮修正了之前的错误口径：`2c/4c/6c/8c` 不表示只有这些核心可参与计算，而是始终允许 8 个 PC 核心参与计算，只是其中 N 个核心有背景负载。

本目录结果仅包含本地 no-SVD 策略：

- `baseline_no_svd`: 8 个 PC 核心全部参与。
- `major_only_no_svd`: 只使用低负载/空闲核心。

协同卸载 `edge_end_no_svd` 尚未补入本目录，因为当前 adb 真机 shell/connect 出现阻塞，不能给有效协同测量。

之前 2c 场景约 `12 tok/s` 的 baseline 来自错误口径：当时只让 2 个核心参与计算。修正为 8 核可参与后，2c 场景 baseline 回到约 `30+ tok/s`。
