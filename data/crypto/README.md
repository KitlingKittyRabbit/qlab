# qlab crypto data root

这里是当前正式 crypto 主线的 canonical 数据根目录。

## 目录约定

- caches/: 正式回测和 live 使用的 pkl caches
- manifests/: cache summary / registry summary 等清单
- raw_history/: 原始历史与 snapshot 追加层

## 边界

- 当前正式 baselines 和后续 live 只应读取这里。
- crypto_research/data 不再是正式主线的数据仓库；当前已归档到 workspace_archive/legacy_root_misc_2026-04/crypto_research/data。
- 如需在服务器上换盘，使用 QLAB_CRYPTO_DATA_DIR 或 COINGLASS_DATA_DIR 覆盖默认路径。