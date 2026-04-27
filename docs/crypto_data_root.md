# qlab crypto data root

这里记录 qlab 当前 crypto 正式数据根的公开约定，以及代码仓和数据仓的边界。

## 当前状态

当前已经进入第二阶段：

1. qlab 不再默认回落到仓内的 `data/crypto`。
2. 正式热数据根必须通过环境变量显式指定到仓外目录。
3. 仓内 `data/crypto` 只保留迁移过渡价值，不再是默认正式数据根。

这一步的目的，是把公开代码仓和正式数据仓彻底解耦，而不是继续保留隐式回退。

## 当前行为

- 如果进程环境里设置了 `QLAB_CRYPTO_DATA_DIR`，则优先使用该目录作为正式数据根。
- 如果没有设置 `QLAB_CRYPTO_DATA_DIR`，但设置了 `COINGLASS_DATA_DIR`，也会使用该目录。
- 如果进程环境里没有这两个变量，qlab 还会继续尝试从当前运行进程显式提供的 env 文件里读取同名配置。
- 如果环境和已知 `.env` 文件里都没有这两个变量，qlab 会直接报错，不再偷偷回到仓内 `data/crypto`。
- 如果需要临时兼容旧目录，也必须显式把 `QLAB_CRYPTO_DATA_DIR` 指向旧的仓内路径。
- 如果没有设置 `QLAB_TRADE_ENV_PATH`，刷新脚本只会按兼容逻辑查看旧的 trade `.env` 路径；公开 qlab 不假定任何私有仓库结构。

## 目录约定

- `caches/`: 正式回测和 live 使用的 pkl caches
- `manifests/`: cache summary / registry summary 等清单
- `raw_history/`: 原始历史与 snapshot 追加层

## 推荐模型

推荐采用下面这套长期模型：

1. `qlab` 保持公开代码仓，不承载滚动热数据。
2. 正式 crypto 数据放在仓库外，或放在单独的私有 data 仓库工作树中。
3. 选一台机器作为唯一写入源，负责刷新和发布最新正式数据。
4. 其它机器只更新代码和正式数据，不直接在公开代码仓里维护 `data/crypto`。
5. 切换 live 设备时，先停旧机器 live，再更新代码和数据，然后启动新机器。

如果你的团队采用“私有 data 仓库快照”模式，那么 `QLAB_CRYPTO_DATA_DIR` 最自然的指向，就是该私有 data 仓库在本机的 checkout 路径。

## 最小校验

最小校验示例：

```bash
export QLAB_CRYPTO_DATA_DIR=/your/external/data/root
python - <<'PY'
from qlab.data.crypto import DATA_ROOT, CACHE_DIR, MANIFEST_DIR, RAW_HISTORY_ROOT
print(DATA_ROOT)
print(CACHE_DIR)
print(MANIFEST_DIR)
print(RAW_HISTORY_ROOT)
PY
```

如果输出的是你的正式数据根，说明当前运行环境已经正确指向外部数据仓。

## 回滚方式

第二阶段的回滚仍然便宜，但不再是“撤环境变量自动回仓内”：

1. 把 `QLAB_CRYPTO_DATA_DIR` 改为你希望使用的其他目录。
2. 如果必须临时回到旧仓内数据，就显式把它设成 `qlab/data/crypto` 的绝对路径。
3. 重启使用 qlab 的 shell、进程或服务。

这样做的目的是避免“忘了配环境变量，却悄悄读回仓内旧数据”。

## 不同设备切换原则

- 写入源机器负责刷新和发布正式数据。
- 其它设备只更新代码和正式数据，不直接改写公开仓库里的 `data/crypto`。
- 如果使用私有 data 仓库，切换运行设备时的核心动作应是：停旧 live、更新代码、更新 data、启动新 live。
- 是否迁移 runtime 状态，取决于你的实盘运营约束；这属于私有运维细节，不在公开 qlab 文档里展开。

## 当前边界

- 当前正式 baselines 和后续 live 只应读取同一份正式数据根。
- 历史上的研究数据目录不再是正式主线数据仓库；正式入口只应是当前配置的外部数据根。
- 仓内 `data` 目录即使被保留，也不应再被当作默认正式数据根。