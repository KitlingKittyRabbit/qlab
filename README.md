# qlab

量化研究基础设施 — 可靠的指标计算、Walk-Forward 验证与回测引擎。

`qlab` 只承载代码与公开文档，不承载滚动热数据、私有运行状态或密钥配置。

## 为什么需要这个库

| 问题 | 解决方案 |
|------|---------|
| `sqrt(365)` 年化 14d 收益导致 Sharpe 膨胀 3.74× | `sharpe(returns, holding_days=14)` 自动用 `sqrt(365/14)` |
| Sortino / Calmar 口径不一致 | `sortino()` 用全样本 downside deviation，`calmar()` 用 CAGR |
| 全样本 IC 方向泄漏到 WF | `ic_direction()` 只接受训练切片 |
| 回测/实盘 z-score 方法不一致 | `zscore()` 和 `zscore_fixed()` 明确区分 |
| 重叠收益伪造高 Sharpe | `run_signal_backtest()` 强制非重叠持仓 |
| 同一研究里各脚本各自重写 horizon IC 衰减 | `forward_returns()` + `ic_decay()` 统一口径 |
| 每次都手写分位收益分组和对齐 | `quantile_returns()` 统一输出分位收益表 |
| 每个脚本重写一遍 Sharpe | 统一 `metrics.py`，一次写对 |

## 安装

```bash
pip install -e ".[dev]"
```

## 数据边界

- `qlab` 默认不再把仓内 `data/crypto` 当作正式数据根。
- crypto 数据应放在仓库外，或放在单独的私有 data 仓库工作树中。
- 运行时通过 `QLAB_CRYPTO_DATA_DIR` 或 `COINGLASS_DATA_DIR` 指向正式数据根。
- 推荐运维模型是：公开代码仓 + 私有 data 仓库快照；切换运行设备时，停旧进程、更新代码、更新 data，再启动新机器。

更具体的公开约定见 [docs/crypto_data_root.md](docs/crypto_data_root.md)。

## 模块

| 模块 | 用途 |
|------|------|
| `qlab.metrics` | Sharpe / Sortino / MaxDD / Calmar / WinRate / PF |
| `qlab.signal` | z-score / IC / 阈值信号 |
| `qlab.diagnostics` | Forward return 构造 / IC 衰减 / 分位收益 |
| `qlab.walkforward` | WF 切割器 (支持 embargo) |
| `qlab.spread` | 价差构建 / 协整检验 / 半衰期 |
| `qlab.cost` | 成本模型 (Crypto / FX) |
| `qlab.backtest.signal_bt` | 固定持仓期回测 (crypto) |
| `qlab.backtest.spread_bt` | 均值回复价差回测 (FX pair trade) |

说明：`max_drawdown()` 只接受严格为正的净值 / NAV 曲线；如果你手里是绝对 PnL 序列，应使用绝对金额回撤而不是百分比回撤。
说明：`run_spread_backtest()` 会返回绝对值 `daily_pnl`，但 `sharpe` / `max_drawdown` 基于 `daily_returns = daily_pnl / capital_base` 计算；要横向比较不同策略，必须使用一致的 `capital_base` 口径。

## 用法

```python
from qlab import sharpe, walk_forward_splits, threshold_signal, ic_decay, quantile_returns
from qlab.backtest import run_signal_backtest, run_spread_backtest

# 正确年化 14 天持仓周期的 Sharpe
sr = sharpe(returns_14d, holding_days=14, trading_days_per_year=365)

# Walk-forward (带 embargo 防止前瞻收益泄漏)
for fold in walk_forward_splits(dates, train_days=90, test_days=90, embargo_days=14):
    train = select_dates(data, fold, "train")
    test = select_dates(data, fold, "test")
    ...

# 非重叠回测
result = run_signal_backtest(signals, prices, holding_days=14, cost_bps=5)
print(f"Sharpe: {result['sharpe']:.2f}")

# 不同 horizon 下的 IC 衰减
decay = ic_decay(factor, prices=prices, horizons=[1, 3, 7, 14])
print(decay)

# 单 horizon 的分位收益
qret = quantile_returns(factor, prices=prices, horizon=14, n_quantiles=5)
print(qret)

# 价差回测
result = run_spread_backtest(spread, entry_z=2.0, exit_z=0.0, stop_z=4.0)
```

## 测试

```bash
pytest
```

说明：某些与私有 live 脚本联动的集成测试需要显式提供外部脚本路径，默认不会在公开仓库里硬编码私有仓库位置。

## 许可证

本项目采用 [MIT License](LICENSE)。
