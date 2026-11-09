# Full Diffusion Geo：CARLA 加载与服务复用

评测入口为 `evaluation/run_full_diffusion_carla_warm.py`，使用隔离环境 `.venvs/full-diffusion-carla/bin/python`。模型继续使用原始 `VPFullDiffusion` 和 checkpoint 配置的 100 步采样。模型、感知编码器、控制器、图像输入、20 Hz 物理步长、路线、计分和场景超时均沿用现有实现。

每张 GPU 有一个 worker 持有 CARLA 服务。每条路线启动新的 Python evaluator、agent、随机数生成器和 Traffic Manager，并完整执行原来的 `load_world(..., reset_settings=False)`。服务启动后通过 RPC 检查就绪；每条路线连接已有服务时省去固定 30 秒等待。服务缓存保留在 WMMPC 的 `.runtime/carla_cache`，日志按 worker/server/route 分开保存。

默认每个服务最多执行 32 条路线。基础设施错误会重启该 worker 自己的服务，仅重试未完成的路线，每条最多两次；驾驶失败保留在成绩中。checkpoint、源码、GPU UUID、服务 PID、各条路线的 evaluator PID、RPC 就绪耗时均有记录。GPU 锁和现有 LSF allocation 的检查继续生效。

`evaluation/run_full_diffusion_carla_fast.py` 是内部单路线子进程辅助代码。早期“一整个 batch 共用一个 Python evaluator”的试验保留在 `results/full_diffusion_carla_20261031_lifecycle_validation/paired01`；该试验第二条路线出现成绩差异，不作为当前入口。当前入口隔离每条路线的 Python 状态。

原始采样文件 `method/full_diffusion/diffusion.py`、`model.py`、`inference/policy.py`、`inference/carla_agent.py` 未修改。`cached_100_ddim.py` 不在当前评测路径中，`torch.load` 缓存和指标日志替换也未启用。静态道路图磁盘缓存未通过原生 waypoint 的精确恢复核验，当前入口明确关闭该试验。地图位置参数和 Engine ini 启动参数在此 CARLA 构建中被忽略，首图预加载同样未启用。

真实闭环验证在 `results/full_diffusion_carla_20261031_lifecycle_validation/warm01`。对照使用 `paired01/baseline`，checkpoint SHA256 均为 `031fd8b5a120200de81bf14a8f170d191da3f2b3ed32691fded2a46120c8cac3`。同一张 A100、每个 worker 四个 CPU 核，路线 1711 和 1773 前后均 Completed，DS=100、RC=100、penalty=1。

| 两条路线合计 | 原流程 | 服务复用流程 |
|---|---:|---:|
| 总耗时 | 700.96 秒 | 595.23 秒 |
| CARLA `duration_system` 合计 | 491.75 秒 | 450.33 秒 |
| 非行驶开销 | 209.20 秒 | 144.90 秒 |
| CARLA 启动次数 | 2 | 1 |

总耗时减少 15.08%；非行驶开销减少 30.74%。非行驶开销定义为总耗时减去 `duration_system`，包括启动、地图加载、agent 准备、收尾，不等同于纯地图加载时间。行驶时长也发生了波动，不能将总耗时收益全部归因于加载优化。两条路线不能代表完整 220 路线耗时，也未证明轨迹逐点一致。

服务 PID 两条路线均为 1000727，evaluator PID 分别为 1001554、1003269，证实复用了 CARLA 而隔离了每条路线的 Python 状态。首个服务启动就绪为 14.42 秒；各路线连接现有服务分别为 0.050 秒和 0.046 秒。原始 100 步采样器检查、完整地图重载事件、进程身份、计时、源码校验和见 `results/full_diffusion_carla_20261031_lifecycle_validation/warm01_loading_report.json`。

W002 的共享 CARLA 数据集导出曾返回 `Errno 116: Stale file handle`。现已在 WMMPC 内建立未改内容的私有运行副本 `.runtime/carla_0.9.15_v1`，供加载器和故障恢复使用。副本记录了来源、大小、关键文件校验和；共享安装和训练环境均未改动。

准备后续完整评测：

```bash
.venvs/full-diffusion-carla/bin/python evaluation/prepare_fast_carla_run.py \
  --checkpoint results/full_diffusion_carla_20261031_epoch10/checkpoint/epoch_10.pt \
  --baseline-run results/full_diffusion_carla_20261031_epoch10 \
  --run-root results/full_diffusion_carla_20261031_epoch10_carla_warm
```

进入已提供且 GPU 空闲的 `battach JOB_ID` 后启动该 allocation 的两个 worker：

```bash
.venvs/full-diffusion-carla/bin/python evaluation/launch_fast_carla_node.py \
  --job-id JOB_ID \
  --run-root results/full_diffusion_carla_20261031_epoch10_carla_warm
```

准备脚本只创建独立 checkpoint、220 路线的八个分片、源码快照和哈希。launcher 检查空闲 GPU、allocation、主机、源码及 checkpoint 身份后，在 allocation 内的 tmux 启动，不提交或取消 LSF 作业。当前 epoch-10 正式结果仍在 `results/full_diffusion_carla_20261031_epoch10`。上述 warm 目录已准备，源码校验已通过；尚未启动另一轮完整 220 路线评测。
