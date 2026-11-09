# WMMPC：CARLA 同卡两路实测与动态队列

2026-10-31。结论：这次同卡两路没有缩短整批评测时间，后续默认采用 **8 张卡、每卡一路、共享动态路线队列**。两路保留为实验选项。

## 实测结果

同节点 rng-dl01-w007、两个 A100 80GB，分别运行一路和两路；每张卡均固定使用 4 个逻辑 CPU。两路时每个 CARLA 实例分到 2 个逻辑 CPU。路线均为 1711、1773、3464、3865，覆盖 Town12、Town13。

| 配置 | 4 条路线总耗时 | DS | RC | SR |
|---|---:|---:|---:|---:|
| 每卡一路，共享推理进程 | 1129.63 秒／18.83 分钟 | 100 | 100% | 100% |
| 同卡两路，共享推理进程 | 1243.11 秒／20.72 分钟 | 100 | 100% | 100% |

两路耗时增加 **10.05%**。总耗时包含启动、地图与 agent 加载、驾驶、路线间切换和 CARLA 收尾；没有只比较 GPU 利用率。

这只有 4 条短路线，不能据此推算完整 220 条的加速比。其中路线 3865 为红灯转弯场景，一路组经历 15.1 秒仿真时间，两路组经历 40.5 秒；停车期间 tick 持续推进，最终正常完成。场景等待时长不同，因此这次结果也不能证明所有路线开两路都会更慢。

另一个节点 rng-dl01-w005 的原直接 GPU 单路对照用时 1132.25 秒，4 条路线同样全部满分。早期 RPC 单路试验用时 1186.61 秒；它早于外层 inference_mode 上下文修正，仅作为开发记录保留。

本次测试沿用 epoch 10 checkpoint，SHA256：

`031fd8b5a120200de81bf14a8f170d191da3f2b3ed32691fded2a46120c8cac3`

原 VPFullDiffusion、100 步去噪、感知、控制器、20 Hz 仿真与计分方式保持原样。diffusion.py、model.py、policy.py、carla_agent.py 的哈希与试验前快照一致。

## 两路实现与验证范围

这批 allocation 请求 exclusive_process 且未启用 MPS，实测同卡第二个独立 CUDA 进程无法初始化。因此实验用单个 CUDA 进程串行服务两个独立 policy session，各自保留原随机数生成器；每路保留独立 CARLA、evaluator、控制器、历史状态和 Traffic Manager。

固定输入封装、传输还原逐值无损；两个 session 的第一、第二次输出分别一致，随机数序列正常前进。但严格对照历史原 GPU 调用的数值一致性检查**未通过**：轨迹最大差异约 0.00265 米，目标速度差异约 0.01371 m/s，部分 BF16 检测输出也有差异。原单模型直接重放可以零差异，双模型原方法调用出现了上述差异，原因尚未完全定位。没有放宽误差阈值，也没有据此宣称完全数值等价。

因此正式默认入口使用 direct 后端，直接调用原 GPU forward。同卡两路暂不作为正式默认配置。

## 动态队列与资源保护

所有 GPU worker 从同一个 GPFS 队列领取路线，完成后继续领取，并优先选择当前 town。只要还有未领取路线，worker 就不会因原分片已跑完而提前退出；末尾剩余路线不足 8 条时仍会有空卡。

队列通过锁和原子写入防止重复领取。已通过单节点 4 进程并发测试，以及两个节点、4 个 worker、64 条模拟任务的跨节点测试：64 次领取全部完成，无重复领取。

驾驶失败作为有效成绩保留，不重跑。基础设施错误最多尝试两次；worker 意外退出后保留 claimed 状态，等待检查，避免盲目重复执行。

本次配对试验中，同一个 LSF 作业同时容纳 3 个 CARLA 实例及两组推理服务，作业内存峰值 **47.76 GiB**，实际 cgroup 上限 **250 GiB**；双路卡显存峰值 **17609 MiB**。本次测试的 OOM、OOM kill、CARLA 异常退出均为 0。未修改 GPU 请求、compute mode、MPS 或作业时限，未提交或取消任何 LSF 作业。

入口每 5 秒检查资源：作业内存软上限 64 GiB、GPU 显存软上限 60 GiB；启动前校验 allocation、主机、GPU UUID、CPU affinity、cgroup 内存、空闲 GPU 和剩余时间。默认单次任务预算 6 小时，并要求额外保留至少 15 分钟的 LSF 时间。新鲜 allocation 记录最多接受 15 分钟。触及保护条件时，只收尾该入口所属的测试进程。

## 已准备的后续入口

远程工作目录：

`/fs/scratch/rb-bd-dlp-rng-dl01-cr-tfx/special/tfx-901/rai-ap/workspace/nix3sgh/WMMPC`

已准备的运行目录：

`/fs/scratch/rb-bd-dlp-rng-dl01-cr-tfx/special/tfx-901/rai-ap/workspace/nix3sgh/WMMPC/results/full_diffusion_carla_dynamic_epoch10_20261031_v1`

配置为 8 GPU、每卡一路、direct 后端、220 条路线、100 步去噪。包含 checkpoint 固定副本、31 个源码与配置文件的快照和校验值、四个 allocation 的运行记录。源码、checkpoint 校验以及 w007 allocation 内预检查均已通过。**只完成准备，尚未用新入口重跑完整 220 条。** 原正式评测继续独立运行。

在登录节点先刷新 allocation 信息：

```bash
cd /fs/scratch/rb-bd-dlp-rng-dl01-cr-tfx/special/tfx-901/rai-ap/workspace/nix3sgh/WMMPC
python evaluation/refresh_carla_parallel_allocations.py \
  --run-root results/full_diffusion_carla_dynamic_epoch10_20261031_v1
```

通过 battach 分别进入四个已有 allocation，在各自 shell 内执行，JOB 替换为对应作业号 13046830、13053006、13047898、13047896：

```bash
.venvs/full-diffusion-carla/bin/python evaluation/launch_carla_parallel_node.py \
  --job-id JOB \
  --run-root results/full_diffusion_carla_dynamic_epoch10_20261031_v1
```

入口为本 allocation 的两个 GPU 启动命名 tmux 会话。GPU 已被占用、源码或 checkpoint 改变、allocation 记录过期、剩余时限不足时会拒绝启动。当前仍在正式评测的 GPU 需先完成已有任务。

汇总动态队列：

```bash
.venvs/full-diffusion-carla/bin/python evaluation/summarize_parallel_carla.py \
  --run-root results/full_diffusion_carla_dynamic_epoch10_20261031_v1 --watch
```

完成后增加 `--extended --merge` 可生成上游指标及合并结果。Comfortness 仍沿用现有上游脚本；其 0.1 秒约定与当前 0.05 秒仿真步长的可比性尚待核对，不能直接声称完全复现论文的该指标。

原始实验记录位于远程工作目录下 `results/full_diffusion_carla_parallel_validation_20261031/`，包含配对结果、资源日志、源码快照、严格重放失败记录、跨节点队列审计及 `parallel_validation_report.json`。本地同名 JSON 保存了这次结果快照。
