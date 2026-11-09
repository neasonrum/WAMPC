# Full Diffusion Geo：原始 100 步的 CARLA 加载优化

当前说明见 [FULL_DIFFUSION_CARLA_LOADING.md](FULL_DIFFUSION_CARLA_LOADING.md)。当前入口为 `evaluation/run_full_diffusion_carla_warm.py`，CARLA 服务常驻，每条路线使用新 Python evaluator 和新 Traffic Manager，并完整重置场景。原始 100 步采样保持不变。

早期 DDIM 常量缓存、整批共用 Python evaluator、静态道路图磁盘缓存均保留为试验记录，未启用在当前入口。
