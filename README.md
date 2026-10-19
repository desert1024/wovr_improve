# WoVR-Improve

基于 [WoVR](https://github.com/DRL-CASIA/WoVR) 的 Vision-Language-Action（VLA）策略 RL 后训练实验。

本仓库从 WoVR 官方快照出发，增加了可靠性修复、分阶段耗时统计、CPU 内存管理、独立的 LIBERO 评估与视频录制，以及可复现的启动脚本。

论文：[WoVR: World Models as Reliable Simulators for Post-Training VLA Policies with RL](https://arxiv.org/abs/2602.13977)

## 与上游项目的关系

本项目依赖以下上游仓库和已发布的模型权重：

| 组件 | 来源 | 说明 |
| --- | --- | --- |
| WoVR（基座） | https://github.com/DRL-CASIA/WoVR | 官方实现，本仓库由此派生 |
| RLinf（框架） | https://github.com/RLinf/RLinf | 基础 RL 框架；WoVR 内置了一份 RLinf 快照（见 `UPSTREAM.md`） |
| 世界模型训练 | https://github.com/youmo445/diffsynth-studio-rlinf | DiffSynth-Studio 分支，提供 `diffsynth` 运行时（commit `6d9ce59`） |
| 策略 | https://github.com/moojink/openvla-oft | OpenVLA-OFT 源码，editable 安装（commit `e4287e9`） |
| 仿真环境 | https://github.com/Lifelong-Robot-Learning/LIBERO | LIBERO 基准 |
| 视频基座 | https://github.com/Wan-Video/Wan2.2 | Wan2.2-TI2V-5B |
| 策略权重 | https://huggingface.co/Haozhan72/Openvla-oft-SFT-libero-spatial-traj1 | LIBERO-Spatial SFT 策略 |
| 世界模型 | https://huggingface.co/RLinf/RLinf-Wan-LIBERO-Spatial | Wan 世界模型 + VAE + reward model + KIR 数据 |

## 相对上游的改动

- 修复 `get_env_attr`，兼容 gymnasium 0.29，使 Wan env 的 `offload` 能被正确找到；
- 增加 `[profile]` 分阶段耗时（`rollout`、`actor_train`、`sync_weights` 等）；
- 每个 step 之间主动回收 CPU 内存（`gc.collect()` + `malloc_trim`）；
- 增加独立的 LIBERO-Spatial 评估配置，支持视频录制；
- 增加 FSDP checkpoint 到 Hugging Face 格式的转换；
- 所有启动/评估/安装脚本统一放到 `scripts/`。

## 环境

实验使用的参考配置：

- 8× NVIDIA A800 80GB，400GB 主机内存
- torch 2.11.0 + cu128，flash-attn 2.8.3（cu12），transformers 4.40.1（OpenVLA-OFT fork）
- `actor`、`rollout` 常驻 GPU，`env`（Wan 世界模型）offload 到 CPU
- 无头渲染使用 `MUJOCO_GL=osmesa`

首次安装环境：

```bash
bash scripts/setup_env.sh
```

模型权重从 Hugging Face 下载，见上表。请把
`examples/embodiment/config/wan_libero_spatial_grpo_openvlaoft.yaml`
里的绝对路径改成你的代码和权重目录。

## 快速开始

```bash
# 申请 GPU worker（与集群调度器相关）
bash scripts/start.sh

# 进入 worker 后启动训练
bash scripts/run_wovr.sh

# 或者限制步数（STEPS=20）
bash scripts/run.sh
```

`scripts/start.sh` 使用集群的 `rlaunch`，请按自己的调度系统调整。

## 独立评估

`scripts/eval_video.sh` 会先把 FSDP checkpoint 转成 Hugging Face 格式，然后在 LIBERO-Spatial 上评估并保存视频：

```bash
bash scripts/eval_video.sh
```

输出：

- 指标：`eval_results/step20/`
- 视频：`eval_results/step20/video/eval/`

可用环境变量 `CKPT`、`HF`、`OUT`、`ENVS` 覆盖默认路径和评估环境数。

## Checkpoint 转换

`scripts/convert_ckpt_to_hf.py` 把训练 checkpoint
（`actor/model_state_dict/full_weights.pt`）转成 Hugging Face checkpoint，
并复用 base SFT checkpoint 的 config、tokenizer 和 processor 文件。

## 说明

- 本集群的 `diffsynth` 实际从 `youmo445/diffsynth-studio-rlinf` 安装，
  因为当时从 `RLinf/diffsynth-studio` 克隆网络不稳定；
- `requirements/install.sh` 可能会修改 `pyproject.toml` 以固定 CUDA 12.8 wheel，
  这属于环境相关改动；
- 默认训练参数不是低内存配置，请根据硬件调整 `total_num_envs`、
  `rollout_epoch`、`micro_batch_size`、`global_batch_size`。

## 引用

```bibtex
@misc{jiang2026wovr,
  title         = {{WoVR}: World Models as Reliable Simulators for Post-Training {VLA} Policies with {RL}},
  author        = {Zhennan Jiang and Shangqing Zhou and Yutong Jiang and Zefang Huang and Mingjie Wei and Yuhui Chen and Tianxing Zhou and Zhen Guo and Hao Lin and Quanlu Zhang and Yu Wang and Haoran Li and Chao Yu and Dongbin Zhao},
  year          = {2026},
  eprint        = {2602.13977},
  archivePrefix = {arXiv},
  primaryClass  = {cs.RO},
  doi           = {10.48550/arXiv.2602.13977},
  url           = {https://arxiv.org/abs/2602.13977}
}
```

## 许可证与归属

本项目使用 Apache License 2.0。它派生自 WoVR，而 WoVR 又基于 RLinf 源码快照初始化。
上游许可证和版权声明均已保留，详见 `LICENSE` 和 `UPSTREAM.md`。
