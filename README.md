# WoVR-Improve

RL post-training experiments for Vision-Language-Action (VLA) policies, built on
[WoVR](https://github.com/DRL-CASIA/WoVR). This repository starts from the official
WoVR snapshot and adds reliability fixes, phase-level profiling, CPU-memory
management, standalone LIBERO evaluation with videos, and reproducible launch
scripts.

Paper: [WoVR: World Models as Reliable Simulators for Post-Training VLA Policies with RL](https://arxiv.org/abs/2602.13977)

## Relationship to upstream projects

This project depends on several upstream repositories and released checkpoints:

| Component | Source | Notes |
| --- | --- | --- |
| WoVR (base) | https://github.com/DRL-CASIA/WoVR | Official implementation; this repo is derived from it |
| RLinf (framework) | https://github.com/RLinf/RLinf | Base RL framework; WoVR vendors an RLinf snapshot (see `UPSTREAM.md`) |
| World-model training | https://github.com/youmo445/diffsynth-studio-rlinf | DiffSynth-Studio fork used for the `diffsynth` runtime (`6d9ce59`) |
| Policy | https://github.com/moojink/openvla-oft | OpenVLA-OFT source, installed editable (`e4287e9`) |
| Simulation | https://github.com/Lifelong-Robot-Learning/LIBERO | LIBERO benchmark |
| Video backbone | https://github.com/Wan-Video/Wan2.2 | Wan2.2-TI2V-5B |
| Policy checkpoint | https://huggingface.co/Haozhan72/Openvla-oft-SFT-libero-spatial-traj1 | LIBERO-Spatial SFT policy |
| World model | https://huggingface.co/RLinf/RLinf-Wan-LIBERO-Spatial | Wan world model + VAE + reward model + KIR data |

## Changes relative to upstream

- Fix `get_env_attr` for gymnasium 0.29 so the Wan env `offload` method is found.
- Add `[profile]` phase timings (`rollout`, `actor_train`, `sync_weights`, ...).
- Release CPU memory between steps (`gc.collect()` + `malloc_trim`).
- Add a standalone LIBERO-Spatial evaluation config with video recording.
- Add FSDP checkpoint to Hugging Face checkpoint conversion.
- Consolidate launch/eval/setup scripts under `scripts/`.

## Environment

Reference setup used in our experiments:

- 8x NVIDIA A800 80GB, 400GB host RAM
- torch 2.11.0 + cu128, flash-attn 2.8.3 (cu12), transformers 4.40.1 (OpenVLA-OFT fork)
- `actor` and `rollout` resident on GPU; `env` (Wan world model) offloaded to CPU
- MuJoCo rendered with `MUJOCO_GL=osmesa` in headless workers

Install the environment once:

```bash
bash scripts/setup_env.sh
```

Model checkpoints are downloaded from Hugging Face; see the table above.
Adjust the absolute paths in `examples/embodiment/config/wan_libero_spatial_grpo_openvlaoft.yaml`
to your checkout and checkpoint directories.

## Quick start

```bash
# request a GPU worker (cluster-specific launcher)
bash scripts/start.sh

# inside the worker
bash scripts/run_wovr.sh

# or a bounded run
bash scripts/run.sh        # STEPS=20 wrapper
```

Launching is cluster-specific (`rlaunch`); adapt `scripts/start.sh` for your scheduler.

## Standalone evaluation

`scripts/eval_video.sh` converts the FSDP checkpoint to Hugging Face format, runs
LIBERO-Spatial evaluation, and saves videos:

```bash
bash scripts/eval_video.sh
```

Outputs:

- metrics: `eval_results/step20/`
- videos: `eval_results/step20/video/eval/`

Override paths with `CKPT`, `HF`, `OUT`, and `ENVS` environment variables.

## Checkpoint conversion

`scripts/convert_ckpt_to_hf.py` converts a training checkpoint
(`actor/model_state_dict/full_weights.pt`) into a Hugging Face checkpoint using the
configuration, tokenizer, and processor files from the base SFT checkpoint.

## Notes

- The local environment in our cluster installs `diffsynth` from
  `youmo445/diffsynth-studio-rlinf` because cloning `RLinf/diffsynth-studio` was
  unreliable on the cluster network.
- `requirements/install.sh` may rewrite `pyproject.toml` to pin CUDA 12.8 wheels;
  this is environment-specific.
- Training defaults are not a minimal-memory preset. Adapt `total_num_envs`,
  `rollout_epoch`, `micro_batch_size`, and `global_batch_size` to your hardware.

## Citation

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

## License and attribution

This project is distributed under the Apache License 2.0. It is derived from WoVR,
which is itself initialized from an RLinf source snapshot. Upstream licenses and
copyright notices are preserved; see `LICENSE` and `UPSTREAM.md`.
