# WoVR

## World Models as Reliable Simulators for Post-Training VLA Policies with RL

**Official implementation of WoVR.** This repository provides the reinforcement learning pipeline for post-training Vision-Language-Action (VLA) policies with learned world models, built on [RLinf](https://github.com/RLinf/RLinf).

[Paper](https://arxiv.org/abs/2602.13977) · [Project Page](https://wovr-corl.github.io/) · [World Model Training](https://github.com/youmo445/diffsynth-studio-rlinf) · [Detailed Wan Guide](https://rlinf.readthedocs.io/en/latest/rst_source/examples/embodied/wan.html)

WoVR addresses unreliable imagined rollouts through an action-conditioned video world model, **Keyframe-Initialized Rollouts (KIR)**, and **World Model-Policy co-evolution**. The policy learns from interactions with the world model, while downstream evaluation measures performance in the target environment.

This repository hosts policy RL and world-model environment integration. Training and fine-tuning the world model are covered by the separate [World Model training repository](https://github.com/youmo445/diffsynth-studio-rlinf).

> **Release scope:** The Quick Start below runs OpenVLA-OFT + GRPO with a **frozen Wan world model** and KIR on LIBERO. It does not automatically execute the full co-evolution loop. To perform co-evolution, collect additional trajectories with the updated policy, retrain the world model in the separate training repository, and update the simulator checkpoints between RL stages.

## Quick Start

The example below uses **LIBERO-Spatial**. Run the commands in **Bash on a Linux GPU machine** with compatible NVIDIA drivers. Docker requires the NVIDIA Container Toolkit. Native Windows training is not covered by this recipe. Model downloads require substantial disk space; GPU memory requirements depend on batch sizes and environment parallelism. See the [installation requirements](https://rlinf.readthedocs.io/en/latest/rst_source/start/installation.html) for platform setup.

### 1. Clone WoVR and install dependencies

```bash
git clone https://github.com/DRL-CASIA/WoVR.git
cd WoVR
```

Choose one installation method.

**Option A: Docker**

```bash
docker run -it --rm --gpus all \
  --shm-size 20g \
  --network host \
  --name wovr \
  -v "$(pwd):/workspace/WoVR" \
  -w /workspace/WoVR \
  rlinf/rlinf:agentic-rlinf0.4-wan

# Run the remaining steps inside the container.
source switch_env openvla-oft
cd /workspace/WoVR
```

**Option B: Local environment**

```bash
bash requirements/install.sh embodied --model openvla-oft --env wan
source .venv/bin/activate
```

The `wan` installation bundle installs the world-model dependencies and the LIBERO environment used for evaluation. For faster dependency downloads in mainland China, the installer also supports `--use-mirror`.

### 2. Download the policy and simulator assets

From the repository root, in the selected environment:

```bash
python -m pip install huggingface-hub
mkdir -p checkpoints

hf download Haozhan72/Openvla-oft-SFT-libero-spatial-traj1 \
  --local-dir checkpoints/Openvla-oft-SFT-libero-spatial-traj1

hf download RLinf/RLinf-Wan-LIBERO-Spatial \
  --local-dir checkpoints/RLinf-Wan-LIBERO-Spatial

# Print the absolute directory to use in the configuration below.
realpath checkpoints
```

The Wan download includes the initialization data, world-model weights, VAE, and reward model:

```text
checkpoints/RLinf-Wan-LIBERO-Spatial/
├── dataset/
│   ├── traj0.npy
│   ├── traj0_kir.npy
│   └── ...
├── model-00001.safetensors
├── Wan2.2_VAE.pth
└── resnet_rm.pth
```

Keep the initialization dataset: the learned simulator needs recorded observations and task context to initialize its rollouts. Files with `_kir` supply context for keyframe initialization. Downloaded checkpoints are excluded from Git by this repository's `.gitignore`.

### 3. Set checkpoint paths

Edit [wan_libero_spatial_grpo_openvlaoft.yaml](examples/embodiment/config/wan_libero_spatial_grpo_openvlaoft.yaml). **Merge the fields below into the existing sections; do not replace the full config or add duplicate YAML keys.** Replace `/absolute/path/to/WoVR` with your checkout path. Inside the Docker container above, it is `/workspace/WoVR`.

```yaml
actor:
  model:
    model_path: /absolute/path/to/WoVR/checkpoints/Openvla-oft-SFT-libero-spatial-traj1
    unnorm_key: libero_spatial_no_noops
    use_proprio: False
    num_images_in_input: 1
    num_action_chunks: 8

rollout:
  model:
    model_path: /absolute/path/to/WoVR/checkpoints/Openvla-oft-SFT-libero-spatial-traj1

env:
  train:
    wan_wm_hf_ckpt_path: /absolute/path/to/WoVR/checkpoints/RLinf-Wan-LIBERO-Spatial
    enable_kir: True
    num_inference_steps: 5
    enable_offload: True
```

Both policy paths must point to the same SFT checkpoint. The world model supplies one camera view without proprioception, and its action chunk length is eight; keep the matching policy settings above.

The inherited [Wan environment config](examples/embodiment/config/env/wan_libero_spatial.yaml) resolves the dataset, VAE, world-model, and reward-model paths from `wan_wm_hf_ckpt_path`. Keep `reset_gripper_open: True` for both training and evaluation. KIR is enabled by default; setting `env.train.enable_kir: False` excludes `_kir` initialization files.

### 4. Launch policy RL

```bash
bash examples/embodiment/run_embodiment.sh wan_libero_spatial_grpo_openvlaoft
```

The example uses a single node and colocates actor, rollout, and environment workers on the available GPUs. Before launching, adapt `cluster.component_placement`, `env.train.total_num_envs`, `actor.micro_batch_size`, and `actor.global_batch_size` to your hardware. Preserve the GRPO grouping and worker divisibility constraints when changing parallelism; the default config is not a minimal-memory preset.

Runs write logs and checkpoints under `logs/<timestamp>-wan_libero_spatial_grpo_openvlaoft/`. Monitor `env/success_once` for training success estimated by the learned simulator, and use LIBERO evaluation to assess actual policy performance.

```bash
tensorboard --logdir logs
```

The config selects **Wan for training** and **LIBERO for evaluation**. Periodic evaluation is disabled by default (`runner.val_check_interval: -1`); set a positive interval, such as `5`, to enable it. Ensure `env.eval.total_num_envs` is divisible by the number of evaluation environment workers. Evaluation videos are controlled by `env.eval.video_cfg.save_video` and saved under the run directory's `video/eval/` when evaluation runs. The default checkpoint interval is five epochs; set `runner.resume_dir` to a saved `checkpoints/global_step_<N>/` directory to resume.

### Other LIBERO suites

Use the matching policy checkpoint, Wan assets, configuration, and action normalization key for each suite:

| Suite | Policy checkpoint | World model and initialization data | RL config | `actor.model.unnorm_key` |
| --- | --- | --- | --- | --- |
| Spatial | [OpenVLA-OFT Spatial](https://huggingface.co/Haozhan72/Openvla-oft-SFT-libero-spatial-traj1) | [Wan Spatial](https://huggingface.co/RLinf/RLinf-Wan-LIBERO-Spatial) | [Spatial config](examples/embodiment/config/wan_libero_spatial_grpo_openvlaoft.yaml) | `libero_spatial_no_noops` |
| Object | [OpenVLA-OFT Object](https://huggingface.co/Haozhan72/Openvla-oft-SFT-libero-object-traj1) | [Wan Object](https://huggingface.co/RLinf/RLinf-Wan-LIBERO-Object) | [Object config](examples/embodiment/config/wan_libero_object_grpo_openvlaoft.yaml) | `libero_object_no_noops` |
| Goal | [OpenVLA-OFT Goal](https://huggingface.co/Haozhan72/Openvla-oft-SFT-libero-goal-traj1) | [Wan Goal](https://huggingface.co/RLinf/RLinf-Wan-LIBERO-Goal) | [Goal config](examples/embodiment/config/wan_libero_goal_grpo_openvlaoft.yaml) | `libero_goal_no_noops` |

The Goal suite uses `TaskEmbedResnetRewModel` with `taskemb_resnet_rm.pth`; Spatial and Object use `ResnetRewModel` with `resnet_rm.pth`. The respective environment configs already select the correct reward model.

After downloading the matching assets and updating all three checkpoint paths in the selected config, launch one of:

```bash
bash examples/embodiment/run_embodiment.sh wan_libero_object_grpo_openvlaoft
# Or:
bash examples/embodiment/run_embodiment.sh wan_libero_goal_grpo_openvlaoft
```

## World Model Training

For dataset preparation and training or fine-tuning the action-conditioned Wan world model, use **[youmo445/diffsynth-studio-rlinf](https://github.com/youmo445/diffsynth-studio-rlinf)** and follow its installation and training instructions in a separate environment.

This repository's installer retains RLinf's [DiffSynth-Studio runtime](https://github.com/RLinf/diffsynth-studio) for the released RL recipe. When using newly trained weights, ensure their architecture and checkpoint format match that runtime. Set `env.train.model_path`, `env.train.VAE_path`, `env.train.initial_image_path`, and `env.train.reward_model.from_pretrained` explicitly if the exported assets do not follow the bundled directory layout.

## Resources

- [WoVR paper on arXiv](https://arxiv.org/abs/2602.13977)
- [Project page and videos](https://wovr-corl.github.io/)
- [World Model training repository](https://github.com/youmo445/diffsynth-studio-rlinf)
- [Wan world-model RL guide](https://rlinf.readthedocs.io/en/latest/rst_source/examples/embodied/wan.html)
- [RLinf installation guide](https://rlinf.readthedocs.io/en/latest/rst_source/start/installation.html)
- [RLinf infrastructure](https://github.com/RLinf/RLinf)

## Acknowledgement

This work was supported by the Beijing Major Science and Technology Project under Contract No.Z251100008125023 and the Beijing Academyof Artificial Intelligence (BAAI).

We thank the contributors to [RLinf](https://github.com/RLinf/RLinf), [DiffSynth-Studio](https://github.com/modelscope/DiffSynth-Studio), [Wan](https://github.com/Wan-Video/Wan2.2), [OpenVLA-OFT](https://github.com/moojink/openvla-oft), and [LIBERO](https://github.com/Lifelong-Robot-Learning/LIBERO) for their open-source work.

## License and Upstream Attribution

WoVR is distributed under the [Apache License 2.0](LICENSE). This is an independent repository initialized from an RLinf source snapshot; the upstream license and source copyright notices are preserved. The `rlinf` Python package name and upstream framework documentation remain for compatibility. See [UPSTREAM.md](UPSTREAM.md) for the exact source revision and the changes made for this release. External models, datasets, and dependencies retain their respective licenses.

## Citation

If you use WoVR in your research, please cite:

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
