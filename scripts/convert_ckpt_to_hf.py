#!/usr/bin/env python3
"""Convert an RLinf FSDP checkpoint (full_weights.pt) into a HF checkpoint dir."""
import argparse
import os
import shutil

import torch
from safetensors.torch import save_file


def is_weight_file(name: str) -> bool:
    return (
        name.endswith(".safetensors")
        or name.endswith(".safetensors.index.json")
        or name.endswith(".bin")
        or name.endswith(".pt")
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True, help="dir like .../checkpoints/global_step_20")
    ap.add_argument(
        "--base",
        default="/data/wovr/code/WoVR/checkpoints/Openvla-oft-SFT-libero-spatial-traj1",
        help="HF checkpoint whose config/tokenizer/processor files are reused",
    )
    ap.add_argument("--out", default="/data/wovr/checkpoints/WoVR-step20-hf")
    args = ap.parse_args()

    src = os.path.join(args.ckpt, "actor", "model_state_dict", "full_weights.pt")
    if not os.path.isfile(src):
        raise FileNotFoundError(src)

    os.makedirs(args.out, exist_ok=True)
    for name in os.listdir(args.base):
        if is_weight_file(name):
            continue
        s = os.path.join(args.base, name)
        d = os.path.join(args.out, name)
        if os.path.isdir(s):
            shutil.copytree(s, d, dirs_exist_ok=True)
        else:
            shutil.copy2(s, d)

    print(f"loading {src}")
    state = torch.load(src, map_location="cpu", mmap=True, weights_only=True)
    state = {k: v.contiguous() for k, v in state.items()}
    out_file = os.path.join(args.out, "model.safetensors")
    save_file(state, out_file, metadata={"format": "pt"})
    print(f"saved {len(state)} tensors -> {out_file}")
    print("done")


if __name__ == "__main__":
    main()
