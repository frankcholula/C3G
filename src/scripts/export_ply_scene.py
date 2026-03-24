"""
Export a PLY file of 3D Gaussians for a single test scene from RealEstate10K.
Also saves context images, renders from context viewpoints, and camera poses.

Usage (2-view, default):
    python src/scripts/export_ply_scene.py

Usage (multiview):
    python src/scripts/export_ply_scene.py --multiview
    python src/scripts/export_ply_scene.py --multiview --num_views 8
    python src/scripts/export_ply_scene.py --multiview --scene 4ec2510baca79e6b

Output:
    outputs/ply_export/<scene_key>/
    ├── context/
    │   ├── 000000.png   # input view 0
    │   └── ...
    ├── render/
    │   ├── 000000.png   # Gaussian render from context view 0
    │   └── ...
    ├── cameras.json
    └── gaussians.ply
"""

import argparse
import json
import sys
from io import BytesIO
from pathlib import Path

import torch
import torchvision.transforms as tf
from einops import rearrange, repeat
from PIL import Image
from torchvision.utils import save_image

sys.path.insert(0, str(Path(__file__).parents[2]))

from src.misc.cam_utils import camera_normalization
from src.model.ply_export import export_ply

CHECKPOINTS = {
    "2view":     "checkpoints/gaussian_decoder.ckpt",
    "multiview": "checkpoints/gaussian_decoder_multiview.ckpt",
}
EVAL_CONFIGS = {
    "2view":     "re10k",
    "multiview": "re10k_multiview",
}


# ---------------------------------------------------------------------------
# Camera helpers
# ---------------------------------------------------------------------------

def convert_poses(poses: torch.Tensor):
    """Convert [N, 18] RE10K camera tensors → extrinsics [N,4,4], intrinsics [N,3,3]."""
    b = poses.shape[0]
    intrinsics = repeat(torch.eye(3), "h w -> b h w", b=b).clone()
    fx, fy, cx, cy = poses[:, :4].T
    intrinsics[:, 0, 0] = fx
    intrinsics[:, 1, 1] = fy
    intrinsics[:, 0, 2] = cx
    intrinsics[:, 1, 2] = cy

    w2c = repeat(torch.eye(4), "i j -> b i j", b=b).clone()
    w2c[:, :3] = rearrange(poses[:, 6:], "b (h w) -> b h w", h=3, w=4)
    return w2c.inverse(), intrinsics


def preprocess_cameras(extrinsics_all: torch.Tensor, context_indices: list[int]):
    """Apply make_baseline_1 + relative_pose, matching DatasetRE10k preprocessing."""
    # make_baseline_1: scale so baseline between first and last context camera = 1
    ctx_ext = extrinsics_all[context_indices]
    scale = (ctx_ext[0, :3, 3] - ctx_ext[-1, :3, 3]).norm()
    extrinsics_all = extrinsics_all.clone()
    extrinsics_all[:, :3, 3] /= scale

    # relative_pose: express all poses relative to the first context camera
    extrinsics_all = camera_normalization(extrinsics_all[context_indices][0:1], extrinsics_all)
    return extrinsics_all, scale.item()


# ---------------------------------------------------------------------------
# Scene loading
# ---------------------------------------------------------------------------

def load_scene(dataset_root: Path, scene_key: str, context_indices: list[int]):
    """Load and preprocess a scene from a .torch chunk file."""
    with open(dataset_root / "test" / "index.json") as f:
        chunk_index = json.load(f)

    chunk_path = dataset_root / "test" / chunk_index[scene_key]
    print(f"Loading chunk: {chunk_path.name}")

    chunk = torch.load(chunk_path, weights_only=False)
    scene_data = next(s for s in chunk if s["key"] == scene_key)

    to_tensor = tf.ToTensor()
    images = torch.stack([
        to_tensor(Image.open(BytesIO(scene_data["images"][i].numpy().tobytes())))
        for i in context_indices
    ])  # [V, 3, H, W]

    extrinsics_all, intrinsics_all = convert_poses(scene_data["cameras"])
    extrinsics_all, scale = preprocess_cameras(extrinsics_all, context_indices)
    print(f"Baseline scale: {scale:.4f}  |  context frames: {context_indices}")

    extrinsics = extrinsics_all[context_indices]   # [V, 4, 4]
    intrinsics = intrinsics_all[context_indices]   # [V, 3, 3]
    return images, extrinsics, intrinsics


def sample_context_indices(eval_index: dict, scene_key: str, num_views: int) -> list[int]:
    """
    Return context frame indices for a scene.
    - num_views=2  → use the canonical 2-view eval pair from the index
    - num_views>2  → evenly sample num_views frames from the full available range
    """
    entry = eval_index[scene_key]
    if num_views == 2:
        return entry["context"]  # canonical pair

    lo, hi = entry["context"][0], entry["context"][-1]
    step = max(1, (hi - lo) // (num_views - 1))
    indices = list(range(lo, hi + 1, step))[:num_views]
    # always include the canonical endpoints
    if indices[0] != lo:
        indices[0] = lo
    if indices[-1] != hi:
        indices[-1] = hi
    return indices


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

def load_encoder(checkpoint: str, eval_config: str, device: torch.device):
    from hydra import compose, initialize_config_dir
    from hydra.core.global_hydra import GlobalHydra
    from src.config import load_typed_root_config
    from src.model.encoder import get_encoder

    GlobalHydra.instance().clear()
    config_dir = str(Path(__file__).parents[2] / "config")
    with initialize_config_dir(config_dir=config_dir, version_base="1.3"):
        cfg = compose(
            config_name="main",
            overrides=[
                f"+evaluation={eval_config}",
                "mode=test",
                "wandb.mode=disabled",
                f"checkpointing.load={checkpoint}",
            ],
        )

    root_cfg = load_typed_root_config(cfg)
    root_cfg.model.encoder.feature_dim = 0
    encoder, _ = get_encoder(root_cfg.model.encoder)

    ckpt = torch.load(checkpoint, map_location="cpu", weights_only=False)
    if "state_dict" in ckpt:
        state = {k[8:]: v for k, v in ckpt["state_dict"].items() if k.startswith("encoder.")}
        encoder.load_state_dict(state, strict=False)
    elif "model" in ckpt:
        encoder.load_state_dict(ckpt["model"], strict=False)
    else:
        raise ValueError("Unknown checkpoint format.")

    return encoder.eval().to(device), root_cfg


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Export 3DGS PLY from a RE10K scene.")
    parser.add_argument("--scene",        type=str,  default=None,
                        help="Scene key (defaults to first in eval index)")
    parser.add_argument("--multiview",    action="store_true",
                        help="Use multiview checkpoint (gaussian_decoder_multiview.ckpt)")
    parser.add_argument("--num_views",    type=int,  default=None,
                        help="Number of context views (default: 2 for standard, 8 for multiview)")
    parser.add_argument("--checkpoint",   type=str,  default=None,
                        help="Override checkpoint path")
    parser.add_argument("--dataset_root", type=str,  default="/mnt/Data/re10k_pixelsplat")
    parser.add_argument("--index",        type=str,  default="assets/evaluation_index_re10k.json")
    parser.add_argument("--output",       type=str,  default="outputs/ply_export")
    args = parser.parse_args()

    mode = "multiview" if args.multiview else "2view"
    checkpoint = args.checkpoint or CHECKPOINTS[mode]
    eval_config = EVAL_CONFIGS[mode]
    num_views = args.num_views or (8 if args.multiview else 2)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # --- Scene selection ---
    with open(args.index) as f:
        eval_index = json.load(f)
    scene_key = args.scene or list(eval_index.keys())[0]
    assert scene_key in eval_index, f"Scene '{scene_key}' not in index."
    context_indices = sample_context_indices(eval_index, scene_key, num_views)
    print(f"Mode: {mode}  |  scene: {scene_key}  |  views: {num_views}")

    # --- Load scene ---
    images, extrinsics, intrinsics = load_scene(
        Path(args.dataset_root), scene_key, context_indices
    )
    v = images.shape[0]
    out_dir = Path(args.output) / f"{scene_key}_{mode}"

    # --- Save context images ---
    ctx_dir = out_dir / "context"
    ctx_dir.mkdir(parents=True, exist_ok=True)
    for i, img in enumerate(images):
        save_image(img, ctx_dir / f"{i:06d}.png")
    print(f"Saved {v} context images → {ctx_dir}")

    # --- Save camera poses ---
    cameras_data = [
        {
            "view": i,
            "frame_index": context_indices[i],
            "c2w": extrinsics[i].tolist(),
            "intrinsics": intrinsics[i].tolist(),
        }
        for i in range(v)
    ]
    cameras_path = out_dir / "cameras.json"
    with open(cameras_path, "w") as f:
        json.dump(cameras_data, f, indent=2)
    print(f"Saved cameras → {cameras_path}")

    # --- Load model ---
    print(f"Loading encoder from {checkpoint} ...")
    encoder, root_cfg = load_encoder(checkpoint, eval_config, device)
    print("Encoder ready.")

    # --- Build batch ---
    images_in = torch.nn.functional.interpolate(
        images, size=(224, 224), mode="bilinear", align_corners=False
    )
    images_in = images_in * 2 - 1  # [-1, 1]

    context = {
        "image":       images_in.unsqueeze(0).to(device),       # [1, V, 3, H, W]
        "extrinsics":  extrinsics.unsqueeze(0).to(device),      # [1, V, 4, 4]
        "intrinsics":  intrinsics.unsqueeze(0).to(device),      # [1, V, 3, 3]
        "near":        torch.full((1, v), 0.1, device=device),
        "far":         torch.full((1, v), 100.0, device=device),
        "index":       torch.arange(v).unsqueeze(0).to(device),
        "overlap":     torch.tensor([0.4]).to(device),
    }

    # --- Run encoder ---
    dump = {}
    with torch.no_grad():
        gaussians = encoder(context, global_step=0, visualization_dump=dump)
    print(f"Generated {gaussians.means.shape[1]} Gaussians.")

    # --- Export PLY ---
    ply_path = out_dir / "gaussians.ply"
    export_ply(
        means=gaussians.means[0],
        scales=dump["scales"][0],
        rotations=dump["rotations"][0],
        harmonics=gaussians.harmonics[0],
        opacities=gaussians.opacities[0],
        path=ply_path,
        shift_and_scale=True,
    )
    print(f"Saved PLY → {ply_path}")

    # --- Render from context cameras ---
    from src.model.decoder import get_decoder

    decoder = get_decoder(root_cfg.model.decoder).eval().to(device)
    h, w = images.shape[2], images.shape[3]
    render_dir = out_dir / "render"
    render_dir.mkdir(parents=True, exist_ok=True)

    with torch.no_grad():
        output = decoder.forward(
            gaussians,
            extrinsics.unsqueeze(0).to(device),
            intrinsics.unsqueeze(0).to(device),
            torch.full((1, v), 0.1, device=device),
            torch.full((1, v), 100.0, device=device),
            (h, w),
        )

    for i, color in enumerate(output.color[0]):
        save_image(color, render_dir / f"{i:06d}.png")
    print(f"Saved {v} renders → {render_dir}")
    print(f"\nOpen {ply_path} in SuperSplat: https://playcanvas.com/supersplat/editor")


if __name__ == "__main__":
    main()
