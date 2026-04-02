"""
Inference script — classify a single X-ray image or a directory of images.

Usage
-----
  python predict.py --image path/to/xray.jpg
  python predict.py --dir   path/to/folder/
  python predict.py --image path/to/xray.jpg --checkpoint checkpoints/best_osa.pth
"""

import os
import argparse
from PIL import Image

import torch
import torch.nn.functional as F

import config
from model   import load_checkpoint
from dataset import get_transforms


# ─── Single-image prediction ──────────────────────────────────────────────────

@torch.no_grad()
def predict_image(
    image_path:      str,
    model:           torch.nn.Module,
    device:          str,
    transform=None,
) -> dict:
    if transform is None:
        transform = get_transforms("test")

    img    = Image.open(image_path).convert("RGB")
    tensor = transform(img).unsqueeze(0).to(device)

    logits = model(tensor)
    probs  = F.softmax(logits, dim=1).squeeze(0)
    pred   = probs.argmax().item()

    return {
        "image":      os.path.basename(image_path),
        "prediction": config.CLASS_NAMES[pred],
        "confidence": round(probs[pred].item() * 100, 2),
        "probabilities": {
            cls: round(probs[i].item() * 100, 2)
            for i, cls in enumerate(config.CLASS_NAMES)
        },
    }


# ─── Batch prediction ─────────────────────────────────────────────────────────

def predict_directory(directory: str, model, device) -> list:
    EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff"}
    transform  = get_transforms("test")
    results    = []

    files = [
        os.path.join(directory, f)
        for f in os.listdir(directory)
        if os.path.splitext(f)[1].lower() in EXTENSIONS
    ]

    if not files:
        print(f"[WARN] No images found in {directory}")
        return results

    print(f"Processing {len(files)} images …")
    for path in files:
        try:
            result = predict_image(path, model, device, transform)
            results.append(result)
            print(
                f"  {result['image']:40s} → "
                f"{result['prediction']:12s} "
                f"({result['confidence']:.1f}%)"
            )
        except Exception as e:
            print(f"  [ERROR] {path}: {e}")

    return results


# ─── CLI entry point ──────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Bone Cancer X-Ray Inference")
    group  = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--image", type=str, help="Path to a single image")
    group.add_argument("--dir",   type=str, help="Path to a directory of images")
    parser.add_argument(
        "--checkpoint", type=str, default=None,
        help="Path to .pth checkpoint (auto-selects if omitted)"
    )
    parser.add_argument(
        "--device", type=str, default="cpu",
        choices=["cpu", "cuda", "mps"]
    )
    args = parser.parse_args()

    # Load model
    ckpt_path = args.checkpoint
    if ckpt_path is None:
        ckpts = [
            f for f in os.listdir(config.CHECKPOINT_DIR)
            if f.startswith("best_") and f.endswith(".pth")
        ]
        if not ckpts:
            raise FileNotFoundError("No checkpoint found.")
        ckpt_path = os.path.join(config.CHECKPOINT_DIR, sorted(ckpts)[-1])

    model = load_checkpoint(ckpt_path, device=args.device)
    model.eval()

    if args.image:
        result = predict_image(args.image, model, args.device)
        print(f"\nImage      : {result['image']}")
        print(f"Prediction : {result['prediction']}")
        print(f"Confidence : {result['confidence']}%")
        print("Probabilities:")
        for cls, prob in result["probabilities"].items():
            print(f"  {cls:12s}: {prob:.2f}%")

    else:
        predict_directory(args.dir, model, args.device)
