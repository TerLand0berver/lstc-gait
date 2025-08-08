import argparse
from pathlib import Path
import numpy as np
from PIL import Image
import random


def make_seq(dir_path: Path, frames: int, h: int, w: int, subject_id: int, seed: int):
    rng = random.Random(seed)
    dir_path.mkdir(parents=True, exist_ok=True)
    # Simple moving bar pattern; subject-dependent thickness/offset
    bar_thickness = 4 + (subject_id % 3) * 2
    phase = (subject_id * 7) % w
    for t in range(frames):
        img = np.zeros((h, w), dtype=np.uint8)
        # Vertical bar moving horizontally
        x0 = (phase + t * 2) % w
        x1 = min(w, x0 + bar_thickness)
        img[:, x0:x1] = 255
        # Add a horizontal stripe to emulate limb motion
        y0 = (t * 3) % h
        y1 = min(h, y0 + 2)
        img[y0:y1, :] = np.maximum(img[y0:y1, :], 180)
        Image.fromarray(img, mode="L").save(dir_path / f"{t+1:06d}.png")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=str, required=True)
    ap.add_argument("--subjects", type=int, default=3)
    ap.add_argument("--seq-per-subject", type=int, default=2)
    ap.add_argument("--frames", type=int, default=20)
    ap.add_argument("--height", type=int, default=64)
    ap.add_argument("--width", type=int, default=44)
    ap.add_argument("--seed", type=int, default=123)
    args = ap.parse_args()

    root = Path(args.out)
    root.mkdir(parents=True, exist_ok=True)

    rng = random.Random(args.seed)
    for s in range(1, args.subjects + 1):
        subj = root / f"subject_{s:04d}"
        for q in range(1, args.seq_per_subject + 1):
            seq_dir = subj / f"seq_{q:04d}"
            make_seq(seq_dir, frames=args.frames, h=args.height, w=args.width, subject_id=s, seed=rng.randint(0, 10_000))

    print(f"Toy dataset created at: {root}")


if __name__ == "__main__":
    main()
