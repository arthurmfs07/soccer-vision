#!/usr/bin/env python3
import argparse
import sys
from pathlib import Path

import cv2
import numpy as np

from src.visual.field import PitchConfig, FieldVisualizer
from src.struct.utils import create_base_square
from src.utils import rf2my



    
def compute_homography(
        lbl_path: Path,
        model_pts: np.ndarray,
        cfg: PitchConfig) -> np.ndarray:
    """
    Reads YOLO-KP label (class,x,y,w,h,k0_x,k0_y,k0_v,...),
    builds src=[(x,y)] in [0..1], dst=[(X/L, Y/W)],
    fits H_norm via RANSAC, normalizes so H[2,2]==1.
    Returns None on failure.
    """
    vals = list(map(float, lbl_path.read_text().split()))
    kp   = np.array(vals[5:], dtype=np.float32).reshape(-1, 3)

    src_pts, dst_pts = [], []
    L, W = cfg.length, cfg.width

    for rf_idx, (xn, yn, vis) in enumerate(kp):
        if vis <= 0 or rf_idx not in rf2my:
            continue
        my_idx = rf2my[rf_idx]
        u, v = model_pts[my_idx]
        src_pts.append([xn, yn])
        dst_pts.append([ u,  v])

    src = np.asarray(src_pts, dtype=np.float32)
    dst = np.asarray(dst_pts, dtype=np.float32)

    if len(src) < 4:
        return None

    Hn, mask = cv2.findHomography(src, dst,
                                  method=cv2.RANSAC,
                                  ransacReprojThreshold=1e-3)
    if Hn is None:
        return None

    Hn = Hn.astype(np.float32)
    return Hn / Hn[2,2]


def healthcheck(
        H: np.ndarray,
        margin: float = 0.20
        ) -> bool:
    """Check (a) invertible, (b) base square corners stay in [0..1]^2."""
    # a) invertibility
    det = np.linalg.det(H)
    if abs(det) < 1e-6:
        return False

    # b) base square corners
    bs = create_base_square(as_tensor=False)       # shape (4,2) in [0..1]
    # cv2 wants shape (N,1,2)
    pts = bs.reshape(-1,1,2).astype(np.float32)
    warped = cv2.perspectiveTransform(pts, H).reshape(-1,2)

    if (warped[:,0] < -margin).any() or (warped[:,0] > 1.0 + margin).any():
        return False
    if (warped[:,1] < -margin).any() or (warped[:,1] > 1.0 + margin).any():
        return False
    return True


def clean_dataset(input_root: Path, output_root: Path) -> None:
    out = output_root
    out.mkdir(parents=True, exist_ok=True)

    # prepare canonical model points
    fv        = FieldVisualizer(PitchConfig())
    model_pts = fv._reference_model_pts()  # (33,2)
    cfg       = fv.cfg

    idx = 0
    total, passed = 0, 0

    for split in ("train","valid","test"):
        img_dir = input_root/split/"images"
        lbl_dir = input_root/split/"labels"

        for img_path in sorted(img_dir.glob("*.jpg")):
            total += 1
            lbl_path = lbl_dir/f"{img_path.stem}.txt"
            if not lbl_path.exists():
                print(f"⚠ skip {split}/{img_path.name}: no label")
                continue

            Hn = compute_homography(lbl_path, model_pts, cfg)
            if Hn is None:
                print(f"⚠ skip {split}/{img_path.name}: <4 pts or RANSAC fail>")
                continue

            good = healthcheck(Hn)
            status = "PASS" if good else "FAIL"
            print(f"[{status}] {split}/{img_path.name}", end="")

            if not good:
                print(" (healthcheck fail)")
                continue

            # OK: save
            passed += 1
            idx    += 1
            out_img = out/f"frame_{idx:06d}.png"
            out_H   = out/f"frame_{idx:06d}_H.npy"

            # copy image & write homography
            cv2.imwrite(str(out_img), cv2.imread(str(img_path)))
            np.save(str(out_H), Hn)
            print(" → saved")

    print(f"\nDone: processed {total} frames, passed={passed}, saved={idx} to {out}")


if __name__ == "__main__":
    p = argparse.ArgumentParser(
        description="Clean KP dataset → frames + normalized homographies"
    )
    p.add_argument("--input",  required=True,
                   help="root of raw YOLO-KP dataset (train/valid/test)")
    p.add_argument("--output", required=True,
                   help="where to write frame_XXXXXX.png + _H.npy")
    args = p.parse_args()

    inp = Path(args.input)
    out = Path(args.output)
    if not inp.exists():
        print("❌ input root not found:", inp)
        sys.exit(1)

    clean_dataset(inp, out)



# python3 -m src.data.clean_rf \
#     --input data/00--raw/football-field-detection.v15i.yolov5pytorch \
#     --output data/01--clean/roboflow