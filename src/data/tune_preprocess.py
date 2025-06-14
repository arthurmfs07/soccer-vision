#!/usr/bin/env python3
# scripts/tune_preprocess.py
#
# Interactive tuner for a global pre-processing pipeline.
# Pure OpenCV – no Albumentations.

import cv2, random, yaml, argparse
import numpy as np
from pathlib import Path

# ─────────────────────────  parameters & order  ────────────
PARAMS = {
    "warp_alpha": 0.03,   # 0→none, 0.15≈wild TV tilt
    "zoom_min"  : 1.00,
    "zoom_max"  : 1.40,
    "rot_deg"   : 3.0,    # camera roll ±deg
    "bri_lim"   : 0.25,   # brightness ±fraction
    "con_lim"   : 0.25,   # contrast   ±fraction
    "hue_shift" : 15,     # HSV hue   ±deg
    "sat_shift" : 25,     # HSV sat   ±%
    "val_shift" : 15,     # HSV value ±%
    "jpeg_low"  : 40,     # worst JPEG quality
    "jpeg_high" : 95,     # best  JPEG quality
    "gray_p"    : 0.10,   # chance to convert to gray
}
ORDER = list(PARAMS.keys())

# ─────────────────────────  CV-only pipeline  ───────────────
def build_pipeline_cv(p):
    def fn(img):
        h, w = img.shape[:2]
        out  = img.copy()

        # 1. perspective warp
        if p["warp_alpha"] > 0:
            src = np.float32([[0,0],[w,0],[w,h],[0,h]])
            jit = p["warp_alpha"] * np.float32([[w,h]])
            dst = src + np.random.uniform(-jit, jit, src.shape).astype(np.float32)
            H, _ = cv2.findHomography(src, dst)
            out  = cv2.warpPerspective(out, H, (w, h),
                                       flags=cv2.INTER_LINEAR,
                                       borderMode=cv2.BORDER_REFLECT)

        # 2. rotation
        if p["rot_deg"] > 0:
            ang = np.random.uniform(-p["rot_deg"], p["rot_deg"])
            M   = cv2.getRotationMatrix2D((w/2, h/2), ang, 1.0)
            out = cv2.warpAffine(out, M, (w, h),
                                 flags=cv2.INTER_LINEAR,
                                 borderMode=cv2.BORDER_REFLECT)

        # 3. zoom crop
        if p["zoom_max"] > 1.0:
            z  = np.random.uniform(p["zoom_min"], p["zoom_max"])
            nw, nh = int(w / z), int(h / z)
            x0 = np.random.randint(0, w - nw + 1)
            y0 = np.random.randint(0, h - nh + 1)
            out = cv2.resize(out[y0:y0+nh, x0:x0+nw], (w, h),
                             interpolation=cv2.INTER_LINEAR)

        # 4. brightness / contrast
        bri = np.random.uniform(-p["bri_lim"], p["bri_lim"])
        con = np.random.uniform(-p["con_lim"], p["con_lim"])
        out = cv2.convertScaleAbs(out, alpha=1.0 + con, beta=255 * bri)

        # 5. HSV shifts
        hsv = cv2.cvtColor(out, cv2.COLOR_RGB2HSV).astype(np.int16)
        hsv[..., 0] = (hsv[..., 0] + np.random.randint(-p["hue_shift"],
                                                       p["hue_shift"] + 1)) % 180
        hsv[..., 1] = np.clip(hsv[..., 1] +
                              np.random.randint(-p["sat_shift"],
                                                p["sat_shift"] + 1), 0, 255)
        hsv[..., 2] = np.clip(hsv[..., 2] +
                              np.random.randint(-p["val_shift"],
                                                p["val_shift"] + 1), 0, 255)
        out = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2RGB)

        # 6. optional grayscale
        if np.random.rand() < p["gray_p"]:
            g = cv2.cvtColor(out, cv2.COLOR_RGB2GRAY)
            out = cv2.cvtColor(g, cv2.COLOR_GRAY2RGB)

        # 7. JPEG artefacts
        qual = np.random.randint(p["jpeg_low"], p["jpeg_high"] + 1)
        ok, enc = cv2.imencode(".jpg",
                               cv2.cvtColor(out, cv2.COLOR_RGB2BGR),
                               [cv2.IMWRITE_JPEG_QUALITY, qual])
        out = cv2.cvtColor(cv2.imdecode(enc, cv2.IMREAD_COLOR),
                           cv2.COLOR_BGR2RGB)
        return out
    return fn

# ─────────────────────────  combined mosaic  ───────────────
def make_canvas(train, infer, cols=3):
    """Return one RGB mosaic; resizes every tile to the first image size."""
    imgs = train + infer
    h0, w0 = imgs[0].shape[:2]           # reference tile size
    res = [cv2.resize(im, (w0, h0)) for im in imgs]

    rows = (len(res) + cols - 1) // cols
    canvas = np.zeros((rows * h0, cols * w0, 3), np.uint8)

    for i, im in enumerate(res):
        r, c = divmod(i, cols)
        canvas[r*h0:(r+1)*h0, c*w0:(c+1)*w0] = cv2.cvtColor(im, cv2.COLOR_RGB2BGR)
    return canvas

# ─────────────────────────  interactive wizard  ────────────
def wizard(train_imgs, infer_imgs):
    while True:
        pipe = build_pipeline_cv(PARAMS)
        canvas = make_canvas([pipe(im) for im in train_imgs],
                             [pipe(im) for im in infer_imgs])
        cv2.imshow("preprocess-tuner  |  top-left = train, bottom-right = infer",
                   cv2.resize(canvas, None, fx=0.45, fy=0.45))
        cv2.waitKey(1)  # draw

        # interactive prompt
        print("\nCurrent parameters:")
        for k in ORDER:
            print(f"  {k:10s}: {PARAMS[k]}")
        print("\nType 'name value'  (e.g.  warp_alpha 0.06 )"
              "\n      b   → undo last change"
              "\n      q   → save & quit\n")

        txt = input("> ").strip()
        if not txt:
            continue
        if txt.lower() in ("q", "quit"):
            break
        if txt.lower() in ("b", "back"):
            # naive undo: reload last saved yaml if exists
            yp = Path("preprocess_tuned.yaml")
            if yp.exists():
                PARAMS.update(yaml.safe_load(yp.read_text()))
                print("↩️  reloaded last saved parameters.")
            continue

        parts = txt.split(maxsplit=1)
        if len(parts) != 2 or parts[0] not in PARAMS:
            print("⚠  format:  <param_name> <number>")
            continue
        k, v = parts
        try:
            PARAMS[k] = float(v) if '.' in v else int(v)
        except ValueError:
            print("⚠  not a number.")
            continue

    cv2.destroyAllWindows()
    Path("preprocess_tuned.yaml").write_text(yaml.dump(PARAMS))
    print("✅  saved parameters to preprocess_tuned.yaml")

# ─────────────────────────  util: sample images  ───────────
def sample_images(folder: Path, k: int):
    paths = [p for p in folder.iterdir() if p.suffix.lower() in (".jpg", ".png")]
    if len(paths) < k:
        raise RuntimeError(f"Only {len(paths)} images in {folder}")
    chosen = random.sample(paths, k)
    return [cv2.cvtColor(cv2.imread(str(p)), cv2.COLOR_BGR2RGB) for p in chosen]

# ─────────────────────────  CLI  ───────────────────────────
if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--train",  required=True, help="folder with training images")
    ap.add_argument("--infer",  required=True, help="folder with inference frames")
    ap.add_argument("--ktrain", type=int, default=6)
    ap.add_argument("--kinfer", type=int, default=3)
    args = ap.parse_args()

    train_imgs  = sample_images(Path(args.train),  args.ktrain)
    infer_imgs  = sample_images(Path(args.infer),  args.kinfer)
    wizard(train_imgs, infer_imgs)


# python3 -m src.data.tune_preprocess \
# --train  data/00--raw/football-field-detection.v15i.yolov5pytorch/train/images \
# --infer  data/00--raw/frames/match_3895113 \
# --ktrain 6 --kinfer 3 \