"""Evaluate relighting results: did the LIGHT change while the BUILDING stayed the same?

Folder layout:
    inputs/<image>.jpg|png                       original photos / renders
    outputs/<option>/<image>__<variant>.png      results, e.g. outputs/sun_direction/exterior_01__sun_left_low.png
    results/runs.csv                             filled by hand: time, cost, notes, 1-5 scores (see below)

Run: python relight_eval.py
Output:
    results/metrics.csv          one row per result
    results/grids/<image>.jpg    before | after per option, with a light-change heatmap under each result

Two automatic axes:
    structure  edge_precision / edge_recall / edge_f1   (Canny on blurred grayscale, 3 px tolerance)
               precision drops when the model INVENTS lines (new windows, a regenerated building),
               recall drops when it LOSES lines.
    light      light_change = 1 - correlation of heavily blurred luminance (input vs output).
               A colour grade or global brightening keeps the spatial pattern -> ~0.
               Moved shadows / a new sun direction change the pattern -> clearly > 0.
               mean_lum_delta = average brightness change (-1..1), e.g. night < 0.
    crop       outputs with a different aspect ratio are compared on the centre crop of the input;
    reframing  some tools silently zoom/shift the frame. The input is aligned to the output first
               (ORB features + similarity transform); zoom > 1 means the tool zoomed in.
               frame_kept = share of the original frame that is still in the output
               (aspect-ratio crop and zoom combined; 1.0 = nothing lost).
"""
import glob
import os

import cv2
import numpy as np
import pandas as pd
from PIL import Image, ImageDraw

TILE_H = 320


def load(path):
    return np.array(Image.open(path).convert("RGB"))


def match_aspect(inp, out):
    """Centre-crop the input to the output's aspect ratio, then resize to the output size."""
    ih, iw = inp.shape[:2]
    oh, ow = out.shape[:2]
    kept = 1.0
    if abs(iw / ih - ow / oh) > 0.02:
        if iw / ih > ow / oh:          # input wider -> crop width
            nw = int(ih * ow / oh)
            x = (iw - nw) // 2
            inp = inp[:, x:x + nw]
            kept = nw / iw
        else:                          # input taller -> crop height
            nh = int(iw * oh / ow)
            y = (ih - nh) // 2
            inp = inp[y:y + nh]
            kept = nh / ih
    inp = cv2.resize(inp, (ow, oh), interpolation=cv2.INTER_AREA)
    return inp, round(kept, 3)


def align(inp, out):
    """Some tools silently zoom or shift the frame. Find the scale + shift that maps the
    input onto the output (ORB features, similarity transform) and warp the input.
    Returns the aligned input, the fraction of the output covered by it, and the zoom factor."""
    g1 = cv2.cvtColor(inp, cv2.COLOR_RGB2GRAY)
    g2 = cv2.cvtColor(out, cv2.COLOR_RGB2GRAY)
    orb = cv2.ORB_create(4000)
    k1, d1 = orb.detectAndCompute(g1, None)
    k2, d2 = orb.detectAndCompute(g2, None)
    if d1 is None or d2 is None:
        return inp, 1.0, 1.0
    matches = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True).match(d1, d2)
    if len(matches) < 30:
        return inp, 1.0, 1.0
    src = np.float32([k1[m.queryIdx].pt for m in matches])
    dst = np.float32([k2[m.trainIdx].pt for m in matches])
    M, inliers = cv2.estimateAffinePartial2D(src, dst, method=cv2.RANSAC, ransacReprojThreshold=4)
    if M is None or inliers.sum() < 20:
        return inp, 1.0, 1.0
    zoom = float(np.hypot(M[0, 0], M[1, 0]))
    shift = float(np.hypot(M[0, 2], M[1, 2]))
    if abs(zoom - 1) < 0.01 and shift < 2:          # already aligned
        return inp, 1.0, 1.0
    h, w = out.shape[:2]
    warped = cv2.warpAffine(inp, M, (w, h), borderMode=cv2.BORDER_REPLICATE)
    mask = cv2.warpAffine(np.ones(inp.shape[:2], np.uint8), M, (w, h))
    return warped, round(float(mask.mean()), 3), round(zoom, 3)


def edges(rgb):
    g = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    g = cv2.GaussianBlur(g, (5, 5), 0)
    return cv2.Canny(g, 50, 150) > 0


def edge_scores(a, b, tol=3):
    k = np.ones((2 * tol + 1, 2 * tol + 1), np.uint8)
    a_d = cv2.dilate(a.astype(np.uint8), k) > 0
    b_d = cv2.dilate(b.astype(np.uint8), k) > 0
    p = (b & a_d).sum() / max(b.sum(), 1)
    r = (a & b_d).sum() / max(a.sum(), 1)
    f1 = 2 * p * r / max(p + r, 1e-9)
    return round(f1, 3), round(p, 3), round(r, 3)


def luminance(rgb):
    return cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB)[:, :, 0].astype(np.float32) / 255.0


def light_scores(a, b):
    la, lb = luminance(a), luminance(b)
    sigma = max(a.shape[:2]) / 60           # keep shadows and light patches, drop texture
    ba = cv2.GaussianBlur(la, (0, 0), sigma)
    bb = cv2.GaussianBlur(lb, (0, 0), sigma)
    corr = np.corrcoef(ba.ravel(), bb.ravel())[0, 1]
    # heatmap of where the lighting changed, after removing the global brightness shift
    diff = (bb - bb.mean()) - (ba - ba.mean())
    return round(1 - corr, 3), round(float(lb.mean() - la.mean()), 3), diff


def heatmap(diff):
    """Blue = got darker than before (relative), red = got brighter."""
    d = np.clip(diff / 0.25, -1, 1)
    img = np.zeros(d.shape + (3,), np.uint8)
    img[..., 0] = (np.clip(d, 0, 1) * 255).astype(np.uint8)
    img[..., 2] = (np.clip(-d, 0, 1) * 255).astype(np.uint8)
    img[..., 1] = (40 * (1 - np.abs(d))).astype(np.uint8)
    return img


def tile(rgb, label):
    im = Image.fromarray(rgb)
    im = im.resize((int(im.width * TILE_H / im.height), TILE_H))
    d = ImageDraw.Draw(im)
    d.rectangle([0, 0, 8 + 7 * len(label), 18], fill="black")
    d.text((4, 3), label, fill="white")
    return im


def main():
    rows, grids = [], {}
    for out_path in sorted(glob.glob("outputs/*/*.*")):
        if not out_path.lower().endswith((".png", ".jpg", ".jpeg", ".webp")):
            continue
        option = os.path.basename(os.path.dirname(out_path))
        stem = os.path.splitext(os.path.basename(out_path))[0]
        image, _, variant = stem.partition("__")
        src = glob.glob(f"inputs/{image}.*")
        if not src:
            print(f"skip {out_path}: no inputs/{image}.*")
            continue
        out = load(out_path)
        inp, kept = match_aspect(load(src[0]), out)
        inp, coverage, zoom = align(inp, out)
        f1, p, r = edge_scores(edges(inp), edges(out))
        light, dlum, diff = light_scores(inp, out)
        h, w = out.shape[:2]
        rows.append(dict(image=image, option=option, variant=variant or "-",
                         out_w=w, out_h=h, zoom=zoom,
                         frame_kept=round(kept / max(zoom, 1.0) ** 2, 3),
                         edge_f1=f1, edge_precision=p, edge_recall=r,
                         light_change=light, mean_lum_delta=dlum))
        print(rows[-1])
        grids.setdefault(image, {"input": inp, "items": []})
        grids[image]["items"].append((f"{option} {variant}".strip(), out, heatmap(diff)))

    if not rows:
        raise SystemExit("No results in outputs/<option>/<image>__<variant>.png")

    df = pd.DataFrame(rows)
    os.makedirs("results/grids", exist_ok=True)
    df.to_csv("results/metrics.csv", index=False)

    for image, g in grids.items():
        cols = [(tile(g["input"], "before"), tile(np.full_like(g["input"], 255), "light change map ->"))]
        cols += [(tile(o, label), tile(hm, "red brighter / blue darker")) for label, o, hm in g["items"]]
        W = sum(t.width for t, _ in cols) + 6 * (len(cols) - 1)
        sheet = Image.new("RGB", (W, 2 * TILE_H + 6), "white")
        x = 0
        for top, bottom in cols:
            sheet.paste(top, (x, 0))
            sheet.paste(bottom, (x, TILE_H + 6))
            x += top.width + 6
        sheet.save(f"results/grids/{image}.jpg", quality=85)

    print("\nMean per option:")
    print(df.groupby("option")[["edge_f1", "edge_precision", "edge_recall",
                                "light_change", "mean_lum_delta", "zoom", "frame_kept"]].mean().round(3))
    runs = "results/runs.csv"
    if os.path.exists(runs):
        manual = pd.read_csv(runs)
        if len(manual):
            print("\nManual scores and cost per option:")
            num = [c for c in ["sec_per_image", "usd_per_image", "geometry", "materials",
                               "light_changed", "shadows_plausible"] if c in manual]
            print(manual.groupby("option")[num].mean().round(2))


if __name__ == "__main__":
    main()
