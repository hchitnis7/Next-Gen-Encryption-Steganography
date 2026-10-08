#!/usr/bin/env python3
"""
Payload sweep -> embedding capacity and payload / imperceptibility trade-off
(Reviewer C, Major Issue 1: embedding capacity utilisation, payload-to-cover ratio,
stego-image size, trade-off between security, imperceptibility and bandwidth).

The payload is increased step by step from a small rate up to the hard capacity of the
embedder. For every rate, on N cover images, PSNR / SSIM / MSE / changed pixels /
chi-square / PNG file size / encoding time are measured. The embedding capacity is then
defined as the largest payload for which imperceptibility is still acceptable:

    capacity(T) = largest embedding rate whose mean PSNR >= T          (rule "mean")
                  [and, stricter, 95 % of the images have PSNR >= T]   (rule "p95")

for T = 50, 45 and 40 dB (40 dB is the usual threshold of perceptual transparency).
It is reported as (1) bits per pixel and (2) cover-to-payload ratio (and the inverse
payload-to-cover ratio used in Table 10 of the paper).

Metric definitions are the same as in alaska2_steganalysis_benchmark_v2_GPU.ipynb
(PSNR, global per-channel SSIM, pairs-of-values chi-square, minimum p over channels).

Usage (from the repository root):
  python xunet_eval/run_capacity_sweep.py --dataset_dir /path/to/alaska2_raw_color_512 --out_dir capacity_results
  python xunet_eval/run_capacity_sweep.py --smoke
"""
import argparse
import csv
import glob
import os
import random
import sys
import time
from pathlib import Path

import cv2
import numpy as np
from scipy.stats import chi2 as chi2_dist

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import IWT_FAST_3CHANNEL as IWT_MOD  # noqa: E402


# ----------------------------------------------------------------------------- metrics
def psnr(cover, stego):
    mse = np.mean((cover.astype(np.float64) - stego.astype(np.float64)) ** 2)
    return float("inf") if mse == 0 else 10 * np.log10(255.0 ** 2 / mse), float(mse)


def ssim_global(cover, stego):
    c1, c2 = (0.01 * 255) ** 2, (0.03 * 255) ** 2
    vals = []
    for ch in range(cover.shape[2]):
        a, b = cover[:, :, ch].astype(np.float64), stego[:, :, ch].astype(np.float64)
        mu_a, mu_b = a.mean(), b.mean()
        sig_a, sig_b = a.std() ** 2, b.std() ** 2
        sig_ab = np.mean((a - mu_a) * (b - mu_b))
        vals.append(((2 * mu_a * mu_b + c1) * (2 * sig_ab + c2)) /
                    ((mu_a ** 2 + mu_b ** 2 + c1) * (sig_a + sig_b + c2)))
    return float(np.mean(vals))


def chi_square_min_p(img):
    """Pairs-of-values chi-square (Westfeld & Pfitzmann), minimum p over B, G, R."""
    ps = []
    for ch in range(3):
        hist = np.bincount(img[:, :, ch].ravel().astype(np.uint8), minlength=256).astype(float)
        obs, exp = [], []
        for k in range(0, 256, 2):
            s = hist[k] + hist[k + 1]
            if s > 0:
                obs.append(hist[k]); exp.append(s / 2.0)
        obs, exp = np.array(obs), np.array(exp)
        stat = float(np.sum((obs - exp) ** 2 / exp))
        dof = len(obs) - 1
        ps.append(float(chi2_dist.sf(stat, dof)) if dof > 0 else 1.0)
    return min(ps)


# ----------------------------------------------------------------------------- data
def load_covers(directory, n, smoke=False):
    if smoke:
        rng = np.random.default_rng(0)
        out = []
        for i in range(n):
            base = cv2.GaussianBlur(rng.integers(0, 255, (512, 512, 3), dtype=np.uint8), (0, 0), 6)
            out.append((f"synthetic_{i:03d}", np.clip(base + rng.normal(0, 3, base.shape), 0, 255).astype(np.uint8)))
        return out
    paths = sorted(glob.glob(os.path.join(directory, "*.tif")) + glob.glob(os.path.join(directory, "*.ppm"))
                   + glob.glob(os.path.join(directory, "*.png")))
    out = []
    for p in paths:
        if len(out) == n:
            break
        img = cv2.imread(p, cv2.IMREAD_COLOR)
        if img is not None and img.shape[:2] == (512, 512):
            out.append((Path(p).stem, img))
    if not out:
        raise FileNotFoundError(f"No usable 512x512 images in {directory}")
    return out


def n_chars(img, bpp, iwt):
    """Message length (characters) for a target rate; 'max' = hard capacity of the embedder."""
    cap = iwt.capacity_bytes(img)
    if bpp == "max":
        return cap
    h, w = img.shape[:2]
    return min(cap, max(1, (int(h * w * float(bpp)) - 64) // 8))


# ----------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset_dir", default="")
    ap.add_argument("--out_dir", default="capacity_results")
    ap.add_argument("--n_images", type=int, default=200)
    ap.add_argument("--rates", nargs="+", default=["0.05", "0.1", "0.2", "0.4", "0.8", "1.2", "1.6", "2.0",
                                                    "2.4", "2.8", "max"])
    ap.add_argument("--thresholds", type=float, nargs="+", default=[50.0, 45.0, 40.0])
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()
    if args.smoke:
        args.n_images, args.rates, args.out_dir = 6, ["0.4", "1.6", "max"], args.out_dir + "_smoke"
    out = Path(args.out_dir); out.mkdir(parents=True, exist_ok=True)
    tmp = out / "tmp"; tmp.mkdir(exist_ok=True)

    covers = load_covers(args.dataset_dir, args.n_images, args.smoke)
    n_img = len(covers)
    h, w = covers[0][1].shape[:2]
    iwt = IWT_MOD.IWT()
    rng = random.Random(args.seed)
    cap_chars = iwt.capacity_bytes(covers[0][1])
    cover_raw_bytes = h * w * 3
    print(f"{n_img} cover images, {w}x{h}; hard capacity {cap_chars:,} characters "
          f"({cap_chars * 8 / (h * w):.4f} bpp); raw cover {cover_raw_bytes:,} B")

    cover_png_kb = {}
    for name, img in covers:                      # cover PNG size, written exactly like the stego PNGs
        p = tmp / f"{name}_cover.png"; cv2.imwrite(str(p), img)
        cover_png_kb[name] = os.path.getsize(p) / 1024.0

    rows = []
    for rate in args.rates:
        t0 = time.time(); verify_fail = 0
        for name, cover in covers:
            n = n_chars(cover, rate, iwt)
            msg = "".join(chr(rng.randint(32, 126)) for _ in range(n))
            p = tmp / f"{name}_{rate}.png"
            t1 = time.perf_counter()
            stego = iwt.encode_image(cover.copy(), msg, output_path=str(p))
            enc_t = time.perf_counter() - t1
            if stego is False or stego is None:
                raise RuntimeError(f"Embedding failed: {name} rate {rate} ({n} chars)")
            if iwt.decode_image(stego) != msg:
                verify_fail += 1
            ps, mse = psnr(cover, stego)
            payload_bits = (len(str(n)) + 1 + n) * 8                  # length prefix + '*' + message
            rows.append({
                "rate": rate, "image": name, "message_chars": n, "payload_bits": payload_bits,
                "bpp": payload_bits / (h * w), "psnr": ps, "mse": mse, "ssim": ssim_global(cover, stego),
                "changed_pixels_pct": 100.0 * float(np.mean(np.any(cover != stego, axis=2))),
                "chi2_p_cover": chi_square_min_p(cover), "chi2_p_stego": chi_square_min_p(stego),
                "cover_png_kb": cover_png_kb[name], "stego_png_kb": os.path.getsize(p) / 1024.0,
                "encode_s": enc_t,
            })
            os.remove(p)
        r = [x for x in rows if x["rate"] == rate]
        print(f"  rate {rate:>5}: {np.mean([x['bpp'] for x in r]):.3f} bpp | PSNR {np.mean([x['psnr'] for x in r]):.2f} dB | "
              f"verified {n_img - verify_fail}/{n_img} | {time.time() - t0:.0f}s")

    with open(out / "sweep_per_image.csv", "w", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=list(rows[0].keys())); wr.writeheader(); wr.writerows(rows)

    # ---------------------------------------------------------------- per-rate summary
    summ = []
    for rate in args.rates:
        r = [x for x in rows if x["rate"] == rate]
        g = lambda k: np.array([x[k] for x in r], dtype=float)
        ps = g("psnr")
        summ.append({"rate": rate, "bpp": g("bpp").mean(), "message_chars": int(g("message_chars").mean()),
                     "payload_bytes": g("payload_bits").mean() / 8, "pct_of_capacity": 100 * g("bpp").mean() / 3.0,
                     "psnr_mean": ps.mean(), "psnr_std": ps.std(ddof=1) if n_img > 1 else 0.0,
                     "psnr_p5": float(np.percentile(ps, 5)), "psnr_min": ps.min(),
                     "ssim_mean": g("ssim").mean(), "mse_mean": g("mse").mean(),
                     "changed_pixels_pct": g("changed_pixels_pct").mean(),
                     "chi2_p_cover": g("chi2_p_cover").mean(), "chi2_p_stego": g("chi2_p_stego").mean(),
                     "cover_png_kb": g("cover_png_kb").mean(), "stego_png_kb": g("stego_png_kb").mean(),
                     "encode_s": g("encode_s").mean()})
    with open(out / "sweep_summary.csv", "w", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=list(summ[0].keys())); wr.writeheader(); wr.writerows(summ)

    # ---------------------------------------------------------------- capacity at each threshold
    order = sorted(summ, key=lambda s: s["bpp"])
    cap_rows = []
    for T in args.thresholds:
        for rule, key in (("mean PSNR >= T", "psnr_mean"), ("95% of images >= T", "psnr_p5")):
            ok = [s for s in order if s[key] >= T]
            if not ok:
                cap_bpp, note = None, f"even the lowest tested rate ({order[0]['bpp']:.3f} bpp) is below {T:g} dB"
            elif ok[-1] is order[-1]:
                cap_bpp, note = order[-1]["bpp"], "not limited by PSNR: the embedder's hard capacity is reached first"
            else:
                lo, hi = ok[-1], order[order.index(ok[-1]) + 1]            # linear interpolation in PSNR
                frac = (lo[key] - T) / (lo[key] - hi[key]) if lo[key] != hi[key] else 0.0
                cap_bpp = lo["bpp"] + frac * (hi["bpp"] - lo["bpp"])
                note = f"between {lo['bpp']:.2f} bpp ({lo[key]:.2f} dB) and {hi['bpp']:.2f} bpp ({hi[key]:.2f} dB)"
            cap_rows.append({"threshold_db": T, "rule": rule, "capacity_bpp": cap_bpp,
                             "payload_bytes_512x512": None if cap_bpp is None else cap_bpp * h * w / 8,
                             "payload_to_cover_pct": None if cap_bpp is None else 100 * cap_bpp / 24.0,
                             "cover_to_payload_ratio": None if cap_bpp is None else 24.0 / cap_bpp,
                             "pct_of_hard_capacity": None if cap_bpp is None else 100 * cap_bpp / 3.0, "note": note})
    with open(out / "capacity_summary.csv", "w", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=list(cap_rows[0].keys())); wr.writeheader(); wr.writerows(cap_rows)

    # ---------------------------------------------------------------- markdown tables
    L = ["| Embedding rate (bpp) | Message (chars) | % of capacity | PSNR (dB) | SSIM | MSE | Changed pixels (%) | "
         "χ² p (cover / stego) | PNG size cover → stego (KB) | Encoding time (s) |", "|" + "---|" * 10]
    for s in summ:
        L.append(f"| {s['bpp']:.3f} | {s['message_chars']:,} | {s['pct_of_capacity']:.1f} | "
                 f"{s['psnr_mean']:.2f} ± {s['psnr_std']:.2f} | {s['ssim_mean']:.4f} | {s['mse_mean']:.3f} | "
                 f"{s['changed_pixels_pct']:.2f} | {s['chi2_p_cover']:.4f} / {s['chi2_p_stego']:.4f} | "
                 f"{s['cover_png_kb']:.0f} → {s['stego_png_kb']:.0f} | {s['encode_s']:.3f} |")
    C = ["| PSNR threshold | Rule | Capacity (bpp) | Payload (bytes, 512×512) | Payload-to-cover (%) | "
         "Cover-to-payload | % of hard capacity |", "|" + "---|" * 7]
    for c in cap_rows:
        if c["capacity_bpp"] is None:
            C.append(f"| {c['threshold_db']:g} dB | {c['rule']} | < {order[0]['bpp']:.2f} | – | – | – | – |")
        else:
            C.append(f"| {c['threshold_db']:g} dB | {c['rule']} | {c['capacity_bpp']:.2f} | {c['payload_bytes_512x512']:,.0f} | "
                     f"{c['payload_to_cover_pct']:.2f} | {c['cover_to_payload_ratio']:.1f} : 1 | {c['pct_of_hard_capacity']:.1f} |")
    notes = [f"\nHard capacity of the embedder: {cap_chars:,} characters = {cap_chars * 8 / (h * w):.3f} bpp "
             f"(3 channels × 1 bit per pixel); raw cover {cover_raw_bytes:,} B; payload-to-cover = payload bytes ÷ raw cover bytes = bpp ÷ 24.",
             f"{n_img} covers; PSNR is mean ± sample std over images. 'Capacity' rows: {'; '.join(f"{c['threshold_db']:g} dB: {c['note']}" for c in cap_rows[::2])}."]
    (out / "sweep_table.md").write_text("\n".join(L) + "\n\n" + "\n".join(C) + "\n" + "\n".join(notes) + "\n")

    # ---------------------------------------------------------------- figure
    try:
        import matplotlib; matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(6.2, 3.8))
        x = [s["bpp"] for s in order]
        ax.plot(x, [s["psnr_mean"] for s in order], "o-", label="mean PSNR")
        ax.fill_between(x, [s["psnr_p5"] for s in order], [s["psnr_mean"] for s in order], alpha=0.2, label="5th percentile to mean")
        for T in args.thresholds:
            ax.axhline(T, ls="--", lw=0.8, color="gray"); ax.text(x[-1], T + 0.3, f"{T:g} dB", ha="right", fontsize=8)
        ax.set_xlabel("Embedding rate (bits per pixel)"); ax.set_ylabel("PSNR (dB)"); ax.grid(alpha=0.3); ax.legend(fontsize=8)
        fig.tight_layout(); fig.savefig(out / "psnr_vs_payload.png", dpi=200); fig.savefig(out / "psnr_vs_payload.pdf")
    except Exception as e:                                                    # plotting is optional
        print("figure skipped:", e)

    print("\n" + "\n".join(L) + "\n\n" + "\n".join(C) + "\n" + "\n".join(notes))
    print(f"\nSaved in {out}/ -> send me sweep_table.md, sweep_summary.csv, capacity_summary.csv and the console output")


if __name__ == "__main__":
    main()
