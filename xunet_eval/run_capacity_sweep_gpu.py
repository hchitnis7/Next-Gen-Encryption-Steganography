#!/usr/bin/env python3
"""
GPU / multi-core version of run_capacity_sweep.py (same outputs, same definitions).

Payload sweep -> embedding capacity and payload / imperceptibility trade-off
(Reviewer C, Major Issue 1).

What runs where
  * IWT embedding + decoding check : CPU (numba), spread over --workers processes. This is the slow part,
    so it is the part that is parallelised. A GPU cannot speed it up without rewriting the embedder.
  * PSNR / MSE / SSIM / chi-square histograms / changed pixels : GPU (torch, float64) if CUDA is available,
    otherwise CPU. Results are computed while the workers keep embedding.
  * Encoding time : measured in a separate, single-process pass (--time_images images per rate) so that it is
    not distorted by the parallel workers.

Capacity(T) = largest embedding rate whose mean PSNR >= T (and, stricter, 95 % of the images >= T),
for T = 50, 45, 40 dB, reported in bpp and as cover-to-payload ratio.

Usage (from the repository root)
  python xunet_eval/run_capacity_sweep_gpu.py --dataset_dir /path/to/alaska2_raw_color_512 --out_dir capacity_results
  python xunet_eval/run_capacity_sweep_gpu.py --smoke
"""
import argparse
import csv
import glob
import multiprocessing as mp
import os
import random
import sys
import time
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

_IWT = None


# ----------------------------------------------------------------------------- worker (CPU)
def _init_worker():
    """Import the embedder once per worker and warm up numba so compile time is not counted."""
    global _IWT
    import IWT_FAST_3CHANNEL as mod
    _IWT = mod.IWT()
    rng = np.random.default_rng(1)
    dummy = rng.integers(0, 255, (64, 64, 3), dtype=np.uint8)
    tmp = os.path.join(os.environ.get("TMPDIR", "/tmp"), f"warm_{os.getpid()}.png")
    _IWT.encode_image(dummy, "warm-up", output_path=tmp)
    if os.path.exists(tmp):
        os.remove(tmp)


def _embed_task(task):
    cover_path, name, rate, n_chars, seed, out_png = task
    cover = cv2.imread(cover_path, cv2.IMREAD_COLOR)
    rng = random.Random(seed)
    msg = "".join(chr(rng.randint(32, 126)) for _ in range(n_chars))
    t0 = time.perf_counter()
    stego = _IWT.encode_image(cover.copy(), msg, output_path=out_png)
    dt = time.perf_counter() - t0
    if stego is False or stego is None:
        return {"error": f"embedding failed: {name} rate {rate} ({n_chars} chars)"}
    verified = (_IWT.decode_image(stego) == msg)
    payload_bits = (len(str(n_chars)) + 1 + n_chars) * 8      # length prefix + '*' + message
    return {"rate": rate, "name": name, "cover_path": cover_path, "message_chars": n_chars,
            "payload_bits": payload_bits, "verified": verified, "encode_s_parallel": dt, "stego_png": out_png}


# ----------------------------------------------------------------------------- metrics (GPU or CPU, torch)
def make_metrics(device):
    import torch
    from scipy.stats import chi2 as chi2_dist

    def to_t(img):
        return torch.from_numpy(img).to(device)

    def psnr_mse(c, s):
        mse = (c.double() - s.double()).pow(2).mean()
        m = float(mse.item())
        return (float("inf") if m == 0 else 10.0 * float(np.log10(255.0 ** 2 / m))), m

    def ssim_global(c, s):
        c1, c2 = (0.01 * 255) ** 2, (0.03 * 255) ** 2
        vals = []
        for ch in range(3):
            a, b = c[:, :, ch].double(), s[:, :, ch].double()
            mu_a, mu_b = a.mean(), b.mean()
            sig_a, sig_b = a.var(unbiased=False), b.var(unbiased=False)
            sig_ab = ((a - mu_a) * (b - mu_b)).mean()
            vals.append(((2 * mu_a * mu_b + c1) * (2 * sig_ab + c2)) /
                        ((mu_a ** 2 + mu_b ** 2 + c1) * (sig_a + sig_b + c2)))
        return float(torch.stack(vals).mean().item())

    def chi_min_p(img):
        ps = []
        for ch in range(3):
            hist = torch.bincount(img[:, :, ch].reshape(-1).long(), minlength=256).double()
            even, odd = hist[0::2], hist[1::2]
            tot = even + odd
            mask = tot > 0
            obs, exp = even[mask], tot[mask] / 2.0
            stat = float(((obs - exp) ** 2 / exp).sum().item())
            dof = int(mask.sum().item()) - 1
            ps.append(float(chi2_dist.sf(stat, dof)) if dof > 0 else 1.0)
        return min(ps)

    def changed_pct(c, s):
        return 100.0 * float((c != s).any(dim=2).double().mean().item())

    return to_t, psnr_mse, ssim_global, chi_min_p, changed_pct


# ----------------------------------------------------------------------------- data
def load_cover_files(directory, n, smoke, tmp):
    """Returns [(name, path)] of 512x512 covers. Smoke mode writes synthetic covers to PNG files."""
    out = []
    if smoke:
        rng = np.random.default_rng(0)
        for i in range(n):
            base = cv2.GaussianBlur(rng.integers(0, 255, (512, 512, 3), dtype=np.uint8), (0, 0), 6)
            img = np.clip(base + rng.normal(0, 3, base.shape), 0, 255).astype(np.uint8)
            p = tmp / f"synthetic_{i:03d}.png"
            cv2.imwrite(str(p), img)
            out.append((p.stem, str(p)))
        return out
    paths = sorted(glob.glob(os.path.join(directory, "*.tif")) + glob.glob(os.path.join(directory, "*.ppm"))
                   + glob.glob(os.path.join(directory, "*.png")))
    for p in paths:
        if len(out) == n:
            break
        img = cv2.imread(p, cv2.IMREAD_COLOR)
        if img is not None and img.shape[:2] == (512, 512):
            out.append((Path(p).stem, p))
    if not out:
        raise FileNotFoundError("No usable 512x512 images in " + str(directory))
    return out


# ----------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset_dir", default="")
    ap.add_argument("--out_dir", default="capacity_results")
    ap.add_argument("--n_images", type=int, default=200)
    ap.add_argument("--rates", nargs="+", default=["0.05", "0.1", "0.2", "0.4", "0.8", "1.2", "1.6", "2.0",
                                                    "2.4", "2.8", "max"])
    ap.add_argument("--thresholds", type=float, nargs="+", default=[50.0, 45.0, 40.0])
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 1))
    ap.add_argument("--time_images", type=int, default=10, help="images per rate for the single-process timing pass")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--cpu", action="store_true", help="force the metrics onto the CPU")
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()
    if args.smoke:
        args.n_images, args.rates, args.workers, args.time_images = 6, ["0.4", "1.6", "max"], min(args.workers, 2), 2
        args.out_dir = args.out_dir + "_smoke"
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    tmp = out / "tmp"
    tmp.mkdir(exist_ok=True)

    covers = load_cover_files(args.dataset_dir, args.n_images, args.smoke, tmp)
    n_img = len(covers)
    h = w = 512
    cap_chars = (h // 8) * (w // 8) * 3 * 64 // 8 - 16          # same as IWT.capacity_bytes for a 512x512 image

    def chars_for(rate):
        if rate == "max":
            return cap_chars
        return min(cap_chars, max(1, (int(h * w * float(rate)) - 64) // 8))

    tasks = []
    for ri, rate in enumerate(args.rates):
        for ii, (name, path) in enumerate(covers):
            seed = args.seed * 1_000_003 + ri * 10_007 + ii
            tasks.append((path, name, rate, chars_for(rate), seed, str(tmp / f"{name}_{rate}.png")))

    # the pool is created BEFORE CUDA is touched in this process
    ctx = mp.get_context("spawn")
    pool = ctx.Pool(processes=args.workers, initializer=_init_worker)

    import torch
    device = torch.device("cpu" if (args.cpu or not torch.cuda.is_available()) else "cuda")
    print(f"{n_img} covers, hard capacity {cap_chars:,} characters ({cap_chars * 8 / (h * w):.4f} bpp) | "
          f"{len(tasks)} embeddings on {args.workers} worker(s) | metrics on {device}")
    to_t, psnr_mse, ssim_global, chi_min_p, changed_pct = make_metrics(device)

    cover_png_kb, cover_chi, cover_t = {}, {}, {}
    for name, path in covers:
        img = cv2.imread(path, cv2.IMREAD_COLOR)
        p = tmp / f"{name}_cover.png"
        cv2.imwrite(str(p), img)
        cover_png_kb[name] = os.path.getsize(p) / 1024.0
        cover_t[name] = to_t(img)
        cover_chi[name] = chi_min_p(cover_t[name])

    rows, t_start, done = [], time.time(), 0
    for res in pool.imap_unordered(_embed_task, tasks, chunksize=1):
        if "error" in res:
            pool.terminate()
            raise RuntimeError(res["error"])
        stego_img = cv2.imread(res["stego_png"], cv2.IMREAD_COLOR)
        st = to_t(stego_img)
        c = cover_t[res["name"]]
        ps, mse = psnr_mse(c, st)
        rows.append({
            "rate": res["rate"], "image": res["name"], "message_chars": res["message_chars"],
            "payload_bits": res["payload_bits"], "bpp": res["payload_bits"] / (h * w), "psnr": ps, "mse": mse,
            "ssim": ssim_global(c, st), "changed_pixels_pct": changed_pct(c, st),
            "chi2_p_cover": cover_chi[res["name"]], "chi2_p_stego": chi_min_p(st),
            "cover_png_kb": cover_png_kb[res["name"]], "stego_png_kb": os.path.getsize(res["stego_png"]) / 1024.0,
            "encode_s_parallel": res["encode_s_parallel"], "verified": int(res["verified"])})
        os.remove(res["stego_png"])
        done += 1
        if done % 25 == 0 or done == len(tasks):
            el = time.time() - t_start
            print(f"  {done}/{len(tasks)} done | {el / 60:.1f} min elapsed | ETA {el / done * (len(tasks) - done) / 60:.1f} min")
    pool.close()
    pool.join()
    bad = sum(1 for r in rows if not r["verified"])
    print(f"round-trip decoding check: {len(rows) - bad}/{len(rows)} verified")
    rows.sort(key=lambda r: (args.rates.index(r["rate"]), r["image"]))

    # ---------------------------------------------------------------- uncontended encoding time
    import IWT_FAST_3CHANNEL as mod
    iwt = mod.IWT()
    enc_time = {}
    for ri, rate in enumerate(args.rates):
        ts = []
        for ii, (name, path) in enumerate(covers[:max(1, args.time_images)]):
            img = cv2.imread(path, cv2.IMREAD_COLOR)
            rng = random.Random(7 + ii)
            msg = "".join(chr(rng.randint(32, 126)) for _ in range(chars_for(rate)))
            t0 = time.perf_counter()
            iwt.encode_image(img, msg, output_path=str(tmp / "timing.png"))
            ts.append(time.perf_counter() - t0)
        enc_time[rate] = float(np.mean(ts[1:] if len(ts) > 1 else ts))      # first call includes numba compile
    for r in rows:
        r["encode_s"] = enc_time[r["rate"]]

    with open(out / "sweep_per_image.csv", "w", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        wr.writeheader()
        wr.writerows(rows)

    # ---------------------------------------------------------------- per-rate summary
    summ = []
    for rate in args.rates:
        r = [x for x in rows if x["rate"] == rate]

        def g(k):
            return np.array([x[k] for x in r], dtype=float)

        ps = g("psnr")
        summ.append({"rate": rate, "bpp": g("bpp").mean(), "message_chars": int(g("message_chars").mean()),
                     "payload_bytes": g("payload_bits").mean() / 8, "pct_of_capacity": 100 * g("bpp").mean() / 3.0,
                     "psnr_mean": ps.mean(), "psnr_std": ps.std(ddof=1) if n_img > 1 else 0.0,
                     "psnr_p5": float(np.percentile(ps, 5)), "psnr_min": ps.min(),
                     "ssim_mean": g("ssim").mean(), "mse_mean": g("mse").mean(),
                     "changed_pixels_pct": g("changed_pixels_pct").mean(),
                     "chi2_p_cover": g("chi2_p_cover").mean(), "chi2_p_stego": g("chi2_p_stego").mean(),
                     "cover_png_kb": g("cover_png_kb").mean(), "stego_png_kb": g("stego_png_kb").mean(),
                     "encode_s": enc_time[rate]})
    with open(out / "sweep_summary.csv", "w", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=list(summ[0].keys()))
        wr.writeheader()
        wr.writerows(summ)

    # ---------------------------------------------------------------- capacity at each threshold
    order = sorted(summ, key=lambda s: s["bpp"])
    cap_rows = []
    for T in args.thresholds:
        for rule, key in (("mean PSNR >= T", "psnr_mean"), ("95% of images >= T", "psnr_p5")):
            ok = [s for s in order if s[key] >= T]
            if not ok:
                cap_bpp, note = None, "even the lowest tested rate (%.3f bpp) is below %g dB" % (order[0]["bpp"], T)
            elif ok[-1] is order[-1]:
                cap_bpp, note = order[-1]["bpp"], "not limited by PSNR: the embedder's hard capacity is reached first"
            else:
                lo, hi = ok[-1], order[order.index(ok[-1]) + 1]
                frac = (lo[key] - T) / (lo[key] - hi[key]) if lo[key] != hi[key] else 0.0
                cap_bpp = lo["bpp"] + frac * (hi["bpp"] - lo["bpp"])
                note = "between %.2f bpp (%.2f dB) and %.2f bpp (%.2f dB)" % (lo["bpp"], lo[key], hi["bpp"], hi[key])
            cap_rows.append({"threshold_db": T, "rule": rule, "capacity_bpp": cap_bpp,
                             "payload_bytes_512x512": None if cap_bpp is None else cap_bpp * h * w / 8,
                             "payload_to_cover_pct": None if cap_bpp is None else 100 * cap_bpp / 24.0,
                             "cover_to_payload_ratio": None if cap_bpp is None else 24.0 / cap_bpp,
                             "pct_of_hard_capacity": None if cap_bpp is None else 100 * cap_bpp / 3.0, "note": note})
    with open(out / "capacity_summary.csv", "w", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=list(cap_rows[0].keys()))
        wr.writeheader()
        wr.writerows(cap_rows)

    # ---------------------------------------------------------------- markdown tables
    L = ["| Embedding rate (bpp) | Message (chars) | % of capacity | PSNR (dB) | SSIM | MSE | Changed pixels (%) | "
         "chi2 p (cover / stego) | PNG size cover -> stego (KB) | Encoding time (s) |", "|" + "---|" * 10]
    for s in summ:
        L.append("| %.3f | %s | %.1f | %.2f +/- %.2f | %.4f | %.3f | %.2f | %.4f / %.4f | %.0f -> %.0f | %.3f |" % (
            s["bpp"], format(s["message_chars"], ","), s["pct_of_capacity"], s["psnr_mean"], s["psnr_std"],
            s["ssim_mean"], s["mse_mean"], s["changed_pixels_pct"], s["chi2_p_cover"], s["chi2_p_stego"],
            s["cover_png_kb"], s["stego_png_kb"], s["encode_s"]))
    C = ["| PSNR threshold | Rule | Capacity (bpp) | Payload (bytes, 512x512) | Payload-to-cover (%) | "
         "Cover-to-payload | % of hard capacity |", "|" + "---|" * 7]
    for c in cap_rows:
        if c["capacity_bpp"] is None:
            C.append("| %g dB | %s | < %.2f | - | - | - | - |" % (c["threshold_db"], c["rule"], order[0]["bpp"]))
        else:
            C.append("| %g dB | %s | %.2f | %s | %.2f | %.1f : 1 | %.1f |" % (
                c["threshold_db"], c["rule"], c["capacity_bpp"], format(round(c["payload_bytes_512x512"]), ","),
                c["payload_to_cover_pct"], c["cover_to_payload_ratio"], c["pct_of_hard_capacity"]))
    cap_note = "; ".join("%g dB: %s" % (c["threshold_db"], c["note"]) for c in cap_rows[::2])
    notes = ["",
             "Hard capacity of the embedder: %s characters = %.3f bpp (3 channels x 1 bit per pixel); raw cover %s B; "
             "payload-to-cover = payload bytes / raw cover bytes = bpp / 24." % (
                 format(cap_chars, ","), cap_chars * 8 / (h * w), format(h * w * 3, ",")),
             "%d covers; PSNR is mean +/- sample std over images; encoding time is a single-process measurement on %d image(s) per rate. "
             "Capacity: %s." % (n_img, min(args.time_images, n_img), cap_note)]
    (out / "sweep_table.md").write_text("\n".join(L) + "\n\n" + "\n".join(C) + "\n" + "\n".join(notes) + "\n")

    # ---------------------------------------------------------------- figure
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(6.2, 3.8))
        x = [s["bpp"] for s in order]
        ax.plot(x, [s["psnr_mean"] for s in order], "o-", label="mean PSNR")
        ax.fill_between(x, [s["psnr_p5"] for s in order], [s["psnr_mean"] for s in order], alpha=0.2,
                        label="5th percentile to mean")
        for T in args.thresholds:
            ax.axhline(T, ls="--", lw=0.8, color="gray")
            ax.text(x[-1], T + 0.3, "%g dB" % T, ha="right", fontsize=8)
        ax.set_xlabel("Embedding rate (bits per pixel)")
        ax.set_ylabel("PSNR (dB)")
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8)
        fig.tight_layout()
        fig.savefig(out / "psnr_vs_payload.png", dpi=200)
        fig.savefig(out / "psnr_vs_payload.pdf")
    except Exception as e:
        print("figure skipped:", e)

    print("\n" + "\n".join(L) + "\n\n" + "\n".join(C) + "\n" + "\n".join(notes))
    print("\nSaved in %s/ -> send me sweep_table.md, sweep_summary.csv, capacity_summary.csv and this console output" % out)


if __name__ == "__main__":
    main()
