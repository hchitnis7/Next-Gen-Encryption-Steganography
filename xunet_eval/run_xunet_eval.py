#!/usr/bin/env python3
"""
Xu-Net steganalysis evaluation with a held-out test set (Reviewer A, comment 6).

What this fixes compared with the earlier notebook evaluation
-------------------------------------------------------------
* Train / validation / TEST are three disjoint sets of IMAGE PAIRS. A cover and its
  stego twin always sit in the same set, and this is asserted in code (no leakage).
* The validation set is used only for early stopping / checkpoint selection.
  The test set is evaluated exactly once per trial, on the selected checkpoint.
* Each batch contains the cover AND stego version of the same images
  (pair-constrained batches), the usual practice for Xu-Net-style training; this
  avoids the collapse to AUC = 0.5 seen at low payload with random batches.
* Every setting is written to config.json, every split to splits_*.json, and every
  trial to trials.csv, so the numbers in the paper can be traced to files.

Settings are kept identical to what the paper already states wherever possible:
ALASKA#2 512x512 colour, 1,000 pairs per rate, rates 0.1/0.2/0.4 bpp,
three trials with seeds 42/1042/2042, Adamax lr=1e-3, batch size 32 images,
at most 30 epochs, 80% / 20% pair split (the 20% is now the TEST set; 10% of the
80% is carved out as validation).

Usage
-----
  python xunet_eval/run_xunet_eval.py --dataset_dir /path/to/alaska2_raw_color_512 \
         --out_dir xunet_results

  python xunet_eval/run_xunet_eval.py --smoke      # 2-minute CPU sanity check on synthetic images

Run from the repository root (it imports IWT_FAST_3CHANNEL.py from there).
"""
import argparse
import csv
import glob
import json
import os
import random
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score, roc_curve

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import IWT_FAST_3CHANNEL as IWT_MOD  # noqa: E402


# ----------------------------------------------------------------------------- model
class AbsActivation(nn.Module):
    def forward(self, x):
        return torch.abs(x)


class XuNet(nn.Module):
    """Xu-Net (Xu et al., 2016) with a fixed KV high-pass filter per input channel."""

    def __init__(self, in_channels: int = 3):
        super().__init__()
        kv = torch.tensor([[-1, 2, -2, 2, -1],
                           [2, -6, 8, -6, 2],
                           [-2, 8, -12, 8, -2],
                           [2, -6, 8, -6, 2],
                           [-1, 2, -2, 2, -1]], dtype=torch.float32) / 12.0
        self.register_buffer("kv", kv.view(1, 1, 5, 5).repeat(in_channels, 1, 1, 1))
        self.groups = in_channels
        self.g1 = nn.Sequential(nn.Conv2d(in_channels, 8, 5, padding=2, bias=False),
                                nn.BatchNorm2d(8), AbsActivation(), nn.Tanh(),
                                nn.AvgPool2d(5, stride=2, padding=2))
        self.g2 = nn.Sequential(nn.Conv2d(8, 16, 5, padding=2, bias=False),
                                nn.BatchNorm2d(16), nn.Tanh(),
                                nn.AvgPool2d(5, stride=2, padding=2))
        self.g3 = nn.Sequential(nn.Conv2d(16, 32, 1, bias=False), nn.BatchNorm2d(32),
                                nn.ReLU(True), nn.AvgPool2d(5, stride=2, padding=2))
        self.g4 = nn.Sequential(nn.Conv2d(32, 64, 1, bias=False), nn.BatchNorm2d(64),
                                nn.ReLU(True), nn.AvgPool2d(5, stride=2, padding=2))
        self.g5 = nn.Sequential(nn.Conv2d(64, 128, 1, bias=False), nn.BatchNorm2d(128),
                                nn.ReLU(True))
        self.fc = nn.Linear(128, 2)

    def forward(self, x):
        x = F.conv2d(x, self.kv, padding=2, groups=self.groups)
        x = self.g5(self.g4(self.g3(self.g2(self.g1(x)))))
        return self.fc(F.adaptive_avg_pool2d(x, 1).flatten(1))


# ----------------------------------------------------------------------------- data
def load_covers(directory, n, smoke=False):
    if smoke:
        rng = np.random.default_rng(0)
        out = []
        for i in range(n):
            base = cv2.GaussianBlur(rng.integers(0, 255, (512, 512, 3), dtype=np.uint8), (0, 0), 6)
            noise = rng.normal(0, 3, base.shape)
            out.append((f"synthetic_{i:05d}", np.clip(base + noise, 0, 255).astype(np.uint8)))
        return out
    paths = sorted(glob.glob(os.path.join(directory, "*.tif")) +
                   glob.glob(os.path.join(directory, "*.ppm")) +
                   glob.glob(os.path.join(directory, "*.png")))
    if not paths:
        raise FileNotFoundError(f"No .tif/.ppm/.png images in {directory}")
    out = []
    for p in paths:
        if len(out) == n:
            break
        img = cv2.imread(p, cv2.IMREAD_COLOR)
        if img is not None and img.shape[:2] == (512, 512):
            out.append((Path(p).stem, img))
    if len(out) < n:
        print(f"WARNING: only {len(out)} usable 512x512 images found (wanted {n})")
    return out


def n_chars_for_bpp(img, bpp):
    h, w = img.shape[:2]
    return max(0, int(h * w * bpp) - 64) // 8       # same rule as the benchmark notebooks


def build_stego(covers, bpp, cache_dir, seed):
    """Embed (or reuse cached PNGs). Returns list of stego arrays aligned with `covers`."""
    d = Path(cache_dir) / f"bpp_{bpp}"
    d.mkdir(parents=True, exist_ok=True)
    rng = random.Random(seed + int(bpp * 1000))
    iwt = IWT_MOD.IWT()
    stegos, verified = [], 0
    for name, cover in covers:
        p = d / f"{name}.png"
        msg = "".join(chr(rng.randint(32, 126)) for _ in range(n_chars_for_bpp(cover, bpp)))
        if p.exists():
            s = cv2.imread(str(p), cv2.IMREAD_COLOR)
        else:
            s = iwt.encode_image(cover.copy(), msg, output_path=str(p))
            if s is False or s is None:
                raise RuntimeError(f"Embedding failed for {name} at {bpp} bpp")
            s = cv2.imread(str(p), cv2.IMREAD_COLOR)          # exactly what is on disk
        stegos.append(s)
    # round-trip check on a few images (cheap)
    for (name, cover), s in list(zip(covers, stegos))[:5]:
        if iwt.decode_image(s) != "":
            verified += 1
    return stegos, verified


def split_pairs(n, seed, test_frac, val_frac):
    rng = np.random.default_rng(seed)
    idx = rng.permutation(n)
    n_test = int(round(n * test_frac))
    test, pool = idx[:n_test], idx[n_test:]
    n_val = int(round(len(pool) * val_frac))
    val, train = pool[:n_val], pool[n_val:]
    # leakage guard: pair indices are disjoint by construction, assert anyway
    assert not (set(train) & set(val)) and not (set(train) & set(test)) and not (set(val) & set(test))
    assert len(train) + len(val) + len(test) == n
    return np.sort(train), np.sort(val), np.sort(test)


def make_batch(covers_u8, stegos_u8, pair_idx, device):
    """Pair-constrained batch: cover and stego of each selected image."""
    c = covers_u8[pair_idx]
    s = stegos_u8[pair_idx]
    x = np.concatenate([c, s], axis=0).astype(np.float32) / 255.0
    y = np.concatenate([np.zeros(len(pair_idx)), np.ones(len(pair_idx))]).astype(np.int64)
    x = torch.from_numpy(x).permute(0, 3, 1, 2).contiguous().to(device)
    return x, torch.from_numpy(y).to(device)


@torch.no_grad()
def predict(model, covers_u8, stegos_u8, pair_idx, device, chunk=16):
    model.eval()
    scores, labels = [], []
    for i in range(0, len(pair_idx), chunk):
        x, y = make_batch(covers_u8, stegos_u8, pair_idx[i:i + chunk], device)
        scores.append(F.softmax(model(x), dim=1)[:, 1].cpu().numpy())
        labels.append(y.cpu().numpy())
    return np.concatenate(labels), np.concatenate(scores)


def pick_threshold(y, s):
    """Decision threshold chosen on the VALIDATION set only (maximises validation accuracy)."""
    cand = np.unique(np.concatenate([s, [0.5]]))
    accs = [np.mean((s >= t).astype(int) == y) for t in cand]
    return float(cand[int(np.argmax(accs))])


def metrics(y, s, thr):
    auc = float(roc_auc_score(y, s))
    acc = float(np.mean((s >= thr).astype(int) == y))
    fpr, tpr, _ = roc_curve(y, s)
    pe_min = float(np.min(0.5 * (fpr + (1 - tpr))))
    return {"auc": auc, "acc": acc, "pe": 1.0 - acc, "pe_min": pe_min}


# ----------------------------------------------------------------------------- trial
def run_trial(covers_u8, stegos_u8, seed, args, device):
    n = covers_u8.shape[0]
    train, val, test = split_pairs(n, seed, args.test_frac, args.val_frac)
    torch.manual_seed(seed); np.random.seed(seed); random.seed(seed)

    model = XuNet().to(device)
    opt = torch.optim.Adamax(model.parameters(), lr=args.lr)
    loss_fn = nn.CrossEntropyLoss()
    pairs_per_batch = args.batch_size // 2          # batch_size counts images (cover + stego)
    rng = np.random.default_rng(seed)

    best_auc, best_state, best_epoch, bad, best_thr = -1.0, None, -1, 0, 0.5
    for epoch in range(args.epochs):
        model.train()
        order = rng.permutation(train)
        for i in range(0, len(order) - pairs_per_batch + 1, pairs_per_batch):
            x, y = make_batch(covers_u8, stegos_u8, order[i:i + pairs_per_batch], device)
            opt.zero_grad()
            loss_fn(model(x), y).backward()
            opt.step()
        yv, sv = predict(model, covers_u8, stegos_u8, val, device)
        v_auc = roc_auc_score(yv, sv)
        if v_auc > best_auc + 1e-4:
            best_auc, best_epoch, bad = v_auc, epoch + 1, 0
            best_thr = pick_threshold(yv, sv)
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        else:
            bad += 1
        if args.verbose:
            print(f"      epoch {epoch + 1:02d}  val AUC {v_auc:.4f}  (best {best_auc:.4f} @ {best_epoch})")
        if bad >= args.patience:
            break

    model.load_state_dict(best_state)
    yt, st = predict(model, covers_u8, stegos_u8, test, device)      # single look at the test set
    m = metrics(yt, st, best_thr)
    m.update({"seed": seed, "best_epoch": best_epoch, "threshold": best_thr, "epochs_run": epoch + 1, "val_auc": float(best_auc),
              "n_train_pairs": int(len(train)), "n_val_pairs": int(len(val)), "n_test_pairs": int(len(test)),
              "collapsed": bool(best_auc < 0.55)})
    splits = {"train": train.tolist(), "val": val.tolist(), "test": test.tolist()}
    return m, splits


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset_dir", default="")
    ap.add_argument("--out_dir", default="xunet_results")
    ap.add_argument("--cache_dir", default="", help="where stego PNGs are cached (default: <out_dir>/stego_cache)")
    ap.add_argument("--n_pairs", type=int, default=1000)
    ap.add_argument("--rates", type=float, nargs="+", default=[0.1, 0.2, 0.4])
    ap.add_argument("--trials", type=int, default=3)
    ap.add_argument("--base_seed", type=int, default=42)
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--patience", type=int, default=8)
    ap.add_argument("--batch_size", type=int, default=32, help="images per batch (16 pairs)")
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--test_frac", type=float, default=0.2)
    ap.add_argument("--val_frac", type=float, default=0.1, help="fraction of the non-test pairs used for validation")
    ap.add_argument("--smoke", action="store_true", help="tiny synthetic run to check the pipeline")
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args()

    if args.smoke:
        args.n_pairs, args.rates, args.trials, args.epochs, args.patience = 40, [0.4], 2, 3, 3
        args.out_dir = args.out_dir + "_smoke"
    out = Path(args.out_dir); out.mkdir(parents=True, exist_ok=True)
    cache = args.cache_dir or str(out / "stego_cache")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    covers = load_covers(args.dataset_dir, args.n_pairs, smoke=args.smoke)
    n = len(covers)
    covers_u8 = np.stack([c for _, c in covers])
    (out / "config.json").write_text(json.dumps({**vars(args), "n_pairs_loaded": n, "device": str(device),
                                                  "torch": torch.__version__,
                                                  "image_ids": [nm for nm, _ in covers]}, indent=1))

    rows, summary = [], []
    for bpp in args.rates:
        t0 = time.time()
        stegos, _ = build_stego(covers, bpp, cache, args.base_seed)
        stegos_u8 = np.stack(stegos)
        diff = np.abs(covers_u8.astype(np.int16) - stegos_u8.astype(np.int16))
        print(f"\n=== {bpp} bpp | {n} pairs | changed pixels {np.mean(diff.max(axis=3) > 0) * 100:.2f}% "
              f"| embed/load {time.time() - t0:.0f}s ===")
        res = []
        for t in range(args.trials):
            seed = args.base_seed + 1000 * t           # 42, 1042, 2042 as stated in the paper
            m, sp = run_trial(covers_u8, stegos_u8, seed, args, device)
            (out / f"splits_{bpp}_seed{seed}.json").write_text(json.dumps(sp))
            m.update({"bpp": bpp, "trial": t + 1})
            res.append(m); rows.append(m)
            print(f"  trial {t + 1}/{args.trials} seed {seed}: TEST AUC {m['auc']:.4f}  PE {m['pe']:.4f}  "
                  f"Acc {m['acc']:.4f}  (best epoch {m['best_epoch']}, val AUC {m['val_auc']:.4f}"
                  f"{', COLLAPSED' if m['collapsed'] else ''})")
        a = np.array([r["auc"] for r in res]); p = np.array([r["pe"] for r in res])
        ac = np.array([r["acc"] for r in res]); pm = np.array([r["pe_min"] for r in res])
        summary.append({"bpp": bpp, "auc_mean": a.mean(), "auc_std": a.std(ddof=1) if len(a) > 1 else 0.0,
                        "pe_mean": p.mean(), "pe_std": p.std(ddof=1) if len(p) > 1 else 0.0,
                        "acc_mean": ac.mean(), "acc_std": ac.std(ddof=1) if len(ac) > 1 else 0.0,
                        "pe_min_mean": pm.mean(), "collapsed_trials": int(sum(r["collapsed"] for r in res)),
                        "n_train": res[0]["n_train_pairs"], "n_val": res[0]["n_val_pairs"],
                        "n_test": res[0]["n_test_pairs"]})
        del stegos, stegos_u8

    with open(out / "trials.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
    with open(out / "summary.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(summary[0].keys())); w.writeheader(); w.writerows(summary)

    lines = ["| Embedding Rate (bpp) | AUC (mean ± std) | PE (mean ± std) | Accuracy (mean ± std) |",
             "|---|---|---|---|"]
    for s in summary:
        lines.append(f"| {s['bpp']} | {s['auc_mean']:.4f} ± {s['auc_std']:.4f} | "
                     f"{s['pe_mean']:.4f} ± {s['pe_std']:.4f} | {s['acc_mean']:.4f} ± {s['acc_std']:.4f} |")
    s0 = summary[0]
    note = (f"\nSplit (pairs): train {s0['n_train']} / validation {s0['n_val']} / test {s0['n_test']}; "
            f"std is the sample std (ddof=1) over {args.trials} trials; accuracy uses a threshold fixed on the validation set; PE = 1 - accuracy; "
            f"collapsed trials (val AUC < 0.55): {sum(s['collapsed_trials'] for s in summary)}.\n")
    (out / "paper_table.md").write_text("\n".join(lines) + "\n" + note)
    print("\n" + "\n".join(lines) + note)
    print(f"Everything saved in {out}/  -> send me paper_table.md, summary.csv, trials.csv, config.json")


if __name__ == "__main__":
    main()
