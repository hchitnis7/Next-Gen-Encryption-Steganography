# Xu-Net held-out evaluation (response to Reviewer A, comment 6)

Run from the **repository root** (needs torch, opencv-python, numba, scikit-learn; a GPU is strongly recommended).

```bash
# 1. sanity check (synthetic images, ~1 min on CPU)
python xunet_eval/run_xunet_eval.py --smoke

# 2. real run: same dataset, rates, pairs and seeds as the paper
python xunet_eval/run_xunet_eval.py --dataset_dir /path/to/alaska2_raw_color_512 --out_dir xunet_results
```

Defaults match the paper: ALASKA#2 512x512 colour, 1,000 pairs per rate, 0.1/0.2/0.4 bpp, 3 trials (seeds 42/1042/2042),
Adamax lr 1e-3, 32 images (16 cover/stego pairs) per batch, at most 30 epochs.

Split per trial (pair level): 20% test, and of the remaining 80% a further 10% validation (720 train / 80 val / 200 test).
Validation is used only for early stopping (patience 8), checkpoint selection and the decision threshold;
the test set is scored once. The script asserts that no pair is in two sets.

Outputs in `xunet_results/`: `paper_table.md` (paste into Table 9), `summary.csv`, `trials.csv`, `config.json`,
`splits_<bpp>_seed<seed>.json` (exact pair indices for each split).
Send back `paper_table.md`, `summary.csv`, `trials.csv` and the console output.
