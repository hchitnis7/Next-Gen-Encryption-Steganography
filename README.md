# Next-Gen Encryption Steganography

A research-grade, multi-layer secure communication system combining **post-quantum cryptography**, **GAN-generated encryption keys**, and **image steganography** to hide encrypted messages imperceptibly inside images. The system has gone through two major architectural generations — this README documents the current, fully updated system.

> 📄 This repository is the implementation basis for the paper:  
> **"A Robust Two-Level Security Model Integrating Post-Quantum Encryption and Image Steganography"**  
> *Under second-round review — Journal of Image and Graphics (JOIG), Scopus indexed, CiteScore 4.3 (2025)*  
> *First author: Harsh Chitnis*

---

## Table of Contents

- [Overview](#overview)
- [System Architecture](#system-architecture)
- [GAN Architecture](#gan-architecture)
- [What Changed Between Versions](#what-changed-between-versions)
  - [GAN: v1 → v2 (CHAOSDCGAN → CHAOSDCGAN_UPDATED)](#gan-v1--v2)
  - [Steganography: Single-channel → 3-channel (IWT → IWT_FAST_3CHANNEL)](#steganography-v1--v2)
  - [Pipeline: Embedded keys → Separated key channel](#pipeline-v1--v2)
  - [XOR: Pure Python → Numba-accelerated](#xor-v1--v2)
- [Repository Structure](#repository-structure)
- [File Reference](#file-reference)
- [Getting Started](#getting-started)
- [API Reference](#api-reference)
- [Steganalysis Benchmarking](#steganalysis-benchmarking)
- [Evaluation & Metrics](#evaluation--metrics)
- [Dependencies](#dependencies)
- [Known Limitations & Future Work](#known-limitations--future-work)
- [License](#license)

---

## Overview

Standard encryption protects data if intercepted, but reveals that a secret communication is occurring. Steganography hides the communication itself. This project combines both:

**Sender side:**
1. Plaintext is triple-encrypted: NTRUEncrypt (post-quantum) → AES-256-GCM → XOR obfuscation
2. All encryption keys are derived from a GAN-generated high-entropy byte pool
3. Session keys (AES key + XOR mask) are encapsulated separately using NTRU and sent via an independent key channel
4. The ciphertext is embedded invisibly into a cover image across all 3 RGB channels using IWT steganography
5. The stego image is optionally uploaded to cloud storage with a QR code for distribution

**Receiver side:**
1. Retrieve stego image (via QR/cloud or direct path)
2. Separately receive and NTRU-decapsulate the key bundle to recover session keys
3. Extract ciphertext from the stego image via IWT decoding
4. Reverse all encryption layers: XOR → AES-GCM → NTRU → plaintext

The critical security property: **an attacker who intercepts the stego image gets only ciphertext. An attacker who intercepts the key bundle gets only NTRU-encrypted key material. Both channels are required simultaneously to recover the plaintext.**

---

## System Architecture

![System Architecture](SYSTEM_ARCHITECTURE.png)

The full pipeline operates across two separate channels:

**Data channel (stego image):** carries the XOR-obfuscated AES-GCM ciphertext, embedded invisibly in the image pixels. No key material is present in this channel.

**Key channel (key bundle):** carries the AES-256 session key and XOR mask, both NTRU-encrypted into a single binary bundle and transmitted independently of the image.

The encryption and decryption sequences are:

```
ENCRYPTION (sender):
  Plaintext
    ↓ NTRU Encryption       (post-quantum, lattice-based)
    ↓ AES-256-GCM           (GAN-derived key + IV, not embedded in image)
    ↓ XOR Obfuscation       (GAN-derived mask, not embedded in image)
    ↓ Base64 Encoding
    ↓ IWT Steganography     (3-channel BGR embedding, 4096×4096)
    → Stego Image  ─────────────────────────────────→  Data Channel

  Session Keys (AES key + XOR mask)
    ↓ NTRU Key Encapsulation
    → Key Bundle  ──────────────────────────────────→  Key Channel (separate)

DECRYPTION (receiver):
  Key Channel  → Key Bundle → NTRU Key Decapsulation → AES Key + XOR Mask
  Data Channel → Stego Image → IWT Extraction → Base64 Decode
    ↓ XOR Decryption        (key from key channel)
    ↓ AES-256-GCM Decryption (key from key channel)
    ↓ NTRU Decryption
    → Plaintext
```

---

## GAN Architecture

![GAN Architecture](GAN_ARCHITECTURE.png)

The GAN is used exclusively to generate high-entropy byte sequences that serve as session key material. It is **not** used for steganographic embedding.

### Generator (training + inference)

The updated generator uses **resize-convolution** (Upsample + Conv2d) throughout its upsampling path, replacing the original ConvTranspose2d(stride=2) blocks:

```
Latent z: N(0,I), shape 1×1×1024
    ↓ Projection Block: ConvTranspose2d(1→4), BN, ReLU, NoiseInjection, Dropout
    ↓ Generator Block 1: Upsample + Conv2d, BN, ReLU, NoiseInjection, Dropout  [4×4 → 8×8]
    ↓ Generator Block 2: Upsample + Conv2d, BN, ReLU, NoiseInjection, Dropout  [8×8 → 16×16]
    ↓ Generator Block 3: Upsample + Conv2d, BN, ReLU, NoiseInjection, Dropout  [16×16 → 32×32]
    ↓ Final Block: Upsample + Conv2d, Tanh activation                           [32×32 → 64×64]
    → Raw output bounded [-1, 1], 6-channel image
```

**At inference only:**
```
Raw Generator Output [-1,1]
    ↓ CDF Quantisation     (rank-based mapping to uniform [0, 255])
    ↓ Cryptographic Shuffle (os.urandom-seeded, breaks spatial correlations)
    → Final Random Byte Stream
```

### Discriminator (training only)

```
Real image from CSPRNG dataset  ─┐
Generated fake image             ─┴→ Input Noise Injection (Gaussian σ=0.10, both real and fake)
    ↓ Discriminator Block 1: Spectral-Norm Conv, LeakyReLU + Dropout2d  [no BatchNorm]
    ↓ Discriminator Block 2: Spectral-Norm Conv, BatchNorm + LeakyReLU + Dropout2d
    ↓ Discriminator Block 3: Spectral-Norm Conv, BatchNorm + LeakyReLU + Dropout2d
    ↓ Final Block: Spectral-Norm Conv, raw logit — NO Sigmoid
    → Real/Fake Logit Score  →  BCEWithLogitsLoss
```

---

## What Changed Between Versions

### GAN: v1 → v2

**`CHAOSDCGAN.py` (v1) → `CHAOSDCGAN_UPDATED.py` (v2)**

| Component | v1 (Original) | v2 (Updated) | Why |
|---|---|---|---|
| Upsampling | `ConvTranspose2d(stride=2)` | `Upsample(nearest) + Conv2d` | ConvTranspose2d with stride=2 creates uneven gradient weighting at alternating spatial positions → checkerboard pattern in output → periodic autocorrelation in byte stream |
| Output activation | None (raw unbounded floats) | `Tanh` (bounded `[-1, 1]`) | Raw float32 byte-casting clusters bits in IEEE 754 exponent fields → non-uniform byte distribution requiring post-processing patches |
| Noise injection scale | 0.3 | 0.15 | Original scale was collapsing gradients during training |
| Dropout type | `Dropout2d` (drops entire feature maps) | `Dropout` (element-wise) | Dropout2d creates spatial correlations by zeroing full channels; element-wise dropout doesn't |
| Byte extraction | Raw `.tobytes()` on float32 arrays | **CDF quantisation** (rank transform) | Guarantees uniform marginal byte distribution by construction — no patches needed |
| Post-processing | CSPRNG mode-replacement loop (iterative) | **Cryptographic shuffle** (one-pass) | The mode-replacement loop was patching the float-casting symptom; shuffle breaks spatial correlations directly |
| Speed | Slow for large `num_bytes` due to iterative loop | Significantly faster | No iterative convergence loop |
| API | Same public functions | Same public functions (backward-compatible) | Legacy stubs retained so existing callers don't break |

**CDF Quantisation — how it works:**

The rank transform maps each float value to its position in the sorted order of all values: if X ~ F (any continuous distribution), then rank(X)/N ~ Uniform[0,1]. This holds regardless of the generator's raw output distribution, guaranteeing a flat byte histogram without any post-processing. The cryptographic shuffle then uses an `os.urandom`-seeded RNG to reorder bytes, destroying the spatial ordering introduced by the convolutional architecture and bringing lag-1 autocorrelation close to 0.

---

### Steganography: v1 → v2

**`IWT.py` (v1) → `IWT_FAST_3CHANNEL.py` (v2)**

| Component | v1 | v2 | Why |
|---|---|---|---|
| Channels used | Blue channel only | All 3 BGR channels | 3× the embedding capacity |
| Capacity (4096×4096) | ~4 MB | ~12 MB | Full tripling |
| Numba JIT | No | Yes (`@njit(cache=True)`) | Significant speedup on IWT forward/inverse passes |
| Traversal order | Implicit (single channel) | Explicit canonical contract | Encoder and decoder share a single `_iter_coefficients` generator — guarantees identical traversal regardless of channel/block boundaries |
| Bit representation | Per-character list of lists | Flat bit stream | Cleaner, allows prefix to span channel boundaries |
| Capacity check | Rough estimate | `capacity_bytes()` method | Accurate, accounts for length prefix overhead |
| Grayscale input | Not handled | Auto-converted to BGR | Defensive |
| Boundary handling | `±1` random nudge | Same, but extracted to `_embed_bit_into_coefficient()` | Cleaner separation |
| Output path | Hardcoded | Configurable `output_path` parameter | More flexible |
| Self-test | None | Built-in round-trip test with synthetic images | Verifiable correctness |

**`IWT_FAST.py`** is an intermediate variant — Numba-accelerated IWT on the blue channel only (no 3-channel extension). Used in `IWT_CLOUD_ENC_DEC_numba.py`.

**Capacity formula for 3-channel IWT:**
```
blocks_per_channel = (width // 8) × (height // 8)
bits_per_channel   = blocks_per_channel × 4 subbands × 16 coefficients
total_capacity     = (bits_per_channel × 3 channels) / 8  −  16 bytes (prefix overhead)

For 4096×4096: (512×512) × 64 × 3 / 8 = 12,582,912 bytes ≈ 12 MB
```

---

### Pipeline: v1 → v2

**`IWT_CLOUD_ENC_DEC.py` (v1) → `IWT_CLOUD_ENC_DEC_numba_key_encap.py` (v2)**

| Component | v1 | v2 | Why |
|---|---|---|---|
| AES key in ciphertext | Yes — key embedded in output string (`key:iv:ciphertext:tag`) | No — key returned separately, NOT in image | Embedding the key in the stego image means an attacker extracting the ciphertext also gets the AES key |
| XOR key in ciphertext | Yes — prepended to output (`xor_key:xor_encrypted`) | No — key returned separately | Same reason |
| Key transport | None (self-contained) | NTRU key encapsulation → `key_bundle.bin` | Session keys travel on an independent channel; two-channel security model |
| Key encapsulation | N/A | `encapsulate_session_keys(aes_key, xor_key)` | AES key + XOR mask NTRU-encrypted into a single bundle |
| Key decapsulation | N/A | `decapsulate_session_keys(key_bundle)` | Recovers both keys from bundle on receiver side |
| IWT module used | `IWT.py` (blue channel) | `IWT_FAST_3CHANNEL.py` (3-channel) | Capacity and speed |
| GAN module used | `CHAOSDCGAN.py` | `CHAOSDCGAN_UPDATED.py` | Updated architecture |
| Cloud upload | Always attempted | Optional (`cloud_upload=True/False`) | More flexible for local-only testing |
| `master_encrypt` returns | `None` | `(encoded_img, qr_code_img, key_bundle_bytes)` | Key bundle must be accessible to caller for transmission |
| `master_decrypt` args | `input_data, downloaded_file_path` | `input_data, key_bundle_path, downloaded_file_path, qr` | Key bundle is now a required separate input |
| NTRU key generation | Fresh keys generated per call | Pre-existing named key (`dead2keys`) | Avoids per-call key generation overhead; keys persist across sessions |
| `qr` flag | N/A | `qr=True/False` | Allows direct file path decrypt without cloud/QR |

---

### XOR: v1 → v2

**Pure Python `xor_cipher` → Numba `@njit xor_cipher_numba`**

The original XOR operated character-by-character in Python. The updated version uses a `@njit`-compiled uint8 array XOR with automatic key cycling, which is orders of magnitude faster for long ciphertexts. The Python wrapper `xor_cipher()` maintains the same string-in/string-out API.

---

## Repository Structure

```
Next-Gen-Encryption-Steganography/
│
├── ── GAN KEY GENERATION ──────────────────────────────────────────────────────
├── CHAOSDCGAN.py                       # v1: ConvTranspose2d, raw floats, CSPRNG loop
├── CHAOSDCGAN_UPDATED.py               # v2: Resize-conv, Tanh, CDF quantisation + shuffle ✅
│
├── ── STEGANOGRAPHY ───────────────────────────────────────────────────────────
├── IWT.py                              # v1: Blue-channel only, pure Python IWT
├── IWT_FAST.py                         # v1.5: Blue-channel, Numba-accelerated IWT
├── IWT_FAST_3CHANNEL.py                # v2: 3-channel BGR, Numba, explicit traversal ✅
│
├── ── FULL PIPELINE ───────────────────────────────────────────────────────────
├── IWT_CLOUD_ENC_DEC.py                # v1: keys embedded in ciphertext string
├── IWT_CLOUD_ENC_DEC_numba.py          # v1.5: Numba XOR + updated GAN + IWT_FAST
├── IWT_CLOUD_ENC_DEC_numba_key_encap.py # v2: separated key channel, NTRU encapsulation ✅
│
├── ── SUPPORTING MODULES ──────────────────────────────────────────────────────
├── CloudQRHandler.py                   # GCS upload, QR code generation and reading
├── STREAMGUI.py                        # Streamlit interactive GUI
├── ENC_EVAL.py                         # Image quality + steganalysis evaluation
│
├── ── NOTEBOOKS ───────────────────────────────────────────────────────────────
├── DCGANTRAINEVAL.ipynb                # v1 GAN training
├── DCGANTRAINEVAL_UPDATES.ipynb        # v2 GAN training (resize-conv architecture) ✅
├── BANGAYAAA.ipynb                     # End-to-end demo
├── metrics.ipynb                       # Standalone GAN byte quality evaluation
├── alaska2_steganalysis_benchmark.ipynb        # ALASKA2 steganalysis benchmark v1
├── alaska2_steganalysis_benchmark_v2_GPU.ipynb # GPU-accelerated benchmark v2
├── alaska2_steganalysis_benchmark_v3_checkpointed.ipynb # Checkpointed v3 ✅
│
├── ── STEGANALYSIS RESULTS ────────────────────────────────────────────────────
├── steganalysis_results/               # v1 benchmark: PSNR, SSIM, chi-square, ROC curves
│   ├── chi_square.png / .pdf
│   ├── composite_results.png / .pdf
│   ├── experimental_table.csv / .tex
│   ├── pixel_diff_distribution.png / .pdf
│   ├── quality_metrics.png / .pdf
│   ├── roc_curves.png / .pdf
│   ├── rs_analysis.png / .pdf
│   └── visual_distortion_{0.1,0.2,0.4}bpp.png
├── steganalysis_results_v2/            # v2 benchmark: detector comparison ROC
│   ├── detector_comparison_roc.png / .pdf
│   └── experimental_table_v2.csv / .tex
│
├── ── PRETRAINED MODELS ───────────────────────────────────────────────────────
├── expanded_generator_dcgan.pth        # v2 generator weights (resize-conv) ✅
├── expanded_generator_dcgan_9CHAN_GPT.pth    # v1 generator weights
├── expanded_discriminator_dcgan.pth          # v2 discriminator weights ✅
├── expanded_discriminator_dcgan_9CHAN_GPT.pth # v1 discriminator weights
│
├── ── OTHER ───────────────────────────────────────────────────────────────────
├── keys.bin                            # Example NTRU key bundle output
├── download_alaska2_covers.sh          # Script to download ALASKA2 dataset
├── encoded_lena.png                    # Sample stego image output
├── lena.png                            # Sample cover image
├── downloaded_file*.png                # Sample decoded stego outputs
├── testqr*.png                         # Sample QR code outputs
├── LICENSE                             # Apache-2.0
└── README.md
```

✅ = current recommended version

---

## File Reference

### `CHAOSDCGAN_UPDATED.py` ✅ (current GAN module)

**`NoiseInjection`** — injects per-element Gaussian noise (scale=0.15) during training only. Reduced from 0.3 to prevent gradient collapse.

**`ResizeConvBlock`** — replaces ConvTranspose2d(stride=2). Upsample(nearest) + Conv2d(3×3) + BN + ReLU + NoiseInjection + Dropout(element-wise). Eliminates checkerboard artifacts from stride-2 overlap ambiguity.

**`Generator`** — full resize-conv DCGAN. z(1024×1×1) → 6-channel 64×64 image, bounded [-1,1] via Tanh. The projection block uses ConvTranspose2d(1→4) safely (no stride-2 overlap at 1×1 input).

**`_cdf_quantise_and_shuffle(images, target_bytes)`** — core byte extraction:
1. Concatenates all float values from generated images
2. Double-argsort to get each value's rank
3. Maps ranks linearly to [0, 255] → perfectly uniform byte histogram by construction
4. Seeds NumPy RNG from `os.urandom(8)` → shuffles byte array to break spatial correlations

**`inference(model_path, num_bytes)`** — generates `num_bytes` of high-entropy bytes. Batches generation with `BATCH_SIZE=32` images per forward pass. No iterative convergence loop — one CDF pass per batch, concatenated until target reached.

**Legacy stubs** — `smart_replace_mode_csprng`, `smart_reduce_mode_frequency_csprng`, `extract_random_bytes_float_no_sigmoid` are kept as no-ops/wrappers for backward API compatibility.

---

### `IWT_FAST_3CHANNEL.py` ✅ (current steganography module)

**`haar_lifting_iwt2_numba` / `haar_lifting_iiwt2_numba`** — Numba `@njit(cache=True)` compiled Haar lifting IWT. JIT-compiled on first import; subsequent calls are native speed. Warmed up at module level on a dummy block so the first real call has no compilation penalty.

**`IWT.SUBBAND_SLICES`** — class-level constant defining subband visit order: HH → LL → LH → HL. Shared between encoder and decoder as the single source of truth for traversal order.

**`_iter_coefficients(iwt_blocks_bgr)`** — canonical generator that yields `(ch_idx, blk_idx, row_slice, col_slice, coeff_idx, flat)` tuples. Both `encode_image` and `decode_image` call this with identical arguments, guaranteeing that bit position N in the encoder corresponds to bit position N in the decoder, even across channel boundaries.

**`_to_bits(message)`** — converts string to a flat list of bits (MSB first), unlike the original which produced a list of 8-element lists per character. Allows the length prefix to span channel boundaries cleanly.

**`_embed_bit_into_coefficient(value, bit)`** — extracted helper for LSB embedding with boundary handling: value=0 → set to 1, value=255 → set to 254, otherwise random ±1 nudge to avoid systematic bias.

**`capacity_bytes(img)` / `capacity_report(img)`** — accurate capacity calculation and human-readable breakdown.

**`encode_image(img, secret_msg, output_path)`** — full 3-channel encode. Applies forward IWT to all blocks across all channels, embeds the full bit stream in canonical traversal order, applies inverse IWT, reconstructs and merges channels.

**`decode_image(img)`** — full 3-channel decode. Same traversal order as encoder. Reads the length prefix first, stops exactly at message end even if it spans channel boundaries.

---

### `IWT_CLOUD_ENC_DEC_numba_key_encap.py` ✅ (current pipeline)

**`_sample_gan_bytes(n)`** — draws n bytes from the GAN pool using `secrets.SystemRandom().sample()` (CSPRNG index selection without positional bias).

**`encrypt_message(plaintext, aes_key=None)`** — AES-256-GCM encryption. Returns `(ciphertext_string, aes_key)` where `ciphertext_string` is `iv:ciphertext:tag` — **the AES key is NOT included** and must be handled separately.

**`decrypt_message(ciphertext_string, aes_key)`** — takes explicit `aes_key` argument (not parsed from string). Raises `ValueError` on GCM tag mismatch.

**`xor_encrypt(aes_encrypted_string, xor_key_bytes=None)`** — returns `(xor_encrypted_string, xor_key_bytes)`. **The XOR key is NOT prepended** to the output — handled separately via key bundle.

**`xor_decrypt(xor_encrypted_string, xor_key_bytes)`** — takes explicit `xor_key_bytes` argument.

**`encapsulate_session_keys(aes_key, xor_key_bytes, ntru_key_name)`** — concatenates `aes_key.hex() + ":" + xor_key_bytes.hex()` then NTRU-encrypts the combined material. Returns bytes for binary file I/O.

**`decapsulate_session_keys(key_bundle_bytes, ntru_key_name)`** — NTRU-decrypts the bundle and splits on `:` to recover both keys. Raises `ValueError` on parse failure.

**`master_encrypt(plaintext, ..., key_bundle_path, cloud_upload)`** — full sender pipeline. Writes stego image and key bundle to separate files. Returns `(encoded_img, qr_code_img, key_bundle_bytes)`.

**`master_decrypt(input_data, key_bundle_path, downloaded_file_path, qr)`** — full receiver pipeline. Reads key bundle from `key_bundle_path`, stego image from `input_data` (direct path if `qr=False`, QR decode + cloud download if `qr=True`).

---

### `ENC_EVAL.py` — Image Quality & Steganalysis Evaluation

Provides the `ENC_EVAL` class for assessing steganographic imperceptibility:

- `calculate_metrics()` — MSE, PSNR (dB), SSIM, entropy difference between cover and stego
- `chi_square_test()` — chi-square test on pixel histograms; statistic <50 = no significant modification detected
- `high_frequency_noise_analysis()` — 2D FFT comparison; detects high-frequency embedding artifacts
- `histogram_analysis()` — side-by-side RGB histogram plots for visual inspection

---

### `alaska2_steganalysis_benchmark_v3_checkpointed.ipynb` ✅ (current benchmark)

Benchmarks the steganography against the ALASKA2 dataset using established steganalysis detectors. Features:
- GPU-accelerated inference
- Checkpointing to resume interrupted evaluations
- ROC curve comparison across multiple detectors (`steganalysis_results_v2/detector_comparison_roc.png`)

Download the ALASKA2 dataset with `download_alaska2_covers.sh` before running.

---

### `CloudQRHandler.py`

- Initialises a Google Cloud Storage client from a service account JSON
- `upload_file(local_path, blob_name)` → public URL
- `generate_qr_code(url, output_path)` → QR code PNG
- `read_qr_code(image_path)` → decoded URL string
- `download_file(url, local_path)` → saves stego image locally

---

### `STREAMGUI.py`

Streamlit web interface for interactive encrypt/decrypt. Run with:
```bash
streamlit run STREAMGUI.py
```

---

## Getting Started

### Prerequisites

- Python 3.8+
- CUDA-capable GPU (recommended for GAN inference; CPU fallback available)
- Google Cloud Storage account + service account credentials JSON (for cloud upload/QR; optional)
- NTRU key pair generated under the name `dead2keys` via `pq_ntru.generate_keys()`

### Installation

```bash
git clone https://github.com/hchitnis7/Next-Gen-Encryption-Steganography.git
cd Next-Gen-Encryption-Steganography
pip install torch numpy opencv-python pycryptodome pq-ntru matplotlib scipy \
            qrcode google-cloud-storage streamlit numba scikit-image
```

### Configuration

Update credentials in `IWT_CLOUD_ENC_DEC_numba_key_encap.py` if using cloud upload:
```python
credentials_path = "/path/to/your/gcs-service-account.json"
bucket_name      = "your-gcs-bucket-name"
MODEL_PATH_G     = "./expanded_generator_dcgan.pth"   # v2 weights — included in repo
```

Generate NTRU keys once before first use:
```python
import pq_ntru
pq_ntru.generate_keys("dead2keys", mode="high")
```

### Train the GAN (optional — pretrained weights included)

```bash
jupyter notebook DCGANTRAINEVAL_UPDATES.ipynb
```

This produces `expanded_generator_dcgan.pth` and `expanded_discriminator_dcgan.pth`. The pretrained weights are already in the repository.

### Run the Full Pipeline

```python
from IWT_CLOUD_ENC_DEC_numba_key_encap import master_encrypt, master_decrypt

# ── SENDER ──
encoded_img, qr_img, key_bundle = master_encrypt(
    plaintext           = "Your secret message",
    cover_image_path    = "lena.png",
    output_path         = "stego.png",
    key_bundle_path     = "keys.bin",    # transmit this separately!
    display             = True,
    cloud_upload        = False,
)
# Send stego.png via one channel, keys.bin via a separate channel.

# ── RECEIVER ──
message = master_decrypt(
    input_data        = "stego.png",
    key_bundle_path   = "keys.bin",
    qr                = False,
)
print("Recovered:", message)
```

Or launch the GUI:
```bash
streamlit run STREAMGUI.py
```

---

## API Reference

### `master_encrypt` (v2)

```python
master_encrypt(
    plaintext:            str,
    cover_image_path:     str  = "lena2.png",
    output_path:          str  = "encoded_image.png",
    output_qr_code_path:  str  = "qr_code.png",
    key_bundle_path:      str  = "key_bundle.bin",
    display:              bool = False,
    cloud_upload:         bool = False,
) -> tuple[np.ndarray, np.ndarray | None, bytes]
```

Returns `(encoded_img, qr_code_img, key_bundle_bytes)`. Writes both `output_path` and `key_bundle_path` to disk. The two output files must be transmitted via separate channels.

---

### `master_decrypt` (v2)

```python
master_decrypt(
    input_data:           str  = None,
    key_bundle_path:      str  = "key_bundle.bin",
    downloaded_file_path: str  = "downloaded_file.png",
    qr:                   bool = True,
) -> str
```

If `qr=False`, reads stego image directly from `input_data`. If `qr=True`, decodes QR code and downloads from GCS. Requires `key_bundle_path` in all cases.

---

### `IWT.encode_image` (v2, 3-channel)

```python
iwt = IWT()
stego = iwt.encode_image(img, secret_msg, output_path=None) -> np.ndarray | False
```

Returns stego image (BGR uint8) or `False` if message exceeds capacity.

---

### `IWT.decode_image` (v2, 3-channel)

```python
message = iwt.decode_image(img) -> str
```

Traverses all 3 channels in canonical order. Returns extracted message string.

---

### `IWT.capacity_bytes` (v2)

```python
max_bytes = iwt.capacity_bytes(img) -> int
```

Returns maximum embeddable bytes for the given image dimensions.

---

### `CHAOSDCGAN_UPDATED.inference`

```python
from CHAOSDCGAN_UPDATED import inference
random_bytes = inference(model_path="./expanded_generator_dcgan.pth", num_bytes=10_000_000) -> bytes
```

---

### `encapsulate_session_keys` / `decapsulate_session_keys`

```python
key_bundle = encapsulate_session_keys(aes_key: bytes, xor_key_bytes: bytes) -> bytes
aes_key, xor_key_bytes = decapsulate_session_keys(key_bundle: bytes) -> tuple[bytes, bytes]
```

---

## Steganalysis Benchmarking

The repository includes three steganalysis benchmark notebooks evaluated against the ALASKA2 dataset. Results are stored in `steganalysis_results/` (v1) and `steganalysis_results_v2/` (v2).

**Metrics evaluated:**

| Metric | Description | Where |
|---|---|---|
| PSNR | Peak Signal-to-Noise Ratio (dB) | `quality_metrics.png` |
| SSIM | Structural Similarity Index | `quality_metrics.png` |
| Chi-Square | Pixel histogram uniformity test | `chi_square.png` |
| RS Analysis | Regular-Singular steganalysis | `rs_analysis.png` |
| Pixel Difference Distribution | Cover vs stego pixel delta histogram | `pixel_diff_distribution.png` |
| ROC Curves (v1) | Detector performance across payloads | `roc_curves.png` |
| Detector Comparison ROC (v2) | Multiple steganalysis detectors vs IWT | `detector_comparison_roc.png` |
| Visual Distortion | Side-by-side at 0.1, 0.2, 0.4 bpp | `visual_distortion_*.png` |

Download ALASKA2 covers before running:
```bash
bash download_alaska2_covers.sh
```

---

## Evaluation & Metrics

### GAN Key Quality

```python
from CHAOSDCGAN_UPDATED import inference, evaluate_all
data    = inference()
metrics = evaluate_all(data)
```

| Metric | Ideal | Description |
|---|---|---|
| Shannon Entropy | ~8.0 bits/byte | Unpredictability of byte distribution |
| KS Statistic | p > 0.05 | Kolmogorov-Smirnov against Uniform; KS capped at 50k samples to avoid over-sensitivity |
| Chi-Square | p > 0.05 | Goodness-of-fit to uniform over 256 bins |
| Autocorrelation (lag-1) | ~0 | Sequential dependence; ~0 means cryptographic shuffle worked |
| Bitwise Balance | ~50% ones | Bit-level uniformity |

### Steganographic Imperceptibility

```python
from ENC_EVAL import ENC_EVAL
evaluator = ENC_EVAL("cover.png", "stego.png")
evaluator.calculate_metrics()
evaluator.chi_square_test()
evaluator.high_frequency_noise_analysis()
evaluator.histogram_analysis()
```

| Metric | Description |
|---|---|
| MSE | Mean Squared Error; ideally <1.0 |
| PSNR | Higher = less visual distortion; typically >40 dB for good steganography |
| SSIM | Structural similarity; values >0.99 indicate imperceptibility |
| Entropy Difference | Should be near 0 |
| Chi-Square | <50 = no significant steganographic modification detected |

---

## Dependencies

```
torch              # GAN training and inference
numpy              # Numerical operations throughout
opencv-python      # Image I/O and processing
pycryptodome       # AES-256-GCM (Crypto.Cipher)
pq-ntru            # Post-quantum NTRUEncrypt
matplotlib         # Visualisation
scipy              # Statistical tests (KS, chi-square, FFT)
numba              # JIT compilation for IWT and XOR
scikit-image       # PSNR, SSIM metrics
qrcode             # QR code generation
google-cloud-storage  # GCS upload/download (optional)
streamlit          # GUI (optional)
```

---

## Known Limitations & Future Work

- **Cover image size** — pipeline resizes all cover images to 4096×4096, which distorts non-square images. Adaptive resizing or padding would be cleaner.
- **NTRU key persistence** — NTRU key pair `dead2keys` must exist on both sender and receiver machines. Key distribution is out of scope and handled manually.
- **GCS dependency** — `cloud_upload=True` requires valid GCS credentials. The pipeline works fully locally with `cloud_upload=False, qr=False`.
- **GAN inference time** — generating 10M bytes takes several minutes on CPU. A GPU is strongly recommended. The CDF approach is significantly faster than the v1 CSPRNG loop but inference is still the bottleneck.
- **Key bundle transmission** — no built-in channel for key bundle delivery. Currently written to a local file; production use would require a separate secure channel (e.g., Signal, encrypted email).

---

## License

Apache-2.0 — see [LICENSE](LICENSE) for details.

---

*Implementation and research by Harsh Chitnis — Uppsala University / University of Mumbai.*
