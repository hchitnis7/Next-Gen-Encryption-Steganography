#!/usr/bin/env python3
"""
IWT_CLOUD_ENC_DEC_numba.py

Implements the full hybrid encryption-steganography pipeline:

    Encryption pipeline (sender):
        Plaintext
            → NTRU post-quantum encryption
            → AES-256-GCM symmetric encryption   (GAN-derived key + IV)
            → XOR obfuscation                    (GAN-derived mask)
            → IWT 3-channel steganographic embedding
            → Stego image (data channel)

        Session keys (AES key + XOR mask)
            → NTRU public-key encapsulation
            → Key bundle (key channel, transmitted separately)

    Decryption pipeline (receiver):
        Key bundle  →  NTRU decapsulation  →  AES key + XOR mask
        Stego image →  IWT extraction      →  XOR-encrypted ciphertext
                    →  XOR decryption      →  AES ciphertext
                    →  AES-GCM decryption  →  NTRU ciphertext
                    →  NTRU decryption     →  Plaintext

KEY MANAGEMENT ARCHITECTURE:
    Session keys are never embedded within the stego image. The AES-256 key
    and XOR mask are encapsulated using the recipient's NTRU public key and
    transmitted on a separate channel (key_bundle.bin or equivalent). This
    ensures that compromising the stego image (data channel) yields only
    ciphertext, while compromising the key bundle yields only key material
    — an attacker requires both to recover the plaintext.

CHANGES FROM PREVIOUS VERSION:
    - encrypt_message(): no longer embeds the AES key or IV in the returned
      string. Returns (ciphertext_string, aes_key) where ciphertext_string
      contains only iv:ciphertext:tag.
    - decrypt_message(): accepts aes_key as an explicit argument rather than
      parsing it from the encrypted string.
    - xor_encrypt(): no longer prepends the XOR key to the output. Returns
      (xor_encrypted_string, xor_key_bytes) for separate key handling.
    - xor_decrypt(): accepts xor_key_bytes as an explicit argument.
    - encapsulate_session_keys(): new — NTRU-encrypts both session keys into
      a transmissible key bundle.
    - decapsulate_session_keys(): new — NTRU-decrypts the key bundle to
      recover AES key and XOR mask.
    - master_encrypt(): returns (encoded_img, qr_code_img, key_bundle_bytes).
      Writes key bundle to key_bundle_path (default: "key_bundle.bin").
    - master_decrypt(): accepts key_bundle_path; no longer self-contained —
      the key channel and data channel are separate inputs.

IMPORTS:
    - CHAOSDCGAN_UPDATED as cdc  (drop-in replacement for CHAOSDCGAN_UPDATED)
    - IWT_FAST_3CHANNEL as IWT_FAST  (3-channel IWT steganography)
"""

import os
import cv2
import base64
import time
import secrets
import numpy as np
import torch
import matplotlib.pyplot as plt
from datetime import datetime
from numba import njit

from Crypto.Cipher import AES
import pq_ntru

import CHAOSDCGAN_UPDATED as cdc
import IWT_FAST_3CHANNEL as IWT_FAST
import CloudQRHandler as cloudthing

# ─────────────────────────────────────────────────────────────────────────────
# Cloud handler initialisation
# ─────────────────────────────────────────────────────────────────────────────

credentials_path = "/teamspace/studios/this_studio/project_dir/cryptonovademo-d887db293060.json"
bucket_name      = "cryptonova"
cloud_handler    = cloudthing.CloudQRHandler(credentials_path, bucket_name)

# ─────────────────────────────────────────────────────────────────────────────
# Global configuration
# ─────────────────────────────────────────────────────────────────────────────

DEVICE       = torch.device("cuda" if torch.cuda.is_available() else "cpu")
LATENT_DIM   = 1024
BATCH_SIZE   = 32
MODEL_PATH_G = "/teamspace/studios/this_studio/expanded_generator_dcgan.pth"
NTRU_KEY_NAME = "dead2keys"   # NTRU key-pair identifier used by pq_ntru

# ─────────────────────────────────────────────────────────────────────────────
# GAN byte pool — generated once at import time and used for all key material
# ─────────────────────────────────────────────────────────────────────────────

gan_bytes_all = cdc.inference()

# ─────────────────────────────────────────────────────────────────────────────
# Numba-accelerated XOR core
# ─────────────────────────────────────────────────────────────────────────────

@njit
def xor_cipher_numba(data_arr: np.ndarray, key_arr: np.ndarray) -> np.ndarray:
    """
    Bitwise XOR of data_arr against cyclically repeated key_arr.
    Both arrays must be dtype uint8. Returns a uint8 result array of
    the same length as data_arr.
    """
    result  = np.empty_like(data_arr)
    key_len = key_arr.shape[0]
    for i in range(data_arr.shape[0]):
        result[i] = data_arr[i] ^ key_arr[i % key_len]
    return result


def xor_cipher(data: str, key: str) -> str:
    """
    Applies cyclic XOR between UTF-8 encoded data and key strings.
    Returns the result decoded as UTF-8 (with error replacement for
    any non-decodable bytes introduced by XOR).
    """
    data_bytes   = np.frombuffer(data.encode("utf-8"), dtype=np.uint8)
    key_bytes    = np.frombuffer(key.encode("utf-8"),  dtype=np.uint8)
    result_bytes = xor_cipher_numba(data_bytes, key_bytes)
    return result_bytes.tobytes().decode("utf-8", errors="ignore")

# ─────────────────────────────────────────────────────────────────────────────
# Session key sampling from the GAN byte pool
# ─────────────────────────────────────────────────────────────────────────────

def _sample_gan_bytes(n: int) -> bytes:
    """
    Draws n bytes from gan_bytes_all by selecting n random, non-repeating
    indices using secrets.SystemRandom (a CSPRNG-backed selector). This
    ensures that the GAN byte pool is sampled without positional bias.
    """
    assert len(gan_bytes_all) >= n, (
        f"GAN pool has {len(gan_bytes_all)} bytes; requested {n}."
    )
    indices = secrets.SystemRandom().sample(range(len(gan_bytes_all)), n)
    return bytes(gan_bytes_all[i] for i in indices)

# ─────────────────────────────────────────────────────────────────────────────
# AES-256-GCM encryption / decryption
# ─────────────────────────────────────────────────────────────────────────────

def encrypt_message(plaintext, aes_key: bytes = None) -> tuple:
    """
    Encrypts plaintext (bytes or str) using AES-256-GCM.

    Parameters
    ----------
    plaintext : bytes or str
        The data to encrypt. If bytes (e.g., NTRU ciphertext output), it is
        used directly. If str, it is UTF-8 encoded first.
    aes_key : bytes, optional
        32-byte AES-256 key. If None, a fresh key is sampled from gan_bytes_all.

    Returns
    -------
    ciphertext_string : str
        Colon-delimited hex string containing ONLY iv:ciphertext:tag.
        The AES key is NOT included — it is returned separately for
        out-of-band transmission via the key bundle.
    aes_key : bytes
        The 32-byte AES key used. The caller is responsible for encapsulating
        this via encapsulate_session_keys() and transmitting it separately.

    Notes
    -----
    The GCM nonce (IV) is 96 bits (12 bytes) as recommended by NIST SP 800-38D.
    A fresh IV is sampled from the GAN byte pool for every call, ensuring
    nonce uniqueness across sessions.
    """
    if aes_key is None:
        aes_key = _sample_gan_bytes(32)

    iv = _sample_gan_bytes(12)

    # Handle bytes input (e.g., NTRU ciphertext) and str input uniformly
    if isinstance(plaintext, str):
        plaintext_bytes = plaintext.encode("utf-8")
    else:
        plaintext_bytes = plaintext

    cipher          = AES.new(aes_key, AES.MODE_GCM, nonce=iv)
    ciphertext, tag = cipher.encrypt_and_digest(plaintext_bytes)

    # Key is intentionally EXCLUDED from the returned string.
    # Only iv, ciphertext, and the authentication tag are embedded.
    ciphertext_string = (
        iv.hex() + ":" +
        ciphertext.hex() + ":" +
        tag.hex()
    )

    return ciphertext_string, aes_key


def decrypt_message(ciphertext_string: str, aes_key: bytes):
    """
    Decrypts an AES-256-GCM ciphertext string produced by encrypt_message().

    Parameters
    ----------
    ciphertext_string : str
        Colon-delimited hex string in the form iv:ciphertext:tag,
        as returned by encrypt_message().
    aes_key : bytes
        The 32-byte AES key, recovered from the key bundle via
        decapsulate_session_keys().

    Returns
    -------
    bytes
        The decrypted plaintext bytes. The caller is responsible for
        further decoding (e.g., passing to NTRU decryption).

    Raises
    ------
    ValueError
        If GCM authentication tag verification fails, indicating that the
        ciphertext has been tampered with or the wrong key was supplied.
    """
    iv_hex, ciphertext_hex, tag_hex = ciphertext_string.split(":")

    iv         = bytes.fromhex(iv_hex)
    ciphertext = bytes.fromhex(ciphertext_hex)
    tag        = bytes.fromhex(tag_hex)

    cipher    = AES.new(aes_key, AES.MODE_GCM, nonce=iv)
    plaintext = cipher.decrypt_and_verify(ciphertext, tag)

    return plaintext

# ─────────────────────────────────────────────────────────────────────────────
# XOR obfuscation layer
# ─────────────────────────────────────────────────────────────────────────────

def xor_encrypt(aes_encrypted_string: str, xor_key_bytes: bytes = None) -> tuple:
    """
    Applies cyclic XOR obfuscation to the AES-encrypted string.

    Parameters
    ----------
    aes_encrypted_string : str
        The iv:ciphertext:tag string produced by encrypt_message().
    xor_key_bytes : bytes, optional
        12-byte raw XOR mask. If None, freshly sampled from gan_bytes_all.

    Returns
    -------
    xor_encrypted_string : str
        The XOR-obfuscated string. The XOR key is NOT prepended — it is
        returned separately for out-of-band transmission via the key bundle.
    xor_key_bytes : bytes
        The raw 12-byte XOR mask. The caller encapsulates this via
        encapsulate_session_keys().

    Notes
    -----
    The raw key bytes are Base64-encoded before being used as the XOR key
    string, ensuring all key characters are printable ASCII and the cyclic
    XOR operates on a reproducible, length-stable key string.
    """
    if xor_key_bytes is None:
        xor_key_bytes = _sample_gan_bytes(12)

    xor_key            = base64.urlsafe_b64encode(xor_key_bytes).decode("utf-8")
    xor_encrypted_string = xor_cipher(aes_encrypted_string, xor_key)

    # XOR key is intentionally NOT prepended to the output.
    return xor_encrypted_string, xor_key_bytes


def xor_decrypt(xor_encrypted_string: str, xor_key_bytes: bytes) -> str:
    """
    Reverses XOR obfuscation applied by xor_encrypt().

    Parameters
    ----------
    xor_encrypted_string : str
        The XOR-obfuscated string, as embedded in the stego image.
    xor_key_bytes : bytes
        The 12-byte raw XOR mask, recovered from the key bundle via
        decapsulate_session_keys().

    Returns
    -------
    str
        The original AES-encrypted string (iv:ciphertext:tag).
    """
    xor_key = base64.urlsafe_b64encode(xor_key_bytes).decode("utf-8")
    return xor_cipher(xor_encrypted_string, xor_key)

# ─────────────────────────────────────────────────────────────────────────────
# NTRU-based session key encapsulation / decapsulation
# ─────────────────────────────────────────────────────────────────────────────

# def encapsulate_session_keys(
#     aes_key:      bytes,
#     xor_key_bytes: bytes,
#     ntru_key_name: str = NTRU_KEY_NAME,
# ) -> bytes:
#     """
#     Encapsulates both session keys into a single NTRU-encrypted key bundle.

#     The AES key and XOR mask are concatenated as hex strings with a colon
#     delimiter, then encrypted using the recipient's NTRU public key. The
#     resulting key bundle is transmitted on a channel separate from the
#     stego image, ensuring that the data channel and key channel remain
#     independent.

#     Parameters
#     ----------
#     aes_key : bytes
#         The 32-byte AES-256 key.
#     xor_key_bytes : bytes
#         The 12-byte XOR mask.
#     ntru_key_name : str
#         The NTRU key identifier used by pq_ntru. Must match the key name
#         used during decapsulation. Defaults to NTRU_KEY_NAME.

#     Returns
#     -------
#     bytes
#         The NTRU-encrypted key bundle, safe for separate transmission.

#     Security note
#     -------------
#     An attacker who obtains the stego image (data channel) recovers only
#     XOR-obfuscated AES-GCM ciphertext. An attacker who obtains the key
#     bundle (key channel) recovers only NTRU ciphertext, which is secure
#     under the hardness of the NTRU lattice problem (Hoffstein et al., 1998)
#     and is considered post-quantum resistant. Recovery of the plaintext
#     requires simultaneous access to both channels and the NTRU private key.
#     """
#     key_material  = aes_key.hex() + ":" + xor_key_bytes.hex()
#     key_bundle    = pq_ntru.encrypt(ntru_key_name, key_material)
#     return key_bundle


# def decapsulate_session_keys(
#     key_bundle:    bytes,
#     ntru_key_name: str = NTRU_KEY_NAME,
# ) -> tuple:
#     """
#     Decapsulates the NTRU-encrypted key bundle to recover both session keys.

#     Parameters
#     ----------
#     key_bundle : bytes
#         The NTRU-encrypted key bundle produced by encapsulate_session_keys().
#     ntru_key_name : str
#         The NTRU key identifier used by pq_ntru. Must match the key name
#         used during encapsulation.

#     Returns
#     -------
#     aes_key : bytes
#         The recovered 32-byte AES-256 key.
#     xor_key_bytes : bytes
#         The recovered 12-byte XOR mask.

#     Raises
#     ------
#     ValueError
#         If the decrypted material cannot be parsed into two hex fields,
#         indicating key bundle corruption or a key-name mismatch.
#     """
#     key_material = pq_ntru.decrypt(ntru_key_name, key_bundle)

#     try:
#         aes_hex, xor_hex = key_material.split(":")
#         aes_key      = bytes.fromhex(aes_hex)
#         xor_key_bytes = bytes.fromhex(xor_hex)
#     except (ValueError, AttributeError) as exc:
#         raise ValueError(
#             "Key bundle decapsulation failed: could not parse recovered material. "
#             "Verify that the NTRU key name matches and the bundle is unmodified."
#         ) from exc

#     return aes_key, xor_key_bytes



def encapsulate_session_keys(
    aes_key:       bytes,
    xor_key_bytes: bytes,
    ntru_key_name: str = NTRU_KEY_NAME,
) -> bytes:
    """
    Encapsulates both session keys into a single NTRU-encrypted key bundle.

    The AES key and XOR mask are concatenated as hex strings with a colon
    delimiter, then encrypted using the recipient's NTRU public key. The
    resulting key bundle is transmitted on a channel separate from the
    stego image, ensuring that the data channel and key channel remain
    independent.

    Parameters
    ----------
    aes_key : bytes
        The 32-byte AES-256 key.
    xor_key_bytes : bytes
        The 12-byte XOR mask.
    ntru_key_name : str
        The NTRU key identifier used by pq_ntru. Must match the key name
        used during decapsulation. Defaults to NTRU_KEY_NAME.

    Returns
    -------
    bytes
        The NTRU-encrypted key bundle, safe for separate transmission.

    Security note
    -------------
    An attacker who obtains the stego image (data channel) recovers only
    XOR-obfuscated AES-GCM ciphertext. An attacker who obtains the key
    bundle (key channel) recovers only NTRU ciphertext, which is secure
    under the hardness of the NTRU lattice problem (Hoffstein et al., 1998)
    and is considered post-quantum resistant. Recovery of the plaintext
    requires simultaneous access to both channels and the NTRU private key.
    """
    key_material   = aes_key.hex() + ":" + xor_key_bytes.hex()
    ntru_encrypted = pq_ntru.encrypt(ntru_key_name, key_material)
    # pq_ntru.encrypt returns a str — encode to bytes for binary file I/O
    if isinstance(ntru_encrypted, str):
        ntru_encrypted = ntru_encrypted.encode("utf-8")
    return ntru_encrypted


def decapsulate_session_keys(
    key_bundle_bytes: bytes,
    ntru_key_name:    str = NTRU_KEY_NAME,
) -> tuple:

    """
    Decapsulates the NTRU-encrypted key bundle to recover both session keys.

    Parameters
    ----------
    key_bundle : bytes
        The NTRU-encrypted key bundle produced by encapsulate_session_keys().
    ntru_key_name : str
        The NTRU key identifier used by pq_ntru. Must match the key name
        used during encapsulation.

    Returns
    -------
    aes_key : bytes
        The recovered 32-byte AES-256 key.
    xor_key_bytes : bytes
        The recovered 12-byte XOR mask.

    Raises
    ------
    ValueError
        If the decrypted material cannot be parsed into two hex fields,
        indicating key bundle corruption or a key-name mismatch.
    """
    # pq_ntru.decrypt expects a str — decode if we stored bytes
    if isinstance(key_bundle_bytes, bytes):
        key_bundle_bytes = key_bundle_bytes.decode("utf-8")
    key_material = pq_ntru.decrypt(ntru_key_name, key_bundle_bytes)
    try:
        aes_hex, xor_hex  = key_material.split(":")
        aes_key           = bytes.fromhex(aes_hex)
        xor_key_bytes     = bytes.fromhex(xor_hex)
    except (ValueError, AttributeError) as exc:
        raise ValueError(
            "Key bundle decapsulation failed: could not parse recovered material. "
            "Verify that the NTRU key name matches and the bundle is unmodified."
        ) from exc
    return aes_key, xor_key_bytes

# ─────────────────────────────────────────────────────────────────────────────
# Master encryption pipeline
# ─────────────────────────────────────────────────────────────────────────────

def master_encrypt(
    plaintext,
    cover_image_path:    str  = "lena2.png",
    output_path:         str  = "encoded_image.png",
    output_qr_code_path: str  = "qr_code.png",
    key_bundle_path:     str  = "key_bundle.bin",
    display:             bool = False,
    cloud_upload:        bool = False,
) -> tuple:
    """
    Full hybrid encryption-steganography pipeline (sender side).

    Executes the following sequence:
        1. NTRU post-quantum encryption of plaintext.
        2. AES-256-GCM encryption using a GAN-derived session key and IV.
        3. XOR obfuscation using a GAN-derived session mask.
        4. IWT 3-channel steganographic embedding of the ciphertext into
           the cover image. Session keys are NOT embedded in the image.
        5. NTRU encapsulation of both session keys into a separate key bundle.
        6. Optional cloud upload and QR code generation for the stego image.

    Parameters
    ----------
    plaintext : str
        The secret message to encrypt and embed.
    cover_image_path : str
        Path to the cover image. Resized to 4096×4096 before embedding.
    output_path : str
        Output path for the stego image (PNG).
    output_qr_code_path : str
        Output path for the QR code image (used only if cloud_upload=True).
    key_bundle_path : str
        Output path for the NTRU-encrypted key bundle. This file must be
        transmitted to the receiver via a channel separate from the stego image.
        Default: "key_bundle.bin".
    display : bool
        If True and cloud_upload=False, displays cover and stego images inline.
    cloud_upload : bool
        If True, uploads the stego image to GCS and generates a QR code.
        If False, saves the stego image locally only.

    Returns
    -------
    encoded_img : np.ndarray
        The stego image as a BGR uint8 array.
    qr_code_img : np.ndarray or None
        The QR code image if cloud_upload=True, otherwise None.
    key_bundle_bytes : bytes
        The NTRU-encrypted key bundle. Also written to key_bundle_path.

    Raises
    ------
    FileNotFoundError
        If cover_image_path cannot be read by OpenCV.
    AssertionError
        If the plaintext exceeds the embedding capacity of the cover image.

    Example
    -------
    >>> encoded_img, qr_img, key_bundle = master_encrypt(
    ...     "Secret message",
    ...     cover_image_path="cover.png",
    ...     output_path="stego.png",
    ...     key_bundle_path="keys.bin",
    ...     display=True,
    ...     cloud_upload=False,
    ... )
    # Transmit stego.png and keys.bin via separate channels.
    """
    # ── Stage 1: NTRU post-quantum encryption ────────────────────────────────
    ntru_encrypted = pq_ntru.encrypt(NTRU_KEY_NAME, plaintext)

    # ── Stage 2: AES-256-GCM encryption ──────────────────────────────────────
    # aes_key is returned separately — not embedded in ciphertext_string
    ciphertext_string, aes_key = encrypt_message(ntru_encrypted)

    # ── Stage 3: XOR obfuscation ──────────────────────────────────────────────
    # xor_key_bytes is returned separately — not prepended to output
    xor_encrypted_string, xor_key_bytes = xor_encrypt(ciphertext_string)

    print(f"Payload length (ciphertext only, no keys): {len(xor_encrypted_string)} chars")

    # ── Stage 4: IWT 3-channel steganographic embedding ──────────────────────
    img = cv2.imread(cover_image_path)
    if img is None:
        raise FileNotFoundError(f"Could not read cover image: {cover_image_path}")
    img = cv2.resize(img, (4096, 4096))

    iwt        = IWT_FAST.IWT()
    encoded_img = iwt.encode_image(img, xor_encrypted_string, output_path=output_path)

    if encoded_img is False:
        raise AssertionError(
            "Plaintext exceeds steganographic capacity of the cover image. "
            "Use a larger image or reduce plaintext length."
        )

    cv2.imwrite(output_path, encoded_img)

    # ── Stage 5: NTRU key encapsulation (key channel) ─────────────────────────
    # Both session keys are encrypted into a single key bundle using NTRU.
    # This bundle is written to a separate file and must be transmitted
    # independently of the stego image.
    key_bundle_bytes = encapsulate_session_keys(aes_key, xor_key_bytes)

    with open(key_bundle_path, "wb") as f:
        f.write(key_bundle_bytes)

    print(f"Stego image written to   : {output_path}")
    print(f"Key bundle written to    : {key_bundle_path}")
    print(f"Key bundle size          : {len(key_bundle_bytes)} bytes")
    print(f"Transmit these two files via SEPARATE channels.")

    # ── Stage 6: Optional cloud upload and QR generation ─────────────────────
    qr_code_img = None
    if cloud_upload:
        blob_name  = output_path + datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        public_url = cloud_handler.upload_file(output_path, blob_name)
        cloud_handler.generate_qr_code(public_url, output_qr_code_path)
        qr_code_img = cv2.imread(output_qr_code_path)
    else:
        if display:
            encoded_img_rgb = cv2.cvtColor(encoded_img, cv2.COLOR_BGR2RGB)
            img_rgb         = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

            plt.figure(figsize=(10, 5))
            plt.subplot(1, 2, 1)
            plt.imshow(encoded_img_rgb)
            plt.title("Encoded Image (stego)")
            plt.axis("off")

            plt.subplot(1, 2, 2)
            plt.imshow(img_rgb)
            plt.title("Cover Image (original)")
            plt.axis("off")

            plt.tight_layout()
            plt.show()

    return encoded_img, qr_code_img, key_bundle_bytes

# ─────────────────────────────────────────────────────────────────────────────
# Master decryption pipeline
# ─────────────────────────────────────────────────────────────────────────────

def master_decrypt(
    input_data:            str  = None,
    key_bundle_path:       str  = "key_bundle.bin",
    downloaded_file_path:  str  = "downloaded_file.png",
    qr:                    bool = True,
) -> str:
    """
    Full hybrid decryption pipeline (receiver side).

    Executes the following sequence:
        1. Retrieves the stego image (via QR/cloud or direct file path).
        2. Loads and NTRU-decapsulates the key bundle to recover AES key
           and XOR mask.
        3. Extracts the XOR-encrypted ciphertext from the stego image using
           IWT 3-channel decoding.
        4. XOR-decrypts to recover the AES-GCM ciphertext.
        5. AES-GCM-decrypts to recover the NTRU ciphertext.
        6. NTRU-decrypts to recover the original plaintext.

    Parameters
    ----------
    input_data : str
        If qr=True: path to the QR code image.
        If qr=False: path to the stego image directly.
    key_bundle_path : str
        Path to the NTRU-encrypted key bundle file produced by master_encrypt().
        Must be received via the key channel (separate from the stego image).
        Default: "key_bundle.bin".
    downloaded_file_path : str
        Local path where the stego image is saved if downloaded from cloud.
        Used only when qr=True.
    qr : bool
        If True, decodes a QR code to retrieve the stego image URL and
        downloads it from cloud storage.
        If False, reads the stego image directly from input_data.

    Returns
    -------
    str
        The recovered plaintext message.

    Raises
    ------
    FileNotFoundError
        If the key bundle or stego image cannot be found.
    ValueError
        If key bundle decapsulation fails or GCM authentication fails.

    Example
    -------
    >>> message = master_decrypt(
    ...     input_data="stego.png",
    ...     key_bundle_path="keys.bin",
    ...     qr=False,
    ... )
    >>> print(message)
    """
    # ── Stage 1: Retrieve stego image ─────────────────────────────────────────
    if qr:
        decoded_url = cloud_handler.read_qr_code(input_data)
        cloud_handler.download_file(decoded_url, downloaded_file_path)
        img = cv2.imread(downloaded_file_path)
    else:
        img = cv2.imread(input_data)

    if img is None:
        raise FileNotFoundError(
            f"Could not read stego image from: "
            f"{downloaded_file_path if qr else input_data}"
        )

    # ── Stage 2: Load and decapsulate key bundle (key channel) ────────────────
    if not os.path.exists(key_bundle_path):
        raise FileNotFoundError(
            f"Key bundle not found at: {key_bundle_path}. "
            "Ensure the key bundle was received via the separate key channel."
        )

    with open(key_bundle_path, "rb") as f:
        key_bundle_bytes = f.read()

    aes_key, xor_key_bytes = decapsulate_session_keys(key_bundle_bytes)

    # ── Stage 3: Extract XOR-encrypted ciphertext from stego image ────────────
    iwt                  = IWT_FAST.IWT()
    xor_encrypted_string = iwt.decode_image(img)

    # ── Stage 4: XOR decryption ───────────────────────────────────────────────
    ciphertext_string = xor_decrypt(xor_encrypted_string, xor_key_bytes)

    # ── Stage 5: AES-256-GCM decryption ──────────────────────────────────────
    ntru_encrypted_bytes = decrypt_message(ciphertext_string, aes_key)

    # ── Stage 6: NTRU post-quantum decryption ─────────────────────────────────
    original_message = pq_ntru.decrypt(NTRU_KEY_NAME, ntru_encrypted_bytes)

    return original_message