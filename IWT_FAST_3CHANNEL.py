#!/usr/bin/env python3
"""
IWT_FAST.py — Integer Wavelet Transform steganography with full 3-channel embedding.

Design contract between encoder and decoder:
  Both traverse IWT coefficients in an identical deterministic order:
    Blue channel blocks → Green channel blocks → Red channel blocks
  Within each channel, blocks are ordered row-major (top-left to bottom-right).
  Within each block, subbands are visited: HH → LL → LH → HL.
  Within each subband, coefficients are visited in row-major order.

  This single shared traversal order is the only thing that guarantees
  correct extraction. As long as both sides use it, the bit stream is
  unambiguous regardless of which channel or block a given bit lands in.

  The length prefix ("NNNNN*") is written into the unified stream starting
  at position 0. The decoder reads it as characters from the unified stream
  before knowing how long the actual message is — so it works even if the
  prefix spans a channel boundary.

Capacity:
  For a 512×512 image: 64×64 = 4096 blocks per channel × 3 channels = 12288 blocks.
  Each block has 4 subbands × 16 coefficients = 64 bits = 8 bytes per block.
  Total capacity: 12288 × 8 = 98,304 bytes (raw), minus the length prefix overhead.
"""

import cv2
import numpy as np
import itertools
import random
from numba import njit


# ─────────────────────────────────────────────────────────────────────────────
# Numba-accelerated IWT forward and inverse — unchanged from original
# ─────────────────────────────────────────────────────────────────────────────

@njit(cache=True)
def haar_lifting_iwt2_numba(block):
    """Forward 2D Integer Haar Wavelet Transform on an 8×8 block."""
    temp = np.zeros_like(block)
    # Row-wise lifting
    for i in range(8):
        for j in range(0, 8, 2):
            even = block[i, j]
            odd  = block[i, j + 1]
            temp[i, j // 2]     = (even + odd) // 2
            temp[i, j // 2 + 4] = odd - even
    out = np.zeros_like(temp)
    # Column-wise lifting
    for j in range(8):
        for i in range(0, 8, 2):
            even = temp[i, j]
            odd  = temp[i + 1, j]
            out[i // 2, j]     = (even + odd) // 2
            out[i // 2 + 4, j] = odd - even
    return out


@njit(cache=True)
def haar_lifting_iiwt2_numba(coeffs):
    """Inverse 2D Integer Haar Wavelet Transform on an 8×8 block."""
    temp = np.zeros_like(coeffs)
    # Column-wise inverse lifting
    for j in range(8):
        for i in range(4):
            avg  = coeffs[i, j]
            diff = coeffs[i + 4, j]
            even = avg - diff // 2
            odd  = diff + even
            temp[2 * i, j]     = even
            temp[2 * i + 1, j] = odd
    out = np.zeros_like(temp)
    # Row-wise inverse lifting
    for i in range(8):
        for j in range(4):
            avg  = temp[i, j]
            diff = temp[i, j + 4]
            even = avg - diff // 2
            odd  = diff + even
            out[i, 2 * j]     = even
            out[i, 2 * j + 1] = odd
    return out


# Warm up Numba JIT compilation on dummy blocks so first real call is instant
_ = haar_lifting_iwt2_numba(np.zeros((8, 8), dtype=np.int32))
_ = haar_lifting_iiwt2_numba(np.zeros((8, 8), dtype=np.int32))


# ─────────────────────────────────────────────────────────────────────────────
# IWT class — full 3-channel encoder / decoder
# ─────────────────────────────────────────────────────────────────────────────

class IWT:
    """
    Integer Wavelet Transform steganography — 3-channel (BGR) embedding.

    Public API:
        encode_image(img, secret_msg)  →  stego_img (np.ndarray, BGR uint8)
        decode_image(img)              →  secret_msg (str)
        capacity_bytes(img)            →  int  (max embeddable bytes)
        capacity_report(img)           →  prints per-channel and total capacity
    """

    # Subband visit order within each IWT block.
    # HH first (highest frequency, least perceptually significant),
    # then LL, LH, HL. Matches original code's order.
    SUBBAND_SLICES = [
        (slice(4, 8), slice(4, 8)),  # HH
        (slice(0, 4), slice(0, 4)),  # LL
        (slice(0, 4), slice(4, 8)),  # LH
        (slice(4, 8), slice(0, 4)),  # HL
    ]

    def __init__(self):
        self.bitMess = None

    # ── Wavelet wrappers ──────────────────────────────────────────────────────

    def haar_lifting_iwt2(self, block: np.ndarray) -> np.ndarray:
        return haar_lifting_iwt2_numba(block)

    def haar_lifting_iiwt2(self, coeffs: np.ndarray) -> np.ndarray:
        return haar_lifting_iiwt2_numba(coeffs)

    # ── Capacity helpers ──────────────────────────────────────────────────────

    def _block_count(self, img: np.ndarray) -> int:
        """Number of 8×8 blocks in one channel of the image."""
        row, col = img.shape[:2]
        return (row // 8) * (col // 8)

    def capacity_bytes(self, img: np.ndarray) -> int:
        """
        Maximum bytes embeddable across all 3 channels.
        Each block holds 4 subbands × 16 coefficients = 64 bits = 8 bytes.
        Subtract 16 bytes conservatively for the length prefix overhead.
        """
        blocks_per_channel = self._block_count(img)
        total_bits = blocks_per_channel * 3 * 64   # 3 channels × 64 bits/block
        return max(0, total_bits // 8 - 16)

    def capacity_report(self, img: np.ndarray) -> None:
        """Prints a breakdown of embedding capacity per channel and in total."""
        row, col = img.shape[:2]
        blocks = self._block_count(img)
        bits_per_ch = blocks * 64
        print(f"Image size:           {col}×{row} pixels")
        print(f"Blocks per channel:   {blocks}  ({col//8}×{row//8})")
        print(f"Bits per channel:     {bits_per_ch:,}  ({bits_per_ch//8:,} bytes)")
        print(f"Total (3 channels):   {bits_per_ch*3:,} bits  ({bits_per_ch*3//8:,} bytes)")
        print(f"Usable capacity:      {self.capacity_bytes(img):,} bytes  (prefix overhead deducted)")

    # ── Internal helpers ──────────────────────────────────────────────────────

    def _pad_image(self, img: np.ndarray) -> np.ndarray:
        """Pad image dimensions to the nearest multiple of 8."""
        row, col = img.shape[:2]
        new_col = col if col % 8 == 0 else col + (8 - col % 8)
        new_row = row if row % 8 == 0 else row + (8 - row % 8)
        return cv2.resize(img, (new_col, new_row))

    def _split_into_blocks(self, channel: np.ndarray) -> list:
        """
        Split a 2D channel array into a list of 8×8 blocks in row-major order.
        Returns a list of views; modifications to returned arrays affect originals.
        This is intentional for the encoder — we modify blocks in place then reconstruct.
        """
        row, col = channel.shape
        return [
            channel[j:j+8, i:i+8]
            for (j, i) in itertools.product(range(0, row, 8), range(0, col, 8))
        ]

    def _reconstruct_channel(self, blocks: list, row: int, col: int) -> np.ndarray:
        """Reconstruct a 2D channel from a list of 8×8 blocks (row-major order)."""
        out = np.zeros((row, col), dtype=np.int32)
        idx = 0
        for j in range(0, row, 8):
            for i in range(0, col, 8):
                out[j:j+8, i:i+8] = blocks[idx]
                idx += 1
        return out

    def _to_bits(self, message: str) -> list:
        """
        Convert a string to a flat list of bits.
        Each character → 8 bits, MSB first.
        Returns a list of ints (0 or 1) of length len(message) × 8.
        """
        bits = []
        for char in message:
            byte = ord(char)
            for shift in range(7, -1, -1):
                bits.append((byte >> shift) & 1)
        return bits

    def _embed_bit_into_coefficient(self, value: int, bit: int) -> int:
        """
        Embed a single bit into the LSB of an IWT coefficient.
        Handles boundary values (0 and 255) by nudging away from the boundary
        to avoid wrap-around corruption. For all other values, perturbs by ±1
        chosen randomly to avoid systematic bias.
        """
        if (value & 1) == bit:
            return value  # LSB already correct, no change needed
        if value == 0:
            return 1
        if value == 255:
            return 254
        return value + random.choice([-1, 1])

    # ── Unified coefficient iterator ──────────────────────────────────────────

    def _iter_coefficients(self, iwt_blocks_bgr: list) -> "generator":
        """
        Yields (block_idx, channel_idx, subband_row, subband_col, flat_idx)
        tuples in the canonical order:
            Blue blocks (all) → Green blocks (all) → Red blocks (all)
            Within each block: HH → LL → LH → HL subbands
            Within each subband: row-major coefficient order

        This generator defines the shared traversal contract.
        Both encode and decode call this with the same block structure
        to guarantee identical traversal order.

        iwt_blocks_bgr is a list of 3 lists: [b_blocks, g_blocks, r_blocks]
        where each inner list contains IWT-transformed 8×8 arrays.
        """
        for ch_idx, ch_blocks in enumerate(iwt_blocks_bgr):
            for blk_idx, block in enumerate(ch_blocks):
                for (row_slice, col_slice) in self.SUBBAND_SLICES:
                    subband = block[row_slice, col_slice]
                    flat = subband.flatten()
                    for coeff_idx in range(len(flat)):
                        yield ch_idx, blk_idx, row_slice, col_slice, coeff_idx, flat

    # ── Public encoder ────────────────────────────────────────────────────────

    def encode_image(
        self,
        img: np.ndarray,
        secret_msg: str,
        output_path: str = None,
    ) -> np.ndarray:
        """
        Embed secret_msg into img across all 3 BGR channels using IWT LSB embedding.

        Parameters
        ----------
        img         : BGR image as np.ndarray (uint8). Grayscale converted to BGR if needed.
        secret_msg  : The string to embed.
        output_path : If provided, saves the stego image to this path.
                      Defaults to 'IWT_3CH.png' in the current directory.

        Returns
        -------
        Stego image as np.ndarray (BGR uint8), or False if message too large.
        """
        if output_path is None:
            output_path = "IWT_3CH.png"

        # Handle grayscale input
        if len(img.shape) == 2:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

        # Pad to 8×8 block boundary if needed
        row, col = img.shape[:2]
        if row % 8 != 0 or col % 8 != 0:
            img = self._pad_image(img)
            row, col = img.shape[:2]

        # Check capacity
        max_bytes = self.capacity_bytes(img)
        if len(secret_msg) > max_bytes:
            print(f"Error: Message ({len(secret_msg)} bytes) exceeds capacity ({max_bytes} bytes)")
            return False

        # Prepend length prefix so decoder knows when to stop
        full_message = str(len(secret_msg)) + '*' + secret_msg
        bit_stream = self._to_bits(full_message)
        total_bits = len(bit_stream)

        # Split image into BGR channels and convert to int32 for IWT arithmetic
        b_ch, g_ch, r_ch = cv2.split(img)
        b_ch = np.int32(b_ch)
        g_ch = np.int32(g_ch)
        r_ch = np.int32(r_ch)

        # Build 8×8 block lists for each channel
        b_blocks = self._split_into_blocks(b_ch)
        g_blocks = self._split_into_blocks(g_ch)
        r_blocks = self._split_into_blocks(r_ch)

        # Forward IWT on every block of every channel
        # Store transformed blocks separately so we can modify then invert
        b_iwt = [self.haar_lifting_iwt2(blk.copy()) for blk in b_blocks]
        g_iwt = [self.haar_lifting_iwt2(blk.copy()) for blk in g_blocks]
        r_iwt = [self.haar_lifting_iwt2(blk.copy()) for blk in r_blocks]

        # ── Embed bit stream into IWT coefficients ────────────────────────────
        bit_idx = 0

        for ch_idx, blk_idx, row_sl, col_sl, coeff_idx, _ in self._iter_coefficients(
            [b_iwt, g_iwt, r_iwt]
        ):
            if bit_idx >= total_bits:
                break

            # Select the correct channel's block
            if ch_idx == 0:
                block = b_iwt[blk_idx]
            elif ch_idx == 1:
                block = g_iwt[blk_idx]
            else:
                block = r_iwt[blk_idx]

            subband = block[row_sl, col_sl]
            flat = subband.flatten()

            bit = bit_stream[bit_idx]
            flat[coeff_idx] = self._embed_bit_into_coefficient(int(flat[coeff_idx]), bit)

            # Write modified flat back into subband in-place
            subband[:] = flat.reshape(subband.shape)
            bit_idx += 1

        # ── Inverse IWT on all modified blocks ────────────────────────────────
        b_reconstructed = [self.haar_lifting_iiwt2(blk) for blk in b_iwt]
        g_reconstructed = [self.haar_lifting_iiwt2(blk) for blk in g_iwt]
        r_reconstructed = [self.haar_lifting_iiwt2(blk) for blk in r_iwt]

        # Reconstruct full channel arrays
        b_out = self._reconstruct_channel(b_reconstructed, row, col).astype(np.uint8)
        g_out = self._reconstruct_channel(g_reconstructed, row, col).astype(np.uint8)
        r_out = self._reconstruct_channel(r_reconstructed, row, col).astype(np.uint8)

        stego_img = cv2.merge((b_out, g_out, r_out))
        cv2.imwrite(output_path, stego_img)
        return stego_img

    # ── Public decoder ────────────────────────────────────────────────────────

    def decode_image(self, img: np.ndarray) -> str:
        """
        Extract the hidden message from a stego image produced by encode_image.

        Traverses all 3 BGR channels in the same canonical order as the encoder.
        Reads the length prefix first, then stops exactly at message end.

        Returns the extracted message string, or empty string if nothing found.
        """
        # Handle grayscale input
        if len(img.shape) == 2:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

        row, col = img.shape[:2]

        # Split and convert channels
        b_ch, g_ch, r_ch = cv2.split(img)
        b_ch = np.int32(b_ch)
        g_ch = np.int32(g_ch)
        r_ch = np.int32(r_ch)

        # Build block lists
        b_blocks = self._split_into_blocks(b_ch)
        g_blocks = self._split_into_blocks(g_ch)
        r_blocks = self._split_into_blocks(r_ch)

        # Forward IWT on all blocks
        b_iwt = [self.haar_lifting_iwt2(blk.copy()) for blk in b_blocks]
        g_iwt = [self.haar_lifting_iwt2(blk.copy()) for blk in g_blocks]
        r_iwt = [self.haar_lifting_iwt2(blk.copy()) for blk in r_blocks]

        # ── Read bit stream from coefficients ─────────────────────────────────
        # We accumulate characters one byte (8 bits) at a time.
        # The length prefix "NNNNN*" tells us when to stop.

        char_buffer  = 0      # accumulates bits into the current character
        bit_count    = 0      # bits accumulated in current character (0-7)
        chars_so_far = []     # decoded characters so far
        message_size = None   # set once we parse the prefix

        for ch_idx, blk_idx, row_sl, col_sl, coeff_idx, _ in self._iter_coefficients(
            [b_iwt, g_iwt, r_iwt]
        ):
            # Select the correct channel's block
            if ch_idx == 0:
                block = b_iwt[blk_idx]
            elif ch_idx == 1:
                block = g_iwt[blk_idx]
            else:
                block = r_iwt[blk_idx]

            subband = block[row_sl, col_sl]
            flat = subband.flatten()
            bit = int(flat[coeff_idx]) & 1

            # Accumulate bit into current byte (MSB first)
            char_buffer = (char_buffer << 1) | bit
            bit_count += 1

            if bit_count == 8:
                # Completed a character
                char = chr(char_buffer)
                chars_so_far.append(char)
                char_buffer = 0
                bit_count   = 0

                # Try to parse the length prefix as soon as we see '*'
                if char == '*' and message_size is None:
                    prefix = ''.join(chars_so_far[:-1])  # everything before '*'
                    try:
                        message_size = int(prefix)
                    except ValueError:
                        # '*' was part of the message content, not the delimiter
                        pass

                # Check if we've read enough characters for the full message
                if message_size is not None:
                    prefix_len = len(str(message_size)) + 1  # digits + '*'
                    if len(chars_so_far) >= prefix_len + message_size:
                        return ''.join(chars_so_far[prefix_len:prefix_len + message_size])

        # Reached end of all coefficients without finding a complete message
        if message_size is not None and chars_so_far:
            prefix_len = len(str(message_size)) + 1
            extracted = ''.join(chars_so_far[prefix_len:])
            if extracted:
                return extracted

        return ""

    # ── Legacy helpers (kept for compatibility) ───────────────────────────────

    def addPadd(self, img, row, col):
        return cv2.resize(img, (col + (8 - col % 8), row + (8 - row % 8)))

    def toBits(self, message):
        """Original per-character bit list format — kept for compatibility."""
        return [list(map(int, bin(ord(char))[2:].rjust(8, '0'))) for char in message]

    def chunks(self, l, n):
        for i in range(0, len(l), int(n)):
            yield l[i:i + int(n)]


# ─────────────────────────────────────────────────────────────────────────────
# Self-test — run directly to verify encode/decode round-trip across channels
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys

    print("=== IWT_FAST 3-Channel Round-Trip Test ===\n")

    # Create a synthetic test image if no path is provided
    if len(sys.argv) > 1:
        img = cv2.imread(sys.argv[1])
        if img is None:
            print(f"Could not load {sys.argv[1]}")
            sys.exit(1)
        print(f"Loaded: {sys.argv[1]}  shape={img.shape}")
    else:
        # 512×512 random BGR image
        img = np.random.randint(0, 256, (512, 512, 3), dtype=np.uint8)
        print("Using synthetic 512×512 random image")

    iwt = IWT()
    iwt.capacity_report(img)
    print()

    # Test 1: short message
    tests = [
        "Hello, world!",
        "A" * 1000,
        "".join(chr(random.randint(32, 126)) for _ in range(5000)),
    ]

    for i, msg in enumerate(tests):
        print(f"Test {i+1}: {len(msg)} characters")
        stego = iwt.encode_image(img.copy(), msg, output_path=f"test_stego_{i}.png")
        if stego is False:
            print(f"  SKIP — message too large for this image")
            continue
        recovered = iwt.decode_image(stego)
        if recovered == msg:
            print(f"  PASS — round-trip correct")
        else:
            print(f"  FAIL — mismatch")
            print(f"    Expected (first 50): {repr(msg[:50])}")
            print(f"    Got      (first 50): {repr(recovered[:50])}")

    print("\nDone.")