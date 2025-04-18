#!/usr/bin/env python3
"""
This module defines:
  - The IWT class that implements an Integer Wavelet Transform (IWT) based steganography technique.
  - Methods to encode secret messages into an image and decode them back.
  - Helper functions to perform image padding, block reconstruction, and bit-level conversions.
  - Numba-accelerated IWT routines for performance.
"""

import cv2
import numpy as np
import itertools
import random
from numba import njit

# Numba-accelerated IWT forward (haar lifting) for 8×8 blocks
@njit (cache=True)
def haar_lifting_iwt2_numba(block):
    temp = np.zeros_like(block)
    # row-wise lifting
    for i in range(8):
        for j in range(0, 8, 2):
            even = block[i, j]
            odd = block[i, j + 1]
            temp[i, j // 2]       = (even + odd) // 2
            temp[i, j // 2 + 4]   = odd - even
    out = np.zeros_like(temp)
    # column-wise lifting
    for j in range(8):
        for i in range(0, 8, 2):
            even = temp[i, j]
            odd  = temp[i + 1, j]
            out[i // 2, j]       = (even + odd) // 2
            out[i // 2 + 4, j]   = odd - even
    return out

# Numba-accelerated inverse IWT for 8×8 blocks
@njit (cache=True)
def haar_lifting_iiwt2_numba(coeffs):
    temp = np.zeros_like(coeffs)
    # column-wise inverse lifting
    for j in range(8):
        for i in range(4):
            avg  = coeffs[i, j]
            diff = coeffs[i + 4, j]
            even = avg - diff // 2
            odd  = diff + even
            temp[2 * i, j]     = even
            temp[2 * i + 1, j] = odd
    out = np.zeros_like(temp)
    # row-wise inverse lifting
    for i in range(8):
        for j in range(4):
            avg  = temp[i, j]
            diff = temp[i, j + 4]
            even = avg - diff // 2
            odd  = diff + even
            out[i, 2 * j]     = even
            out[i, 2 * j + 1] = odd
    return out

# Warm up compilation on a dummy block so first real call is faster
_ = haar_lifting_iwt2_numba(np.zeros((8, 8), dtype=np.int32))
_ = haar_lifting_iiwt2_numba(np.zeros((8, 8), dtype=np.int32))


class IWT:
    def __init__(self):
        self.bitMess = None

    # Override to call Numba-accelerated forward IWT
    def haar_lifting_iwt2(self, block):
        return haar_lifting_iwt2_numba(block)

    # Override to call Numba-accelerated inverse IWT
    def haar_lifting_iiwt2(self, coeffs):
        return haar_lifting_iiwt2_numba(coeffs)

    def encode_image(self, img, secret_msg):
        secret = str(len(secret_msg)) + '*' + secret_msg
        self.bitMess = self.toBits(secret)
        row, col = img.shape[:2]

        if (col // 8) * (row // 8) * 16 < len(secret_msg) * 8:
            print("Error: Message too large to encode in image")
            return False

        if row % 8 != 0 or col % 8 != 0:
            img = self.addPadd(img, row, col)

        bImg, gImg, rImg = cv2.split(img)
        bImg = np.int32(bImg)

        imgBlocks = [
            bImg[j:j+8, i:i+8]
            for (j, i) in itertools.product(range(0, row, 8), range(0, col, 8))
        ]

        messIndex = 0
        letterIndex = 0

        for idx in range(len(imgBlocks)):
            if messIndex >= len(self.bitMess):
                break

            block = imgBlocks[idx]
            iwt_block = self.haar_lifting_iwt2(block)

            subbands = [
                iwt_block[4:8, 4:8],
                iwt_block[0:4, 0:4],
                iwt_block[0:4, 4:8],
                iwt_block[4:8, 0:4],
            ]

            for band in subbands:
                flat = band.flatten()
                for i in range(len(flat)):
                    bit = self.bitMess[messIndex][letterIndex]
                    if (flat[i] & 1) != bit:
                        if flat[i] == 0:
                            flat[i] += 1
                        elif flat[i] == 255:
                            flat[i] -= 1
                        else:
                            flat[i] += random.choice([-1, 1])
                    letterIndex += 1
                    if letterIndex == 8:
                        letterIndex = 0
                        messIndex += 1
                        if messIndex == len(self.bitMess):
                            break
                band[:] = flat.reshape(band.shape)
                if messIndex == len(self.bitMess):
                    break

            imgBlocks[idx] = self.haar_lifting_iiwt2(iwt_block)

        sImg = self.reconstruct_image(imgBlocks, row, col)
        sImg = cv2.merge((sImg.astype(np.uint8), gImg, rImg))
        cv2.imwrite('/teamspace/studios/this_studio/IWT_L1.png', sImg)
        return sImg

    def decode_image(self, img):
        row, col = img.shape[:2]
        bImg, _, _ = cv2.split(img)
        bImg = np.int32(bImg)

        imgBlocks = [
            bImg[j:j+8, i:i+8]
            for (j, i) in itertools.product(range(0, row, 8), range(0, col, 8))
        ]

        messageBits = []
        buff = 0
        i = 0
        messSize = None

        for block in imgBlocks:
            iwt_block = self.haar_lifting_iwt2(block)

            subbands = [
                iwt_block[4:8, 4:8],
                iwt_block[0:4, 0:4],
                iwt_block[0:4, 4:8],
                iwt_block[4:8, 0:4],
            ]

            for band in subbands:
                flat = band.flatten()
                for val in flat:
                    bit = val & 1
                    buff = (buff << 1) | bit
                    i += 1
                    if i == 8:
                        char = chr(buff)
                        messageBits.append(char)
                        buff = 0
                        i = 0
                        if char == '*' and messSize is None:
                            try:
                                messSize = int(''.join(messageBits[:-1]))
                            except ValueError:
                                pass
                        if messSize and len(messageBits) - len(str(messSize)) - 1 == messSize:
                            return ''.join(messageBits)[len(str(messSize)) + 1:]
        return ""

    def reconstruct_image(self, blocks, row, col):
        sImg = []
        for chunkRowBlocks in self.chunks(blocks, col // 8):
            for rowBlockNum in range(8):
                for block in chunkRowBlocks:
                    sImg.extend(block[rowBlockNum])
        return np.array(sImg).reshape(row, col)

    def addPadd(self, img, row, col):
        return cv2.resize(img, (col + (8 - col % 8), row + (8 - row % 8)))

    def toBits(self, message):
        return [list(map(int, bin(ord(char))[2:].rjust(8, '0'))) for char in message]

    def chunks(self, l, n):
        for i in range(0, len(l), int(n)):
            yield l[i:i + int(n)]


if __name__ == "__main__":
    img_path = "input_image.png"  # Replace with your image file path
    img = cv2.imread(img_path)
    if img is None:
        print("Error: Image not found.")
    else:
        secret_message = "Hello, world!"
        iwt_obj = IWT()
        encoded_img = iwt_obj.encode_image(img, secret_message)
        print("Image encoded successfully.")

        decoded_message = iwt_obj.decode_image(encoded_img)
        print("Decoded message:", decoded_message)
