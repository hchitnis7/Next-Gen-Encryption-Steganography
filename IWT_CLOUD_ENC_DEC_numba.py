#!/usr/bin/env python3
"""
This module defines:
  - The IWT class that implements an Integer Wavelet Transform (IWT) based steganography technique.
  - Methods to encode secret messages into images by partitioning the image into 8x8 blocks, applying the IWT,
    modifying selected subband coefficients to embed secret bits, and reconstructing the image.
  - Methods to decode hidden messages from stego-images by reversing the IWT process and extracting the embedded bits.
  - Helper functions to perform image padding, block reconstruction, and bit-level conversions to support the encoding and decoding processes.
  - Integration with various libraries (e.g., OpenCV, NumPy, and Crypto modules) for image processing, encryption, and randomness,
    ensuring robust and secure steganography operations.
"""


import cv2
import numpy as np
from Crypto.Cipher import AES
import pq_ntru
import cv2
import base64
import time
import matplotlib.pyplot as plt
import CHAOSDCGAN as cdc
import torch
import secrets
import IWT_FAST 
import base64
from Crypto.Cipher import AES
import time
import cv2
from datetime import datetime 
import CloudQRHandler as cloudthing

credentials_path = "/teamspace/studios/this_studio/project_dir/logical-factor-455520-m8-2405e5b6a50c.json"
bucket_name = "dctimages"
# Create a CloudQRHandler object
cloud_handler = cloudthing.CloudQRHandler(credentials_path, bucket_name)

# ---------------------------
# Global Configuration
# ---------------------------
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
LATENT_DIM = 1024
BATCH_SIZE = 32
MODEL_PATH_G = "/teamspace/studios/this_studio/expanded_generator_dcgan_9CHAN_GPT.pth"
# Tolerance for smart randomisation (mode frequency must be within ±20% of average)
TOLERANCE = 0.45




gan_bytes_all = cdc.inference()




import secrets
import base64
from Crypto.Cipher import AES
import numpy as np
from numba import njit

# AES Encryption (returns a single string)
def encrypt_message(plaintext, aes_key=None):
    if aes_key is None:
        assert len(gan_bytes_all) >= 32, "Not enough bytes to select from."
        # Securely select 32 random indices from gan_bytes_all using a CSPRNG
        random_indices = secrets.SystemRandom().sample(range(len(gan_bytes_all)), 32)
        # Extract the randomly chosen bytes and convert to a bytes object using list comprehension
        # Explicitly build a list to guarantee 32 bytes
        key_list = []
        for i in random_indices:
            key_list.append(gan_bytes_all[i])
        aes_key = bytes(key_list)        
    # Securely select 12 random bytes for the IV (recommended length for AES-GCM)
    iv_random_indices = secrets.SystemRandom().sample(range(len(gan_bytes_all)), 12)
    iv_list = []
    for i in iv_random_indices:
        iv_list.append(gan_bytes_all[i])
    iv = bytes(iv_list)
    # AES-GCM recommended IV length
    cipher = AES.new(aes_key, AES.MODE_GCM, nonce=iv)
    ciphertext, tag = cipher.encrypt_and_digest(plaintext.encode('utf-8'))

    # Convert everything to a single string
    encrypted_string = (
        aes_key.hex() + ":" + iv.hex() + ":" +
        ciphertext.hex() + ":" + tag.hex()
    )

    return encrypted_string

# AES Decryption (parses the string back)
def decrypt_message(encrypted_string):
    key_hex, iv_hex, ciphertext_hex, tag_hex = encrypted_string.split(":")

    key = bytes.fromhex(key_hex)
    iv = bytes.fromhex(iv_hex)
    ciphertext = bytes.fromhex(ciphertext_hex)
    tag = bytes.fromhex(tag_hex)

    cipher = AES.new(key, AES.MODE_GCM, nonce=iv)
    plaintext = cipher.decrypt_and_verify(ciphertext, tag)

    return plaintext.decode('utf-8')

# Numba-accelerated XOR core
@njit
def xor_cipher_numba(data_arr, key_arr):
    result = np.empty_like(data_arr)
    key_len = key_arr.shape[0]
    for i in range(data_arr.shape[0]):
        result[i] = data_arr[i] ^ key_arr[i % key_len]
    return result

# Wrapper to keep original API unchanged
def xor_cipher(data, key):
    data_bytes = np.frombuffer(data.encode('utf-8'), dtype=np.uint8)
    key_bytes = np.frombuffer(key.encode('utf-8'), dtype=np.uint8)
    result_bytes = xor_cipher_numba(data_bytes, key_bytes)
    # Decode ignoring any partial-byte issues
    return result_bytes.tobytes().decode('utf-8', errors='ignore')

def xor_encrypt(aes_encrypted_string):
    # Generate XOR key: randomly select 32 bytes from gan_bytes_all using a CSPRNG
    xor_random_indices = secrets.SystemRandom().sample(range(len(gan_bytes_all)), 12)
    # Explicitly build a list to guarantee 32 bytes
    xor_key_list = []
    for i in xor_random_indices:
        xor_key_list.append(gan_bytes_all[i])
    xor_key_bytes = bytes(xor_key_list)
    # Base64-encode the XOR key for safe transmission/storage
    xor_key = base64.urlsafe_b64encode(xor_key_bytes).decode('utf-8')
    
    xor_encrypted_string = xor_cipher(aes_encrypted_string, xor_key)
    
    # Combine the Base64-encoded key and the XOR-encrypted string
    final_encrypted_string = xor_key + ":" + xor_encrypted_string
    return final_encrypted_string

def xor_decrypt(final_encrypted_string):
    xor_key, xor_encrypted_string = final_encrypted_string.split(":", 1)
    decrypted_aes_string = xor_cipher(xor_encrypted_string, xor_key)
    return decrypted_aes_string


def master_encrypt(plaintext, cover_image_path = 'lena2.png', output_path = 'encoded_image.png',
                 output_qr_code_path = 'qr_code.png', display=False):
    """Encrypts a message using NTRU, AES, and XOR, then outputs a final encrypted string."""
    
    # NTRU Encryption
    # ntru_key_name = "high"
    # startkey = time.time()
    # pq_ntru.generate_keys(ntru_key_name, mode="high", debug=True)
    # key_gentime = time.time() - startkey
    # print("NTRU Keys generated in:", key_gentime)

    ntru_encrypted = pq_ntru.encrypt('dead2keys', plaintext)  # Bytes output

    # AES Encryption
    # aes_key = bytes(gan_bytes_all[i] for i in [256])  # AES-256 key

    aes_encrypted = encrypt_message(ntru_encrypted)  # Still bytes

    # XOR Encryption
    xor_encrypted = xor_encrypt(aes_encrypted)  # Final encrypted bytes
    print(f'Final Length: {len(xor_encrypted)}')

    # # Save XOR-encrypted result directly as binary
    # with open('encrypted_package.bin', 'wb') as f:
    #     f.write(xor_encrypted.encode('utf-8'))

    # print("Encryption complete and saved to 'encrypted_package.bin'.")
    # DCT Steganography
    img = cv2.imread(cover_image_path)
    img = cv2.resize(img, (4096, 4096))  # Resize to 4096x4096
    dct = IWT_FAST.IWT()
    encoded_img = dct.encode_image(img, xor_encrypted)  # Embeds bytes directly
    # encoded_img = embed_message(img,xor_encrypted)
    cv2.imwrite(output_path, encoded_img)
    blob_name = output_path + datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    # payload_image = encoded_image
    public_url = cloud_handler.upload_file(output_path, blob_name)
    cloud_handler.generate_qr_code(public_url, output_qr_code_path)
    qr_code_img = cv2.imread(output_qr_code_path)

    # if display:
    #     # Read images using OpenCV
    #     qr_code_img = cv2.imread(output_qr_code_path)
    #     qr_code_img = cv2.cvtColor(qr_code_img, cv2.COLOR_BGR2RGB)

    #     encoded_img_rgb = cv2.cvtColor(encoded_img, cv2.COLOR_BGR2RGB)
    #     img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    #     # Plot images using matplotlib
    #     plt.figure(figsize=(15, 5))

    #     plt.subplot(1, 3, 1)
    #     plt.imshow(qr_code_img)
    #     plt.title("QR Code")
    #     plt.axis('off')

    #     plt.subplot(1, 3, 2)
    #     plt.imshow(encoded_img_rgb)
    #     plt.title("Encoded Image")
    #     plt.axis('off')

    #     plt.subplot(1, 3, 3)
    #     plt.imshow(img_rgb)
    #     plt.title("Cover Image")
    #     plt.axis('off')

    #     plt.tight_layout()
    #     plt.show()
    
    return encoded_img, qr_code_img



# Master decryption function
def master_decrypt(input_data=None, downloaded_file_path = 'downloaded_file.png'):
    """
    Decrypts the given input data using XOR, AES, and NTRU decryption.
    
    Parameters:
    - input_data (str or None): If given, uses this directly; otherwise, reads from .bin file.

    Returns:
    - str: The original decrypted message.
    """
    decoded_url = cloud_handler.read_qr_code(input_data)
    cloud_handler.download_file(decoded_url, downloaded_file_path)

    # Extract encrypted text from the stego image
    img = cv2.imread(downloaded_file_path)  
    dct = IWT_FAST.IWT()
    final_encrypted_string = dct.decode_image(img)  # This now returns a plain string
    # final_encrypted_string = extract_message(img)

    # XOR Decryption
    aes_encrypted_string = xor_decrypt(final_encrypted_string)

    # AES Decryption
    ntru_encrypted_message = decrypt_message(aes_encrypted_string)

    # NTRU Decryption
    original_message = pq_ntru.decrypt("dead2keys", ntru_encrypted_message)

    return original_message


