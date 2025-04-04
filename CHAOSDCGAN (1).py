#!/usr/bin/env python3
"""
dcgan_module.py

This module defines:
  - The generator model (ExpandedGeneratorNoSigmoid) exactly as used during training.
  - An inference function that generates N bytes of random data,
    applying a smart randomisation step (using CSPRNG replacement on the mode)
    so that the final output has a more uniform distribution.
  - Evaluation functions that compute individual metrics (Shannon entropy, KS test,
    Chi-Square test, Autocorrelation, Bitwise Balance, and Frequency Spectrum),
    as well as functions to produce and save plots.

Usage (from another notebook):
  import dcgan_module as dcm
  random_bytes = dcm.inference(model_path="expanded_generator_dcgan.pth", num_bytes=100000)
  metrics = dcm.evaluate_all(random_bytes)
  print(metrics)
"""

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import ks_1samp, chisquare, uniform
from numpy.fft import fft
import random
import secrets

# ---------------------------
# Global Configuration
# ---------------------------
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
LATENT_DIM = 1024
BATCH_SIZE = 32
MODEL_PATH_G = "expanded_generator_dcgan_9CHAN_GPT.pth"
# Tolerance for smart randomisation (mode frequency must be within ±20% of average)
TOLERANCE = 0.2

# ---------------------------
# Model Definition (Same as Training)
# ---------------------------
class NoiseInjection(nn.Module):
    """Inject Gaussian noise into feature maps."""
    def __init__(self, scale=0.3):
        super().__init__()
        self.scale = scale
    def forward(self, x):
        if self.training:
            noise = torch.randn_like(x) * self.scale
            return x + noise
        return x

class ExpandedGeneratorNoSigmoid(nn.Module):
    """
    DCGAN-like Generator modified to output raw values without a Sigmoid.
    This version includes NoiseInjection and Dropout layers as used during training.
    The output will have NUM_CHANNELS channels (default 6).
    """
    def __init__(self, latent_dim=LATENT_DIM, ngf=64, dropout=0.5, noise_scale=0.3, num_channels=6):
        super().__init__()
        self.main = nn.Sequential(
            nn.ConvTranspose2d(latent_dim, ngf*8, 4, 1, 0, bias=False),
            nn.BatchNorm2d(ngf*8),
            nn.ReLU(True),
            NoiseInjection(noise_scale),
            nn.Dropout2d(dropout),
            
            nn.ConvTranspose2d(ngf*8, ngf*4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf*4),
            nn.ReLU(True),
            NoiseInjection(noise_scale),
            nn.Dropout2d(dropout),
            
            nn.ConvTranspose2d(ngf*4, ngf*2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf*2),
            nn.ReLU(True),
            NoiseInjection(noise_scale),
            nn.Dropout2d(dropout),
            
            nn.ConvTranspose2d(ngf*2, ngf, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf),
            nn.ReLU(True),
            NoiseInjection(noise_scale),
            nn.Dropout2d(dropout),
            
            nn.ConvTranspose2d(ngf, num_channels, 4, 2, 1, bias=False)
            # No activation (raw output)
        )
    def forward(self, z):
        return self.main(z)

# ---------------------------
# Inference Functions
# ---------------------------
def load_generator(model_path=MODEL_PATH_G):
    """
    Loads the trained generator from the specified model path.
    Returns the generator in evaluation mode.
    """
    gen = ExpandedGeneratorNoSigmoid(latent_dim=LATENT_DIM).to(DEVICE)
    gen.load_state_dict(torch.load(model_path, map_location=DEVICE))
    gen.eval()
    return gen

def generate_expanded_images(gen, n_images=10):
    """
    Generates n_images from the generator using random latent vectors.
    Returns a list of generated image tensors.
    """
    gen.eval()
    images = []
    with torch.no_grad():
        for _ in range(n_images):
            z = torch.randn(1, LATENT_DIM, 1, 1, device=DEVICE)
            fake_img = gen(z)
            images.append(fake_img.cpu())
    return images

def extract_random_bytes_float_no_sigmoid(images, final_bytes=1024):
    """
    Converts each generated image (raw float32 tensor) to bytes (without scaling),
    concatenates them, and returns exactly final_bytes.
    """
    all_bytes = bytearray()
    for img in images:
        arr = img.squeeze().detach().cpu().numpy().astype(np.float32)
        all_bytes.extend(arr.tobytes())
        if len(all_bytes) >= final_bytes:
            break
    return bytes(all_bytes)[:final_bytes]

# ---------------------------
# Smart Randomisation Functions (using os.urandom)
# ---------------------------
def generate_one_random_byte_excluding_mode_csprng(exclude_val, max_tries=100):
    """
    Generates a single random byte using os.urandom that is NOT equal to exclude_val.
    """
    for _ in range(max_tries):
        rand_byte = os.urandom(1)[0]
        if rand_byte != exclude_val:
            return rand_byte
    return np.random.randint(0, 256)

def smart_replace_mode_csprng(data, exclude_val, fraction):
    """
    Replaces a fraction of occurrences of the byte exclude_val in data with random bytes from os.urandom.
    Returns the modified data and the number of replacements made.
    """
    data_array = bytearray(data)
    indices = [i for i, b in enumerate(data_array) if b == exclude_val]
    n_occurrences = len(indices)
    if n_occurrences == 0:
        return data, 0
    n_to_replace = int(n_occurrences * fraction)
    if n_to_replace <= 0:
        return data, 0
    replace_indices = np.random.choice(indices, size=n_to_replace, replace=False)
    replacements = 0
    for idx in replace_indices:
        new_val = generate_one_random_byte_excluding_mode_csprng(exclude_val)
        data_array[idx] = new_val
        replacements += 1
    return bytes(data_array), replacements

def smart_reduce_mode_frequency_csprng(data, fraction=0.5, tolerance=0.2, verbose=False):
    """
    Iteratively reduces the frequency of the mode in the data by replacing occurrences with os.urandom-based random bytes,
    until the mode's frequency is within ±20% of the average frequency.
    Returns the final data and total replacements made.
    """
    data_out = data
    total_len = len(np.frombuffer(data_out, dtype=np.uint8))
    avg = total_len / 256.0
    lower_bound = (1 - tolerance) * avg
    upper_bound = (1 + tolerance) * avg

    total_replacements = 0
    iteration = 0
    while True:
        arr = np.frombuffer(data_out, dtype=np.uint8)
        counts = np.bincount(arr, minlength=256)
        mode_val = int(np.argmax(counts))
        mode_freq = counts[mode_val]
        if verbose:
            print(f"Iteration {iteration}: Mode {mode_val} frequency = {mode_freq:.2f} (avg = {avg:.2f}, bounds = [{lower_bound:.2f}, {upper_bound:.2f}])")
        if lower_bound <= mode_freq <= upper_bound:
            break
        data_out, replacements = smart_replace_mode_csprng(data_out, mode_val, fraction)
        total_replacements += replacements
        iteration += 1
        if verbose:
            print(f"  Replaced {replacements} occurrences of {mode_val}")
    # print(f"Final mode frequency: {mode_freq} (target: between {lower_bound:.2f} and {upper_bound:.2f}).")
    # print(f"Total replacements made: {total_replacements}")
    return data_out, total_replacements

# ---------------------------
# Inference with Smart Randomisation
# ---------------------------
def inference(model_path = '/teamspace/studios/this_studio/expanded_generator_dcgan_9CHAN_GPT.pth', num_bytes = 10000000):
    """
    Loads the generator, generates raw random data from images, applies the smart randomisation
    step (reducing mode frequency using CSPRNG-based replacement), and returns the final randomised bytes.
    
    Parameters:
      model_path (str): Path to the saved generator model.
      num_bytes (int): Number of random bytes to generate.
    
    Returns:
      bytes: Final randomised byte sequence.
    """
    gen = load_generator(model_path)
    all_bytes = bytearray()
    # Generate images until we have enough bytes.
    while len(all_bytes) < num_bytes:
        images = generate_expanded_images(gen, n_images=BATCH_SIZE)
        extracted = extract_random_bytes_float_no_sigmoid(images, final_bytes=num_bytes)
        all_bytes.extend(extracted)
    raw_bytes = bytes(all_bytes)[:num_bytes]
    # Apply smart randomisation using os.urandom-based replacement.
    final_bytes, total_repl = smart_reduce_mode_frequency_csprng(raw_bytes, fraction=0.5, tolerance=0.45, verbose=False)
    # print(f"Total CSPRNG replacements during inference: {total_repl}")
    return final_bytes

# ---------------------------
# Evaluation Functions
# ---------------------------
def evaluate_shannon_entropy(data):
    counts = np.bincount(np.frombuffer(data, dtype=np.uint8), minlength=256)
    probabilities = counts / float(len(data))
    probabilities = probabilities[np.nonzero(probabilities)]
    return -np.sum(probabilities * np.log2(probabilities))

def evaluate_ks_test(data):
    data_norm = np.frombuffer(data, dtype=np.uint8).astype(np.float32) / 255.0
    statistic, p_value = ks_1samp(data_norm, uniform.cdf)
    return statistic, p_value

def evaluate_chi_square(data):
    counts = np.bincount(np.frombuffer(data, dtype=np.uint8), minlength=256)
    expected = np.full(256, len(data)/256.0)
    chi2_stat, p_value = chisquare(counts, f_exp=expected)
    return chi2_stat, p_value

def evaluate_autocorrelation(data, lag=1):
    data_arr = np.frombuffer(data, dtype=np.uint8).astype(np.float32)
    n = len(data_arr)
    mean = np.mean(data_arr)
    c0 = np.sum((data_arr - mean) ** 2) / n
    c1 = np.sum((data_arr[:n-lag] - mean) * (data_arr[lag:] - mean)) / n
    return c1 / c0

def evaluate_bitwise_balance(data):
    bits = np.unpackbits(np.frombuffer(data, dtype=np.uint8))
    ones = np.sum(bits)
    zeros = len(bits) - ones
    return zeros, ones

def plot_histogram(data, title="Histogram", filename=None):
    arr = np.frombuffer(data, dtype=np.uint8)
    plt.figure(figsize=(8,6))
    plt.hist(arr, bins=256, range=(0,255), color='blue', alpha=0.7)
    plt.title(title)
    plt.xlabel("Byte Value")
    plt.ylabel("Frequency")
    if filename:
        plt.savefig(filename)
    else:
        plt.show()

def plot_frequency_spectrum(data, title="Frequency Spectrum", filename=None):
    data_arr = np.frombuffer(data, dtype=np.uint8)
    spectrum = np.abs(fft(data_arr))
    plt.figure(figsize=(10,4))
    plt.plot(spectrum[:len(spectrum)//2])
    plt.title(title)
    plt.xlabel("Frequency")
    plt.ylabel("Magnitude")
    if filename:
        plt.savefig(filename)
    else:
        plt.show()

def evaluate_all(data):
    metrics = {}
    metrics['shannon_entropy'] = evaluate_shannon_entropy(data)
    metrics['ks_statistic'], metrics['ks_p_value'] = evaluate_ks_test(data)
    metrics['chi_square_statistic'], metrics['chi_square_p_value'] = evaluate_chi_square(data)
    metrics['autocorrelation'] = evaluate_autocorrelation(data)
    metrics['bitwise_balance'] = evaluate_bitwise_balance(data)
    
    plot_histogram(data, title="Generated Data Histogram", filename="histogram.png")
    plot_frequency_spectrum(data, title="Generated Data Frequency Spectrum", filename="frequency_spectrum.png")
    
    return metrics

# ---------------------------
# Main Section (Example Usage)
# ---------------------------
if __name__ == '__main__':
    # Generate 100,000 bytes using the inference function.
    num_bytes_to_generate = 100000
    final_random_bytes = inference(MODEL_PATH_G, num_bytes_to_generate)
    print("First 16 bytes of final randomised output:", final_random_bytes[:16])
    
    # Evaluate the final output.
    eval_results = evaluate_all(final_random_bytes)
    print("Evaluation Metrics:")
    for k, v in eval_results.items():
        print(f"{k}: {v}")
