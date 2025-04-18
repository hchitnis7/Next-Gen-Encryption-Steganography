
#!/usr/bin/env python3
"""
This module defines:
  - The ENC_EVAL class for evaluating steganographic modifications between an original (cover) image
    and a stego (encoded) image.
  - Methods to compute image quality and steganalysis metrics including MSE, PSNR, SSIM, entropy difference,
    chi-square test statistics, high-frequency noise analysis, and histogram analysis.
  - Each method provides detailed output along with interpretations to help assess the level of steganographic embedding.
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim
from scipy.stats import entropy, chisquare
from scipy.fftpack import fft2, fftshift


class ENC_EVAL:
    def __init__(self, cover_image_path, stego_image_path):
        """
        Initializes the ENC_EVAL object by loading and preprocessing the cover and stego images.

        Args:
            cover_image_path (str): Path to the cover (original) image.
            stego_image_path (str): Path to the stego (encoded) image.
        """
        self.cover_image = cv2.imread(cover_image_path)
        self.stego_image = cv2.imread(stego_image_path)

        if self.cover_image is None or self.stego_image is None:
            raise FileNotFoundError("One or both images could not be loaded. Check file paths.")

        # Convert images to grayscale for analysis
        self.cover_gray = cv2.cvtColor(self.cover_image, cv2.COLOR_BGR2GRAY)
        self.stego_gray = cv2.cvtColor(self.stego_image, cv2.COLOR_BGR2GRAY)

    def calculate_metrics(self):
        """
        Computes MSE, PSNR, SSIM, and entropy difference between the cover and stego images.
        Prints the metrics and returns them as a dictionary.

        Returns:
            dict: A dictionary containing the computed metrics.
        """
        # Compute Mean Squared Error (MSE)
        mse_value = np.mean((self.cover_gray - self.stego_gray) ** 2)

        # Compute Peak Signal-to-Noise Ratio (PSNR)
        psnr_value = psnr(self.cover_gray, self.stego_gray, data_range=255)

        # Compute Structural Similarity Index (SSIM)
        ssim_value, _ = ssim(self.cover_gray, self.stego_gray, full=True)

        # Compute Entropy for both images and determine the difference
        entropy_cover = entropy(np.histogram(self.cover_gray, bins=256, range=(0, 255))[0])
        entropy_stego = entropy(np.histogram(self.stego_gray, bins=256, range=(0, 255))[0])
        entropy_diff = abs(entropy_cover - entropy_stego)

        print(f"🔹 MSE: {mse_value:.4f}")
        print(f"🔹 PSNR: {psnr_value:.2f} dB")
        print(f"🔹 SSIM: {ssim_value:.4f}")
        print(f"🔹 Entropy Difference: {entropy_diff:.4f}")

        return {
            "MSE": mse_value,
            "PSNR": psnr_value,
            "SSIM": ssim_value,
            "Entropy_Difference": entropy_diff
        }

    def chi_square_test(self):
        """
        Performs a chi-square test on the pixel intensity histograms of the cover and stego grayscale images.
        Prints and returns the chi-square statistic.

        Returns:
            float: Chi-square statistic.
        """
        # Compute histograms for both images
        hist_cover = np.histogram(self.cover_gray, bins=256, range=(0, 255))[0]
        hist_stego = np.histogram(self.stego_gray, bins=256, range=(0, 255))[0]

        # Normalize histograms
        hist_cover = hist_cover / hist_cover.sum()
        hist_stego = hist_stego / hist_stego.sum()

        # Add a small value to avoid division by zero
        hist_cover += 1e-8
        hist_stego += 1e-8

        chi_square_stat, _ = chisquare(hist_stego, f_exp=hist_cover)

        print(f"🔍 Chi-Square Statistic: {chi_square_stat:.2f}")
        if chi_square_stat < 50:
            print("✅ No significant steganographic modifications detected.")
        elif 50 <= chi_square_stat < 200:
            print("⚠️ Possible slight embedding detected, further analysis recommended.")
        else:
            print("❌ High deviation detected – Strong evidence of steganographic content!")

        return chi_square_stat

    def high_frequency_analysis(self):
        """
        Computes high-frequency noise difference using Fourier transform between the cover and stego images.
        Prints and returns the noise level.

        Returns:
            float: Average high-frequency noise level.
        """
        fft_cover = np.abs(fftshift(fft2(self.cover_gray)))
        fft_stego = np.abs(fftshift(fft2(self.stego_gray)))

        noise_level = np.mean(np.abs(fft_stego - fft_cover))

        print(f"🔍 High-Frequency Noise Level: {noise_level:.4f}")
        if noise_level < 0.005:
            print("✅ No significant noise detected – Image likely unaltered.")
        elif 0.005 <= noise_level < 0.02:
            print("⚠️ Minor high-frequency noise detected – Possible modifications.")
        else:
            print("❌ Significant noise detected – Strong evidence of steganographic embedding!")

        return noise_level

    def histogram_analysis(self):
        """
        Plots histograms of pixel intensity distributions for both the cover and stego images.
        Also computes and prints a histogram difference score.
        """
        plt.figure(figsize=(10, 5))
        bins = 350

        plt.subplot(1, 2, 1)
        plt.hist(self.cover_gray.ravel(), bins, color='blue', alpha=1)
        plt.title("Cover Image Histogram")

        plt.subplot(1, 2, 2)
        plt.hist(self.stego_gray.ravel(), bins, color='red', alpha=1)
        plt.title("Encoded Image Histogram")

        plt.show()

        # Compute histogram difference score
        hist_diff = np.sum(np.abs(self.cover_gray - self.stego_gray)) / self.cover_gray.size
        print(f"🔍 Histogram Difference Score: {hist_diff:.4f}")
        if hist_diff < 0.01:
            print("✅ Almost identical histograms – No noticeable steganography.")
        elif 0.01 <= hist_diff < 0.05:
            print("⚠️ Small variations detected – Might be minor modifications.")
        elif 0.05 <= hist_diff < 0.15:
            print("❗ Noticeable differences – Possible hidden data.")
        else:
            print("❌ Strong histogram shift – Likely steganographic content detected!")

    def run_all_metrics(self):
        """
        Runs all evaluation metrics (calculate_metrics, chi_square_test, high_frequency_analysis, histogram_analysis).
        Returns a dictionary with the computed metric values.
        """
        metrics = self.calculate_metrics()
        metrics["Chi_Square"] = self.chi_square_test()
        metrics["High_Frequency_Noise"] = self.high_frequency_analysis()
        self.histogram_analysis()
        return metrics


if __name__ == "__main__":
    # Example usage:
    cover_path = "lena2.png"  # Replace with your cover image path
    stego_path = "/teamspace/studios/this_studio/DCT_STEG_IMG_crazy.png"  # Replace with your stego image path

    try:
        evaluator = ENC_EVAL(cover_path, stego_path)
        all_metrics = evaluator.run_all_metrics()
    except FileNotFoundError as e:
        print(f"Error: {e}")




# # Example usage for ENC_EVAL class:
# from enc_eval import ENC_EVAL

# # Dummy paths for the cover (original) and stego (encoded) images
# cover_image_path = "path/to/cover_image.png"
# stego_image_path = "path/to/stego_image.png"

# # Create an ENC_EVAL object
# eval_obj = ENC_EVAL(cover_image_path, stego_image_path)

# # Run all evaluation metrics (MSE, PSNR, SSIM, entropy difference, chi-square test, high-frequency analysis, and histogram analysis)
# all_metrics = eval_obj.run_all_metrics()
# print("Evaluation Metrics:", all_metrics)