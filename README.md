# Next-Gen Encryption Steganography

This repository demonstrates a highly sophisticated, complex, and secure encryption and steganography system. The project showcases a unique multi-layered approach to data security, where text is encrypted using several robust techniques and then hidden within an image using steganography. Finally, the image is uploaded to a secure cloud server with a generated QR code for discreet sharing.

## Table of Contents

- [Project Overview](#project-overview)
- [How It Works](#how-it-works)
- [Repository Structure](#repository-structure)
- [File Descriptions](#file-descriptions)
- [Getting Started](#getting-started)
  - [Prerequisites](#prerequisites)
  - [Installation](#installation)
  - [Training the GAN](#training-the-gan)
  - [Running the Demo](#running-the-demo)
- [Usage Instructions](#usage-instructions)
- [Evaluation](#evaluation)
- [Dependencies](#dependencies)
- [License](#license)
- [Acknowledgments](#acknowledgments)

## Project Overview

The goal of this project is to demonstrate an end-to-end secure system for encrypting and hiding text within images. The process involves:
- **Encryption Pipeline:**  
  1. **NTRUEncrypt:** The text is first encrypted using NTRUEncrypt.
  2. **AES-256 Encryption:** The output is further encrypted with AES-256.
  3. **XOR Encryption:** An additional layer of XOR-based encryption is applied.
- **Steganography:**  
  - **Integer Wavelet Transform (IWT):** The encrypted string is then embedded into an image using IWT-based steganography.
- **Cloud Integration and Sharing:**  
  - The steganographed image is uploaded to a secure cloud server, and a QR code that links to the image is generated for secure, discreet sharing.

## How It Works

1. **Encryption Process:**  
   - The original text message undergoes multiple encryption stages (NTRUEncrypt → AES-256 → XOR) to produce a secure cipher text.
2. **Steganography:**  
   - The cipher text is embedded into an image using the Integer Wavelet Transform (IWT) technique, making the encrypted data imperceptible to the naked eye.
3. **GAN-Generated Keys:**  
   - A Generative Adversarial Network (GAN) is used to produce highly random and unpredictable chaotic byte sequences. These sequences serve as encryption keys for AES and XOR encryption.
4. **Cloud and QR Code:**  
   - The resulting steganographed image is uploaded to a cloud server. A QR code linking to the image is generated to facilitate secure and easy sharing.

## Repository Structure

```
Next-Gen-Encryption-Steganography/
├── CHAOSDCGAN.py               # GAN architecture and inference code for key generation.
├── CloudQRHandler.py           # Functions to handle image upload to cloud and QR code generation/read.
├── IWT.py                      # Implementation of the Integer Wavelet Transform for steganography.
├── IWT_CLOUD_ENC_DEC.py        # Complete pipeline: encryption, upload, download, and decryption.
├── DCGANTRAINEVAL.ipynb        # Notebook for training and evaluating the GAN model.
├── BANGAYAAA.ipynb             # Demo notebook that demonstrates the encryption and decryption pipeline.
└── ENC_EVAL.py                 # Evaluation code for assessing the encryption and steganography process.
```

## File Descriptions

- **CHAOSDCGAN.py**  
  Implements the GAN architecture and inference routines. The GAN is trained to produce chaotic, random byte sequences, which are then used as encryption keys for AES-256 and XOR encryption.

- **CloudQRHandler.py**  
  Contains helper functions to manage cloud interactions. This includes uploading the steganographed image to a secure server, generating a QR code that links to the image, and reading the QR code to download the file.

- **IWT.py**  
  Provides the full implementation of the Integer Wavelet Transform used for embedding the encrypted text into an image.

- **IWT_CLOUD_ENC_DEC.py**  
  Integrates the entire process into a pipeline. This file includes all helper functions for encryption and steganography, as well as master functions that can be directly invoked by the user for the complete encryption–upload–download–decryption workflow.

- **DCGANTRAINEVAL.ipynb**  
  A Jupyter Notebook that contains the code for training and evaluating the GAN model. This is the first step required to generate the encryption keys.

- **BANGAYAAA.ipynb**  
  A demo notebook that illustrates the basic encryption and decryption pipeline using the trained models and provided functions. It serves as a quick-start guide to understanding the process.

- **ENC_EVAL.py**  
  Contains scripts and functions dedicated to the evaluation of the encryption and steganography process, ensuring that the system performs securely and as expected.

## Getting Started

### Prerequisites

Before running the code, ensure you have the following:
- Python 3.x installed.
- Jupyter Notebook for running the `.ipynb` files.
- Additional dependencies (see [Dependencies](#dependencies) below).

### Installation

1. **Clone the Repository:**

   ```bash
   git clone https://github.com/hchitnis7/Next-Gen-Encryption-Steganography.git
   ```

2. **Navigate to the Project Directory:**

   ```bash
   cd Next-Gen-Encryption-Steganography
   ```

3. **Install Dependencies:**

   Install the required Python packages. (A detailed list of dependencies will be provided in the repository; please refer to that list when available.)

   ```bash
   pip install -r requirements.txt
   ```

### Training the GAN

The GAN model must be trained before you can use it for encryption key generation. To train the GAN:

1. Open the `DCGANTRAINEVAL.ipynb` notebook:

   ```bash
   jupyter notebook DCGANTRAINEVAL.ipynb
   ```

2. Follow the instructions within the notebook to train the Generator and Discriminator models. Make sure to save the trained model weights for later use.

### Running the Demo

Once the GAN has been trained and the model weights are stored, you can test the encryption and decryption pipeline using the demo notebook:

1. Open the `BANGAYAAA.ipynb` notebook:

   ```bash
   jupyter notebook BANGAYAAA.ipynb
   ```

2. Execute the cells sequentially. The demo notebook will showcase:
   - Encrypting a text message using the multi-layer encryption pipeline.
   - Embedding the encrypted message into an image using IWT.
   - Uploading the steganographed image to a cloud server and generating a QR code.
   - Downloading and decrypting the image to retrieve the original message.

## Usage Instructions

To use the complete encryption and steganography pipeline:

1. **Train the GAN:**  
   Ensure that you have trained the GAN using `DCGANTRAINEVAL.ipynb` and saved the Generator and Discriminator models.

2. **Invoke the Pipeline:**  
   Use the functions provided in `IWT_CLOUD_ENC_DEC.py` to:
   - Encrypt the text.
   - Upload the resulting image to the cloud.
   - Generate a QR code for the uploaded image.
   - Download and decrypt the image to recover the original text.

3. **Evaluation:**  
   Utilize the scripts in `ENC_EVAL.py` to assess the performance and security of the encryption and steganography process.

## Evaluation

The repository includes evaluation tools to:
- Measure the randomness and unpredictability of the encryption keys generated by the GAN.
- Assess the robustness of the encryption and decryption pipeline.
- Validate the effectiveness of the IWT-based steganography in concealing the encrypted message.

## Dependencies

A list of required dependencies is provided (to be updated as per your instructions). Ensure all dependencies are installed before running the notebooks or scripts. Typically, these may include:
- TensorFlow / PyTorch (depending on the GAN implementation)
- NumPy
- OpenCV
- scikit-image
- Other standard Python libraries

## License

This project is provided for educational and research purposes. Please review the LICENSE file for details on how the project can be used and distributed.

## Acknowledgments

This project was developed to showcase a cutting-edge integration of encryption and steganography techniques. 
