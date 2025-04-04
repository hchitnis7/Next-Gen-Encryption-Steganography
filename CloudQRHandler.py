#!/usr/bin/env python3
"""
This module defines:
  - The CloudQRHandler class, which provides methods for interacting with Google Cloud Storage
    and handling QR code operations.
  - Methods include:
      - upload_file: Upload a local file to a specified Google Cloud Storage bucket and return its public URL.
      - generate_qr_code: Generate and save a QR code from a provided URL.
      - read_qr_code: Read and decode a QR code from an image file.
      - download_file: Download a file from a given URL and save it locally.
"""

import os
import cv2
import requests
import qrcode
from google.cloud import storage


class CloudQRHandler:
    def __init__(self, credentials_path: str, bucket_name: str):
        """
        Initializes the CloudQRHandler with the Google Cloud credentials and bucket name.

        Args:
            credentials_path (str): Path to the Google Cloud service account JSON key file.
            bucket_name (str): Name of the Google Cloud Storage bucket.
        """
        os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = credentials_path
        self.bucket_name = bucket_name
        self.storage_client = storage.Client()

    def upload_file(self, local_file_path: str, destination_blob_name: str) -> str:
        """
        Uploads a local file to Google Cloud Storage and makes it publicly accessible.

        Args:
            local_file_path (str): Path to the local file.
            destination_blob_name (str): Destination blob name in the bucket.

        Returns:
            str: Public URL of the uploaded file.
        """
        bucket = self.storage_client.get_bucket(self.bucket_name)
        blob = bucket.blob(destination_blob_name)
        blob.upload_from_filename(local_file_path)
        blob.make_public()

        public_url = f"https://storage.googleapis.com/{self.bucket_name}/{destination_blob_name}"
        return public_url

    def generate_qr_code(self, file_url: str, output_filename: str = "qr_code.png") -> None:
        """
        Generates a QR code from the provided URL and saves it as a PNG image.

        Args:
            file_url (str): URL to encode in the QR code.
            output_filename (str): Filename for saving the QR code image (default is 'qr_code.png').
        """
        qr = qrcode.make(file_url)
        qr.save(output_filename)
        print(f"QR Code saved as {output_filename}!")

    def read_qr_code(self, qr_code_image_path: str) -> str:
        """
        Reads and decodes a QR code from the specified image file.

        Args:
            qr_code_image_path (str): Path to the image containing the QR code.

        Returns:
            str: Decoded value from the QR code, or None if not detected.
        """
        image = cv2.imread(qr_code_image_path)
        detector = cv2.QRCodeDetector()
        decoded_value, points, _ = detector.detectAndDecode(image)

        if decoded_value:
            print(f"QR Code detected. Extracted URL: {decoded_value}")
            return decoded_value
        else:
            print("No QR Code detected.")
            return None

    def download_file(self, file_url: str, download_path: str) -> None:
        """
        Downloads a file from a specified URL and saves it locally.

        Args:
            file_url (str): URL of the file to download.
            download_path (str): Local path where the file should be saved.
        """
        response = requests.get(file_url)
        if response.status_code == 200:
            with open(download_path, 'wb') as file:
                file.write(response.content)
            print(f"File downloaded and saved to {download_path}.")
        else:
            print("Failed to download the file.")


if __name__ == "__main__":
    # Example usage:
    credentials = "/teamspace/studios/this_studio/project_dir/logical-factor-455520-m8-2405e5b6a50c.json"
    bucket = "your-bucket-name"  # Replace with your actual bucket name

    # Initialize the handler
    handler = CloudQRHandler(credentials, bucket)

    # Upload a file to Google Cloud Storage
    local_file = "example.png"  # Replace with your local file path
    destination_blob = "uploads/example.png"  # Destination path in the bucket
    public_url = handler.upload_file(local_file, destination_blob)
    print("Uploaded file public URL:", public_url)

    # Generate a QR code for the uploaded file's URL
    handler.generate_qr_code(public_url, "uploaded_qr.png")

    # Read the QR code from the saved image
    decoded_url = handler.read_qr_code("uploaded_qr.png")

    # Download the file from the public URL
    handler.download_file(public_url, "downloaded_example.png")


# # Example usage for CloudQRHandler class:
# from cloud_qr_handler import CloudQRHandler

# # Dummy credentials and bucket name for Google Cloud Storage
# credentials_path = "path/to/google_credentials.json"
# bucket_name = "your-bucket-name"

# # Create a CloudQRHandler object
# cloud_handler = CloudQRHandler(credentials_path, bucket_name)

# # Dummy file paths and blob destination for upload
# local_file_path = "path/to/local_file.png"
# destination_blob_name = "uploads/local_file.png"

# # Upload the file and retrieve its public URL
# public_url = cloud_handler.upload_file(local_file_path, destination_blob_name)
# print("Public URL:", public_url)

# # Generate a QR code for the public URL
# cloud_handler.generate_qr_code(public_url, output_filename="qr_code.png")

# # Read and decode the QR code from the generated image
# decoded_url = cloud_handler.read_qr_code("qr_code.png")
# print("Decoded QR Code URL:", decoded_url)

# # Download the file from the public URL to a local path
# download_path = "path/to/downloaded_file.png"
# cloud_handler.download_file(public_url, download_path)