import streamlit as st

# Must be first
st.set_page_config(page_title="Cryptonova", layout="wide")
import time
import cv2
import numpy as np
import tempfile, os
import warnings
warnings.filterwarnings("ignore")

from IWT_CLOUD_ENC_DEC_numba import master_encrypt, master_decrypt
from pyzbar.pyzbar import decode as decode_qr
from PIL import Image

# Hide branding
st.markdown("""
    <style>
        #MainMenu {visibility: hidden;}
        footer {visibility: hidden;}
        header {visibility: hidden;}
    </style>
""", unsafe_allow_html=True)

st.markdown("""
    <h1 style="text-align: center;">🔐 Cryptonova </h1>
    <p style="text-align: center; font-size: 18px;">
        Securely embed and extract messages from images using QR codes.
    </p>
    <p style="text-align: center; font-size: 16px;">
        Try it out with your own images and messages!
    </p>
    <p style="text-align: center; font-size: 14px;">    
        <br>1. In the "Encrypt" tab, enter your secret message and upload a cover image. Click "Encrypt & Embed" to generate a stego image and QR code.
        <br>2. Download the stego image and QR code from the provided options.
        <br>3. In the "Decrypt" tab, upload your QR code image and click "Decode & Decrypt" to recover the hidden message.
    </p>
""", unsafe_allow_html=True)

# Track session state
if 'show_encrypt_result' not in st.session_state:
    st.session_state.show_encrypt_result = False
if 'show_decrypt_result' not in st.session_state:
    st.session_state.show_decrypt_result = False

tab1, tab2 = st.tabs(["🔐 Encrypt", "🔓 Decrypt"])

with tab1:
    st.markdown("### 🛠️ Step 1: Encrypt & Embed Your Secret")

    col1, col2 = st.columns([2, 1])
    with col1:
        plaintext = st.text_area("Enter the secret message to encrypt", height=150)
    with col2:
        cover_file = st.file_uploader("Upload cover image", type=["png", "jpg", "jpeg"])

    col3, col4, col5 = st.columns([1, 2, 1])
    with col4:
        if st.button("🔐 Encrypt & Embed"):
            if not plaintext:
                st.error("Please enter a message to encrypt.")
            elif not cover_file:
                st.error("Please upload a cover image.")
            else:
                suffix = os.path.splitext(cover_file.name)[1]
                tmp_cover = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
                tmp_cover.write(cover_file.getvalue())
                tmp_cover.flush(); tmp_cover.close()

                # Inside the "🔐 Encrypt & Embed" button block
                start_time = time.time()
                

                st.info("Running encryption + stego encoding…")
                stego_img, qr_img = master_encrypt(
                    plaintext,
                    cover_image_path=tmp_cover.name,
                    output_path="stego.png",
                    output_qr_code_path="qr.png"
                )


                elapsed = time.time() - start_time
                st.markdown(f"⏱️ **Encryption time:** `{elapsed:.3f} seconds`")  # 👈 shows up on the page

                st.session_state.stego_rgb = cv2.cvtColor(stego_img, cv2.COLOR_BGR2RGB)
                st.session_state.qr_rgb = cv2.cvtColor(qr_img, cv2.COLOR_BGR2RGB)
                st.session_state.stego_buf = cv2.imencode(".png", stego_img)[1].tobytes()
                st.session_state.qr_buf = cv2.imencode(".png", qr_img)[1].tobytes()
                st.session_state.show_encrypt_result = True

    if st.session_state.show_encrypt_result:
        col_img1, col_img2 = st.columns([3, 1])  # Adjusted column widths
        with col_img1:
            st.markdown("#### 🖼️ Stego Image")
            st.image(st.session_state.stego_rgb, width=600)
            st.download_button("⬇️ Download Stego Image", st.session_state.stego_buf, "stego.png", "image/png")
        with col_img2:
            st.markdown("#### 📶 QR Code")
            st.image(st.session_state.qr_rgb, width=350)
            st.download_button("⬇️ Download QR Code", st.session_state.qr_buf, "qr.png", "image/png")
    
        st.markdown("---")
        st.button("🔁 Start Over", on_click=lambda: st.session_state.update(show_encrypt_result=False))



with tab2:
    st.markdown("### 🔓 Step 2: Decode & Decrypt")

    qr_input = st.file_uploader("Upload the QR code image", type=["png", "jpg", "jpeg"])

    col_dec1, col_dec2, col_dec3 = st.columns([1, 2, 1])
    with col_dec2:
        if st.button("🔓 Decode & Decrypt"):
            if not qr_input:
                st.error("Please upload a QR code image.")
            else:
                try:
                    suffix = os.path.splitext(qr_input.name)[1]
                    tmp_qr = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
                    tmp_qr.write(qr_input.getvalue())
                    tmp_qr.flush(); tmp_qr.close()

                    st.info("Reading QR, downloading stego image, extracting & decrypting…")
                    secret = master_decrypt(input_data=tmp_qr.name)
                    st.session_state.decrypted_text = secret
                    st.session_state.show_decrypt_result = True
                except Exception as e:
                    st.error(f"Failed to decrypt: {e}")

    if st.session_state.show_decrypt_result:
        st.success("✅ Success! Here’s your recovered message:")
        st.text_area("Decrypted Plaintext", st.session_state.decrypted_text, height=200)

        if st.button("🔁 Start Over", key="reset-decrypt"):
            st.session_state.show_decrypt_result = False
