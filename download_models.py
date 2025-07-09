import os
from huggingface_hub import hf_hub_download
import logging

# Setup basic logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- Configuration ---
REPO_ID = "yl4579/StyleTTS2-LibriTTS"
SUBFOLDER = "Models/LibriTTS"
LOCAL_DIR = "StyleTTS2/Models/LibriTTS"

FILES_TO_DOWNLOAD = [
    "epochs_2nd_00020.pth",
    "config.yml",
    "style_encoder.pth",
    "text_aligner.pth"
]

def download_styletts2_models():
    """
    Downloads all necessary model files for StyleTTS2 from Hugging Face
    into the correct local directory structure.
    """
    logging.info(f"Ensuring local directory exists: {LOCAL_DIR}")
    os.makedirs(LOCAL_DIR, exist_ok=True)
    
    logging.info(f"Starting download of {len(FILES_TO_DOWNLOAD)} files from repo '{REPO_ID}'...")
    
    for filename in FILES_TO_DOWNLOAD:
        try:
            logging.info(f"Downloading '{filename}'...")
            hf_hub_download(
                repo_id=REPO_ID,
                filename=filename,
                subfolder=SUBFOLDER,
                local_dir=LOCAL_DIR,
                local_dir_use_symlinks=False  # Recommended for portability
            )
            logging.info(f"✅ Successfully downloaded '{filename}'.")
        except Exception as e:
            logging.error(f"❌ Failed to download '{filename}'. Error: {e}")
            logging.error("Please check your internet connection and Hugging Face token.")
            return

    logging.info("🎉 All StyleTTS2 models have been downloaded successfully.")

if __name__ == "__main__":
    download_styletts2_models()
