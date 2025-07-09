import os
from huggingface_hub import hf_hub_download
import logging

# Setup basic logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- Configuration ---
REPO_ID = "yl4579/StyleTTS2-LibriTTS"
# This is the single LOCAL directory where all files need to end up.
LOCAL_DIR = "StyleTTS2/Models/LibriTTS"

# This list now contains the correct remote subfolder for EACH file.
# `None` means the file is at the root of the repository.
FILES_TO_DOWNLOAD = [
    {
        "filename": "epochs_2nd_00020.pth",
        "subfolder": "Models/LibriTTS" # This one is nested
    },
    {
        "filename": "config.yml",
        "subfolder": None # This is at the root
    },
    {
        "filename": "style_encoder.pth",
        "subfolder": None # This is at the root
    },
    {
        "filename": "text_aligner.pth",
        "subfolder": None # This is at the root
    }
]

def download_styletts2_models():
    """
    Downloads all necessary model files for StyleTTS2 from their specific
    locations in the Hugging Face repo into a single local directory.
    """
    logging.info(f"Ensuring local directory exists: {LOCAL_DIR}")
    os.makedirs(LOCAL_DIR, exist_ok=True)
    
    logging.info(f"Starting download of {len(FILES_TO_DOWNLOAD)} files from repo '{REPO_ID}'...")
    
    for file_info in FILES_TO_DOWNLOAD:
        filename = file_info["filename"]
        subfolder = file_info["subfolder"]
        
        try:
            logging.info(f"Downloading '{filename}' from remote subfolder: '{subfolder or 'root'}'...")
            
            # The key change is passing the correct subfolder for each file
            hf_hub_download(
                repo_id=REPO_ID,
                filename=filename,
                subfolder=subfolder,
                local_dir=LOCAL_DIR,
                local_dir_use_symlinks=False
            )
            logging.info(f"✅ Successfully downloaded '{filename}' to '{LOCAL_DIR}'.")
        except Exception as e:
            logging.error(f"❌ Failed to download '{filename}'. Error: {e}")
            logging.error("Please check your internet connection and Hugging Face token.")
            return

    logging.info(f"🎉 All StyleTTS2 models have been downloaded successfully to {LOCAL_DIR}.")

if __name__ == "__main__":
    download_styletts2_models()
