#!/bin/bash
set -e

CACHE_DIR="/cache/models"
MODELS_DIR="/models"

mkdir -p /root/.cache/torch/hub/checkpoints

download() {
  local file_url="$1"
  local destination_path="$2"
  local cache_path="${CACHE_DIR}/${destination_path##*/}"
  mkdir -p "$(dirname "$cache_path")"
  mkdir -p "$(dirname "$destination_path")"
  if [ ! -e "$cache_path" ]; then
    echo "Downloading $file_url to cache..."
    wget -O "$cache_path" "$file_url"
  else
    echo "Using cached version of ${cache_path##*/}"
  fi
  cp "$cache_path" "$destination_path"
}

# ===============================
# Download Faster Whisper Model
# ===============================
faster_whisper_model_dir="${MODELS_DIR}/faster-whisper-large-v3"
mkdir -p "$faster_whisper_model_dir"

download "https://huggingface.co/Systran/faster-whisper-large-v3/resolve/main/config.json"               "$faster_whisper_model_dir/config.json"
download "https://huggingface.co/Systran/faster-whisper-large-v3/resolve/main/model.bin"                "$faster_whisper_model_dir/model.bin"
download "https://huggingface.co/Systran/faster-whisper-large-v3/resolve/main/preprocessor_config.json" "$faster_whisper_model_dir/preprocessor_config.json"
download "https://huggingface.co/Systran/faster-whisper-large-v3/resolve/main/tokenizer.json"           "$faster_whisper_model_dir/tokenizer.json"
download "https://huggingface.co/Systran/faster-whisper-large-v3/resolve/main/vocabulary.json"          "$faster_whisper_model_dir/vocabulary.json"

echo "Faster Whisper model downloaded."

# ===================================
# Python block: Hugging Face downloads
# ===================================
python3 -c "
import os
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    print('WARNING: python-dotenv not installed, skipping .env loading')

from huggingface_hub import snapshot_download

# Leggi HF token da BuildKit secret o env
hf_token = None
try:
    with open('/run/secrets/hf_token', 'r') as f:
        hf_token = f.read().strip()
    print('HF token loaded from secret file.')
except Exception as e:
    print('No secret file found, fallback a env var:', e)
    hf_token = os.environ.get('HF_TOKEN')

# SpeechBrain (pubblico, no token)
snapshot_download(repo_id='speechbrain/spkrec-ecapa-voxceleb')

# PyAnnote (richiede token + accettazione termini su HF)
if hf_token:
    snapshot_download(repo_id='pyannote/embedding',                  token=hf_token)
    snapshot_download(repo_id='pyannote/speaker-diarization-3.1',    token=hf_token)
    print('PyAnnote models downloaded.')
else:
    print('WARNING: HF_TOKEN non impostato, skip pyannote models.')
"

echo "All models downloaded successfully."