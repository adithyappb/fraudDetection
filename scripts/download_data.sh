#!/usr/bin/env bash
# Download the real datasets into data/raw/ (replaces the phantoms).
#   bash scripts/download_data.sh            # NIH sample (5.6k images, 2 GB) + MSD spleen (1.5 GB)
#   bash scripts/download_data.sh full       # full NIH ChestX-ray14 (112k images, 45 GB)
# NIH needs a Kaggle API token: kaggle.com -> Settings -> API -> Create New Token -> ~/.kaggle/kaggle.json
set -euo pipefail
cd "$(dirname "$0")/.."
export PYTHONPATH=src
mkdir -p data/raw/downloads

# --- NIH ChestX-ray14 (Kaggle mirror of the NIH Clinical Center release) ---
DS=nih-chest-xrays/sample; [ "${1:-}" = "full" ] && DS=nih-chest-xrays/data
pip show kaggle >/dev/null 2>&1 || pip install -q kaggle
kaggle datasets download -d "$DS" -p data/raw/downloads --unzip
rm -rf data/raw/nih-cxr14
python -c "from medai.data.nih import import_kaggle; import_kaggle('data/raw/downloads', 'data/raw/nih-cxr14')"

# --- Medical Segmentation Decathlon Task09_Spleen (MONAI's official mirror) ---
rm -rf data/raw/Task09_Spleen
python -c "from monai.apps import DecathlonDataset; DecathlonDataset('data/raw', 'Task09_Spleen', 'training', download=True, cache_rate=0)"

echo "Real data in data/raw. Next: dvc add data/raw && dvc repro"
