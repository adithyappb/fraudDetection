#!/usr/bin/env bash
# Downloads the IEEE-CIS Fraud Detection dataset into data/raw/.
# Prerequisites (one time):
#   1. Kaggle account -> https://www.kaggle.com/settings -> "Create New Token" -> kaggle.json
#   2. mkdir -p ~/.kaggle && mv ~/Downloads/kaggle.json ~/.kaggle/ && chmod 600 ~/.kaggle/kaggle.json
#   3. Accept the competition rules: https://www.kaggle.com/competitions/ieee-fraud-detection/rules
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p data/raw

if ! command -v kaggle >/dev/null; then pip install -q kaggle; fi
if [ ! -f ~/.kaggle/kaggle.json ] && [ -z "${KAGGLE_USERNAME:-}" ]; then
  echo "No Kaggle credentials. See the header of this script." >&2; exit 1
fi

kaggle competitions download -c ieee-fraud-detection -f train_transaction.csv -p data/raw
kaggle competitions download -c ieee-fraud-detection -f train_identity.csv -p data/raw
cd data/raw
for f in *.zip; do [ -f "$f" ] && unzip -o -q "$f" && rm "$f"; done
ls -lh
echo "OK - now run: make train"
