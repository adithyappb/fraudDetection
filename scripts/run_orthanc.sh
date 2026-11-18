#!/usr/bin/env bash
# Start a local Orthanc with the project config (storage in data/orthanc-db).
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p data/orthanc-db
python3 - <<'PY'
import json; c = json.load(open("orthanc/orthanc.json"))
c["StorageDirectory"] = c["IndexDirectory"] = "data/orthanc-db"
json.dump(c, open("data/orthanc.local.json", "w"), indent=2)
PY
if command -v Orthanc >/dev/null; then
  exec Orthanc data/orthanc.local.json
else
  echo "Orthanc not installed - using Docker"; exec docker run --rm -p 8042:8042 -p 4242:4242 \
    -v "$PWD/orthanc/orthanc.json:/etc/orthanc/orthanc.json:ro" orthancteam/orthanc:24.10.3
fi
