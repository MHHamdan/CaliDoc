#!/usr/bin/env bash
# Download official FUNSD from https://guillaumejaume.github.io/FUNSD/
# Verifies archive checksum is recorded and document counts match 149/50.
set -euo pipefail

RAW_DIR="/home/mhamdan/Calidoc/data/raw/funsd"
URL="https://guillaumejaume.github.io/FUNSD/dataset.zip"
ZIP="$RAW_DIR/dataset.zip"

mkdir -p "$RAW_DIR"

if [[ ! -f "$ZIP" ]]; then
    curl -fSL --retry 3 --retry-delay 5 -o "$ZIP" "$URL"
fi

sha256sum "$ZIP" | tee "$RAW_DIR/dataset.zip.sha256"

rm -rf "$RAW_DIR/dataset" "$RAW_DIR/__MACOSX"
python3 -m zipfile -e "$ZIP" "$RAW_DIR"

TRAIN=$(find "$RAW_DIR/dataset/training_data/annotations" -name '*.json' | wc -l)
TEST=$(find "$RAW_DIR/dataset/testing_data/annotations" -name '*.json' | wc -l)
TRAIN_IMG=$(find "$RAW_DIR/dataset/training_data/images" -name '*.png' | wc -l)
TEST_IMG=$(find "$RAW_DIR/dataset/testing_data/images" -name '*.png' | wc -l)

OK=PASS
[[ "$TRAIN" -eq 149 && "$TEST" -eq 50 && "$TRAIN_IMG" -eq 149 && "$TEST_IMG" -eq 50 ]] || OK=FAIL
echo "funsd    | download+extract       | expected 149/50 (ann=img) | actual ${TRAIN}/${TEST} ann, ${TRAIN_IMG}/${TEST_IMG} img | $OK"
[[ "$OK" == PASS ]]
