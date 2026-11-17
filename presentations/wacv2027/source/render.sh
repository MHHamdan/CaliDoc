#!/usr/bin/env bash
# Build the deck, validate it, export the PDF with LibreOffice (official 26.2.7
# Linux build; Carlito/Caladea installed as metric-compatible substitutes for
# Calibri/Cambria), and rasterize every slide for visual QA.
#   NODE_MODULES=<dir with pptxgenjs, react, react-dom, react-icons, sharp>
#   PPTX_SKILL_DIR=<pptx skill dir, for apply_theme.js + validate.py>
#   SOFFICE=<path to soffice>   LO_PROFILE=<writable dir for the LibreOffice profile>
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
OUTDIR="$(dirname "$HERE")"
DECK="$OUTDIR/CaliDoc_WACV2027_Presentation.pptx"
QA="${QA_DIR:-$OUTDIR/source/qa}"
python3 "$HERE/make_equations.py"
NODE_PATH="$NODE_MODULES" PPTX_SKILL_DIR="$PPTX_SKILL_DIR" node "$HERE/build.js" "$DECK"
node "$HERE/gen_speaker_notes.js"
python3 "$PPTX_SKILL_DIR/scripts/office/validate.py" "$DECK"
TMP="$(mktemp -d)"; cp "$DECK" "$TMP/deck.pptx"
"$SOFFICE" -env:UserInstallation="file://$LO_PROFILE" --headless --convert-to pdf --outdir "$TMP" "$TMP/deck.pptx" >/dev/null
cp "$TMP/deck.pdf" "$OUTDIR/CaliDoc_WACV2027_Presentation.pdf"; rm -rf "$TMP"
mkdir -p "$QA"; rm -f "$QA"/slide-*.png
python3 - "$OUTDIR/CaliDoc_WACV2027_Presentation.pdf" "$QA" <<'PY'
import sys, fitz
d = fitz.open(sys.argv[1])
for i, p in enumerate(d):
    p.get_pixmap(dpi=110).save(f"{sys.argv[2]}/slide-{i+1:02d}.png")
print("pages", d.page_count)
PY
