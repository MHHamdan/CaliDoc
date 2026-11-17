# CaliDoc WACV 2027 talk: editable source

| File | Role |
|---|---|
| `build.js` | pptxgenjs generator for the deck: theme, layouts, native shapes, native charts, embedded notes |
| `notes.js` | Speaker notes; the single source for the PPTX notes pages and `../CaliDoc_WACV2027_Speaker_Notes.md` |
| `gen_speaker_notes.js` | Writes the Speaker Notes Markdown from `notes.js` |
| `make_equations.py` | Renders the display equations (matplotlib mathtext, 300 dpi PNG) into `assets/` |
| `assets/` | Equation PNGs and `cord_test_0014_showcase.png` (CORD v2, CC BY 4.0, copied from `assets/examples/`) |
| `render.sh` | Full pipeline: equations → PPTX → validation → LibreOffice PDF → per-slide PNGs for QA |
| `verify_numbers.py` | Cross-checks every number in the built PPTX against the camera-ready PDFs |
| `video_scripts.js` | Narration for the full (≤10 min) and short (≤5 min) talk videos |
| `make_video.py` | ElevenLabs narration (cached per text hash) + captions + chapters + ffmpeg assembly → `../video/` |

## Rebuild

```bash
# needs: node + pptxgenjs, jszip, react, react-dom, react-icons, sharp; python3 + matplotlib, PyMuPDF, python-pptx;
#        LibreOffice (tested: official 26.2.7 Linux build); fonts Carlito + Caladea (metric-compatible with Calibri/Cambria)
export NODE_MODULES=/path/to/node_modules PPTX_SKILL_DIR=/path/to/pptx-skill \
       SOFFICE=/path/to/soffice LO_PROFILE=/tmp/lo-profile QA_DIR=/tmp/calidoc-qa
./render.sh
python3 verify_numbers.py ../CaliDoc_WACV2027_Presentation.pptx \
  ../../../docs/CaliDoc_WACV2027_camera_ready/main.pdf ../../../docs/CaliDoc_WACV2027_supp_s12/01597_supp.pdf
```

`build.js` post-processes the written package to work around two pptxgenjs output issues:

- it removes undeclared chart axis IDs, which made LibreOffice drop combo charts;
- it keeps only the first `<a:pPr>` per paragraph, because multi-run paragraphs otherwise lost their bullets.

`validate.py` from the pptx skill passes on the result.

The deck uses the Office fonts Cambria (titles) and Calibri (body). The PDF was rendered with Caladea and Carlito, which are metric-compatible, so line breaks match PowerPoint.

## Talk videos

```bash
ELEVENLABS_API_KEY=... python3 make_video.py full short   # needs the built PDF; ffmpeg from imageio-ffmpeg
```

The synthesizer gets a phonetic version of the script (`SAY` in `make_video.py`): SROIE → "S-ROY", decimals spelled digit by digit so that 1.000 and 0.0008 cannot be misread. Captions keep the written text. The narration cache (`video_cache/`) is not committed.
