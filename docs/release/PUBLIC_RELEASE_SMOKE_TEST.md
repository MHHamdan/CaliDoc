# Public release smoke test

Date: 2026-10-17. Machine: local Linux workstation, fresh virtualenv, CPU inference.

## Status

| component | source | result |
|---|---|---|
| Hugging Face models | public `MHamdan/calidoc-layoutlmv3-large-*`, **anonymous** (no token, empty `HF_HOME`) | PASS |
| CaliDoc package and example inputs | `git clone` of commit `e41fc3d` from the local repository | PASS |
| CaliDoc package from **public GitHub**, clone path | `git clone https://github.com/MHHamdan/CaliDoc.git` at `2a83fa7`, `pip install .` | PASS (rerun 2026-10-17) |
| CaliDoc package from **public GitHub**, no clone | `pip install "calidoc @ git+https://github.com/MHHamdan/CaliDoc"`, inputs fetched from `raw.githubusercontent.com` | PASS (rerun 2026-10-17) |

## Commands (as run)

```bash
python -m venv venv && . venv/bin/activate
git clone <local checkout> CaliDoc && cd CaliDoc       # after the push: git clone https://github.com/MHHamdan/CaliDoc.git
pip install .
pip install matplotlib                                   # renderer only
unset HF_TOKEN HUGGING_FACE_HUB_TOKEN
export HF_HOME=$PWD/../hf_home HF_HUB_DISABLE_IMPLICIT_TOKEN=1

python -m calidoc.predict --model MHamdan/calidoc-layoutlmv3-large-cord \
  --image assets/examples/cord/cord_test_0042/document.png \
  --ocr   assets/examples/cord/cord_test_0042/ocr.json --out run1.json --device cpu
python -m calidoc.predict ... --out run2.json --device cpu   # same arguments
python tools/render_showcase_examples.py <dir with document.png, ocr.json, prediction.json>
```

## Results

The steps below follow the order of the commands above.

1. **Install `calidoc`.** `pip install .` resolved torch 2.14.1+cu130 and transformers
   5.20.1. The checkpoints were exported with transformers 5.12.1, so the test also covers a
   newer stack.
2. **Download the model.** The public CORD checkpoint downloaded anonymously.
3. **Load the processor and model.** `AutoProcessor` and `AutoModelForTokenClassification`
   loaded from the repository through `CaliDocPredictor`.
4. **Run on a license-cleared public example.** `cord_test_0042` (CORD v2 test, CC BY 4.0),
   using the image and OCR shipped in the repository.
5. **Check determinism.** `run1.json == run2.json`.
6. **Apply calibration.**
   - Thresholds come from the repository's `calibration/` files and are identical to the
     committed showcase: max |Δλ| = 0.
   - CPU scores differ from the GPU showcase by at most 3.0e-7.
   - Every field type, text, label, set size and status is identical.
   - Route `review`; 9 fields, all `ambiguous_review`.
7. **Produce output JSON.** `run1.json`.
8. **Render the visualization.** `showcase.png` was produced, at the same byte size as the
   committed figure.

## Hub-side verification (before the repositories were made public)

- **Hashes.** Each private repository was downloaded into an empty cache. The
  `model.safetensors` sha256 equals the exported file for all four models (see
  `HF_MODEL_ARTIFACT_AUDIT.md`).
- **Parity.** Full test-split F1 and the cache parity reproduce exactly:
  SROIE 0.949249, CORD 0.934445, FUNSD 0.827038, WildReceipt 0.940379.
- **Showcase.** All showcase JSON regenerated from the Hub checkpoints is identical to
  the run from the local export.
- **Model card code.** The Python blocks in the model card and in the README Quick Start
  were executed verbatim and ran without error.

## Public GitHub rerun (2026-10-17, after the push)

The rerun used fresh virtualenvs and anonymous Hub access, with torch 2.14.1+cu130 and
transformers 5.20.1.

- **A: `git clone` + `pip install .`**
  - Runs on `cord_test_0042` are deterministic.
  - Field types, texts, labels, set sizes, statuses and thresholds are identical to the
    committed showcase.
  - Route `review`; 9 fields, all `ambiguous_review`. The figure renders.
- **B: `pip install git+…` with no clone**
  - Uses the README / model-card Python API on `cord_test_0014`, with the image and OCR
    downloaded from GitHub raw.
  - Result: 25 fields, 24 `ambiguous_review` and 1 `conflict_review`. The conflict is
    *Coupon 100,000*, matching the committed example.

## Hugging Face Space

`MHamdan/CaliDoc-demo` was **not created**. The Hub rejected repository creation with
`402 Payment Required`: hosting Gradio Spaces on free cpu-basic hardware requires a PRO
subscription. The app is prepared and tested locally in `release/space/`. Its
`requirements.txt` pins `calidoc` to commit `2a83fa7`.

## Interactive demo clean room (2026-10-21)

Each check ran in its own fresh virtualenv outside the repository, with anonymous Hub
access (no token, empty `HF_HOME`).

| check | source | result |
|---|---|---|
| A. Demo examples | anonymous `hf_hub_download` from dataset `MHamdan/calidoc-demo-examples` | PASS |
| B. Package from public GitHub | `pip install "calidoc @ git+https://github.com/MHHamdan/CaliDoc.git"` | PASS |
| C. Gradio app | clone of the demo commit, `pip install -e ".[demo]"`, `python demo/gradio/app.py` (gradio 6.30.0, transformers 5.20.2), driven by headless Chromium | PASS |
| D. Colab notebook | `notebooks/CaliDoc_WACV2027_Demo.ipynb` executed with nbclient in a fresh kernel environment | PASS |

**A. Demo examples.** All file sha256 hashes match `metadata.json`. Image sizes are
1836×3264. Word counts equal box counts (20 and 40), each `expected_prediction.json`
carries its own `doc_id`, and the license is CC BY 4.0.

**B. Package from public GitHub.** The installed package was run on the downloaded
`cord_test_0014`. Its output equals `expected_prediction.json` (fields, labels, set sizes,
statuses): 24 `ambiguous_review` and 1 `conflict_review`.

**C. Gradio app.**
- The default tab loaded `cord_test_0042` with no input.
- Both built-in examples ran, and each reported "matches the published expected output".
- `cord_test_0014` showed the *Coupon 100,000* field as conflict / review.
- No private dataset access was needed.
- *Upload Your Own* was checked separately:
  - pixel and `normalized_1000` boxes give identical output;
  - malformed OCR JSON is rejected with a message.

**D. Colab notebook.**
- The optional `demo.launch(share=True)` cell is excluded; C covers it.
- All 7 code cells ran with 0 errors.
- Both examples reproduced the published output; `cord_test_0014` explains the Coupon
  conflict.
- **Stand-in:** C and D cloned the local commit instead of GitHub, because the demo commit
  is not pushed yet. Only the clone URL differs.
- **After the push:** re-run C and D against `https://github.com/MHHamdan/CaliDoc.git`, then
  open the Colab badge link.

No Hugging Face Space is used. Hosting a Gradio Space on the free tier requires a paid
account (HTTP 402), so the public demo paths are the GitHub source, Google Colab, and the
local app.
