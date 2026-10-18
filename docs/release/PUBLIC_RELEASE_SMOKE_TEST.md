# Public release smoke test

Date: 2026-10-17. Machine: lytos (Linux), fresh virtualenv, CPU inference.

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
git clone /home/mhamdan/Calidoc CaliDoc && cd CaliDoc    # after the push: git clone https://github.com/MHHamdan/CaliDoc.git
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
