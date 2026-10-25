# CaliDoc Interactive Demo

A Gradio app for the released WACV 2027 checkpoints. It shows, field by field, whether
each complete extracted field should pass through, be reviewed, or abstain, and why.

## Fastest option

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/MHHamdan/CaliDoc/blob/main/notebooks/CaliDoc_WACV2027_Demo.ipynb)

The notebook installs CaliDoc from GitHub, runs both real examples, lets you upload your own
image, and can launch this app with a temporary public link. It needs no account, token or
API key.

## Run locally

```bash
git clone https://github.com/MHHamdan/CaliDoc.git
cd CaliDoc
pip install -e ".[demo]"
python demo/gradio/app.py
```

Python 3.10 or newer. pip selects docTR 1.0.x on Python 3.10 and docTR 1.1.x on 3.11+. Both
were tested, as was Colab's Python 3.12.

Then open http://127.0.0.1:7860.

On first use the app downloads:
- the CORD checkpoint (1.4 GB) and the two example documents, from the Hugging Face Hub;
- the docTR OCR weights (about 100 MB), the first time an image is uploaded.

CaliDoc runs on a GPU if one is available, otherwise on CPU. OCR always runs on CPU.

## Inputs

- **Try a Real Example** (default tab): no input needed. Two CORD v2 test receipts are
  fetched from the public dataset repository
  [`MHamdan/calidoc-demo-examples`](https://huggingface.co/datasets/MHamdan/calidoc-demo-examples):
  - `cord_test_0042`, the representative receipt;
  - `cord_test_0014`, the review-routing case. It contains the real *Coupon 100,000*
    extraction, which the released model routes to review.

  The model is fixed to the CORD checkpoint, and each run is checked against the published
  expected output.
- **Upload Your Own**: upload a document image (PNG, JPEG or WEBP). The public demo performs
  OCR locally with [docTR](https://github.com/mindee/doctr). Advanced users may optionally
  provide OCR JSON to reproduce a specific preprocessing pipeline:

  ```json
  {"words": ["Total", "16,500"], "boxes": [[435, 536, 520, 567], [600, 536, 686, 567]], "box_format": "pixel"}
  ```

  `box_format` is `"pixel"` (the default) or `"normalized_1000"`. Automatic OCR and OCR JSON
  go through the same CaliDoc inference call. docTR was not used in the WACV experiments, and
  user uploads are exploratory: they do not receive the paper's formal guarantee. Uploaded
  images stay on the machine running the app. PDF input is not supported yet.

## Output

| column | meaning |
|---|---|
| Set size | how many labels the calibrated set keeps for the field's words (a range when words differ) |
| Calibration | `field`: the field type has its own SpanCRC threshold. `pooled`: it falls back to the pooled threshold |
| State | `singleton` (set is exactly the prediction), `ambiguous` (set keeps the prediction plus other labels), `conflict` (set excludes the prediction), `empty` (no label clears the threshold) |
| Action | `pass`, `review` or `abstain` |

Colours in the overlay show operational routing, not a probability of correctness. The
raw statuses (`covered`, `ambiguous_review`, `conflict_review`, `empty_abstain`) stay in
the prediction JSON.

## Models

Each model is dataset-specific:

| dataset | model |
|---|---|
| CORD | [MHamdan/calidoc-layoutlmv3-large-cord](https://huggingface.co/MHamdan/calidoc-layoutlmv3-large-cord) |
| FUNSD | [MHamdan/calidoc-layoutlmv3-large-funsd](https://huggingface.co/MHamdan/calidoc-layoutlmv3-large-funsd) |
| WildReceipt | [MHamdan/calidoc-layoutlmv3-large-wildreceipt](https://huggingface.co/MHamdan/calidoc-layoutlmv3-large-wildreceipt) |
| SROIE | [MHamdan/calidoc-layoutlmv3-large-sroie](https://huggingface.co/MHamdan/calidoc-layoutlmv3-large-sroie) |

The weights are CC BY-NC-SA 4.0 (non-commercial), inherited from LayoutLMv3.

## Scientific scope

- **Guarantee.** CaliDoc's finite-sample guarantee bounds the expected per-document
  fraction of missed gold fields. It holds for documents exchangeable with the calibration
  documents of the chosen checkpoint. Documents from other domains, layouts or OCR engines
  do not carry it.
- **Calibration is dataset-specific.** On CORD most field types fall back to the pooled
  threshold, and one supported type (`MENU.SUB.CNT`) has threshold 1.0. As a result, most
  CORD fields are routed for review. The paper documents this behaviour.
- **The audit is corpus-level.** The boundary-escape annotation audit runs over a labelled
  corpus and is not part of single-document inference.
