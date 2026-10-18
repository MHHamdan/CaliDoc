#!/usr/bin/env python3
"""Write the Hugging Face model card (README.md) for each released checkpoint.

All numbers come from machine-written artifacts: the release's
`calidoc_config.json` (this checkpoint) and the camera-ready
`results_macros.tex` (paper values). Nothing is typed by hand.

    python tools/write_model_cards.py --release release/hf \
        --macros docs/CaliDoc_WACV2027_camera_ready/results_macros.tex
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

GITHUB = "https://github.com/MHHamdan/CaliDoc"
SUPP = f"{GITHUB}/releases/download/wacv2027-camera-ready/CaliDoc_WACV2027_Supplementary.pdf"

META = {
    "sroie": {
        "title": "SROIE",
        "hub_datasets": ["darentang/sroie"],
        "about": "scanned Malaysian retail receipts (ICDAR 2019 SROIE Task 3), four fields: "
                 "COMPANY, ADDRESS, DATE, TOTAL",
        "lineage": "the `darentang/sroie` pre-tagged BIO lineage (official 626/347 train/test "
                   "split; 60 training documents held out for early stopping and temperature)",
        "data_terms": "SROIE is distributed by the ICDAR 2019 Robust Reading Competition "
                      "(https://rrc.cvc.uab.es/?ch=13, registration required). A CC BY 4.0 "
                      "license is asserted by third-party mirrors; we could not confirm it on "
                      "the organizers' public pages.",
        "citation": "Huang et al., ICDAR2019 Competition on Scanned Receipt OCR and Information "
                    "Extraction, ICDAR 2019",
    },
    "cord": {
        "title": "CORD",
        "hub_datasets": ["naver-clova-ix/cord-v2"],
        "about": "Indonesian restaurant receipts (CORD v2), 29 field types / 54 BIO labels "
                 "derived from the data (no O class)",
        "lineage": "`naver-clova-ix/cord-v2` at revision "
                   "`7f0115a4b758a71d6473b8d085751692da2fef98` (800/100/100; the 100 "
                   "validation receipts are split 50 early-stopping / 50 conformal calibration)",
        "data_terms": "CORD v2 is released under CC BY 4.0 by its authors.",
        "citation": "Park et al., CORD: A Consolidated Receipt Dataset for Post-OCR Parsing, "
                    "Document Intelligence Workshop at NeurIPS 2019",
    },
    "funsd": {
        "title": "FUNSD",
        "hub_datasets": [],
        "about": "noisy scanned administrative forms (FUNSD), labels HEADER, QUESTION, ANSWER "
                 "and O",
        "lineage": "the official FUNSD archive (https://guillaumejaume.github.io/FUNSD/), "
                   "149/50 train/test, with 15 training forms held out for early stopping and "
                   "temperature",
        "data_terms": "FUNSD is distributed under the FUNSD License for non-commercial research "
                      "and educational use; its images (from RVL-CDIP) remain under "
                      "third-party copyright.",
        "citation": "Jaume, Ekenel, Thiran, FUNSD: A Dataset for Form Understanding in Noisy "
                    "Scanned Documents, ICDAR-OST 2019",
    },
    "wildreceipt": {
        "title": "WildReceipt",
        "hub_datasets": [],
        "about": "receipts photographed in the wild (WildReceipt), 24 key/value field types "
                 "on OCR boxes (`Others` mapped to O, `Ignore` boxes dropped)",
        "lineage": "the OpenMMLab WildReceipt release, obtained from two byte-identical Kaggle "
                   "mirrors because the canonical OpenMMLab host was unreachable (official "
                   "1,267/472 split; 100 training receipts held out for early stopping and "
                   "temperature)",
        "data_terms": "WildReceipt's authors state no license (the MMOCR dataset zoo lists it "
                      "as N/A); obtain it from the official MMOCR instructions.",
        "citation": "Sun et al., Spatial Dual-Modality Graph Reasoning for Key Information "
                    "Extraction, arXiv:2103.14470, 2021",
    },
}

DS_NOTES = {
    "sroie": """\
**This is a case-study checkpoint, not a clean benchmark model.** The pre-tagged
SROIE labels come from an upstream fuzzy string alignment; our own alignment
gate measured about 5–6% of entity strings that do not align exactly. In the
paper, the boundary-escape audit flags exactly `ADDRESS` (edge enrichment
{sroieAddrEnr}, Holm-adjusted p = {sroieAddrHolm}) and `COMPANY` ({sroieCompEnr}, p =
{sroieCompHolm}), and a separately trained repaired-label lineage (not released
here) removes the positional signal. Expect boundary errors on ADDRESS and COMPANY
that partly reflect the training labels.

SROIE is where token-level conformal prediction fails most visibly: under a single
token-marginal threshold the seed-42 model reaches {sroieTokenCov}% token coverage at
a 90% target while **none** of the {sroieNEvSpans} gold field spans is fully covered
(span coverage {sroieSpanCovTok}), because the threshold is sized for the background
class that holds {sroieOShare}% of tokens. Field-conditional SpanCRC restores span
coverage to {sroieSpanCovFld}.""",
    "cord": """\
**Expect almost every CORD field to be flagged for review.** CORD has 29 field
types, and most are rare: in the released calibration split only
{n_supported} field types reach m0 = 25 calibration spans and get their own
threshold, while {n_fallback} fall back to the pooled threshold (including types
never seen in calibration). Some supported long-tail types need very large
thresholds (`MENU.SUB.CNT` has λ = {lam_subcnt}, `MENU.SUB.NM` λ = {lam_subnm}), so
their labels enter nearly every word's calibrated set. The paper reports this as
CaliDoc's weakest case: average set size {cordSizeTok} → {cordSizeFld} under the
field-conditional rule, with span coverage {cordSpanCovTok} → {cordSpanCovFld}. Under this
release, the useful review signal on CORD is *which* extra labels a field's set
contains (`other_labels_in_set`), not whether it is ambiguous. The paper's
interpolated variant, which removes most of this set-size cost, is not what this
repository ships.

The seed-42 CORD checkpoint is the low tail of the three training seeds
({cordFSeedList} test F1): early stopping fired at epoch 10 on a 50-document split. It
is released because seed 42 is the paper's predeclared ablation checkpoint, not
because it is the best one.""",
    "funsd": """\
FUNSD thresholds are large (pooled λ = {pooled:.3f}), because form fields are long and
the max-token score is conservative for long spans. Most fields therefore get
calibrated sets with more than one label and are routed to review. The
calibration split is small (35 forms), so the document-level bound is loose.""",
    "wildreceipt": """\
WildReceipt is CaliDoc's easiest case: most field types have enough support for
their own threshold, and calibrated sets stay close to singletons (paper: average
set size {wildreceiptSizeFld}, span coverage {wildreceiptSpanCovFld}). Most WildReceipt
fields are one or two words long, so the **boundary-escape audit has almost no
power here**. In the paper only {wildreceiptNEligible} of {wildreceiptNPresent} field types were eligible for
the test, and a clean audit result on short fields means *untested*, not *clean*.""",
}


def card(ds: str, cfg: dict, fields: dict, m: dict, num_labels: int, temp: dict,
         collection: str) -> str:
    meta = META[ds]
    name = f"calidoc-layoutlmv3-large-{ds}"
    t = cfg["released_checkpoint_test"]
    ev = cfg["evaluation_side_of_unit"]
    sup = cfg["supported_field_types"]
    fb = cfg["fallback_field_types"]
    P = lambda k: m[f"{ds}{k}"]
    fmt = dict(m)
    fmt.update(n_supported=len(sup), n_fallback=len(fb), pooled=cfg["pooled_lambda"],
               lam_subcnt=f"{fields.get('MENU.SUB.CNT', {}).get('lambda', 0):.3f}",
               lam_subnm=f"{fields.get('MENU.SUB.NM', {}).get('lambda', 0):.3f}")
    note = DS_NOTES[ds].format(**fmt)
    yaml_ds = "".join(f"- {d}\n" for d in meta["hub_datasets"])
    yaml_ds = f"datasets:\n{yaml_ds}" if yaml_ds else ""
    unit = ("the fixed 50-receipt calibration split" if cfg["calibration_unit"] == "fixed"
            else "calibration split seed 42")
    rows = "\n".join(
        f"| `{f}` | {v['lambda']:.4f} | {v['n_cal_spans']} | "
        f"{'pooled fallback' if v['fallback'] else 'own threshold'} |"
        for f, v in sorted(fields.items()))
    example_note = (
        "A license-cleared CORD test receipt with its OCR file and the full output is in the "
        f"GitHub repository under [`assets/examples/cord/`]({GITHUB}/tree/main/assets/examples/cord)."
        if ds == "cord" else
        f"This model's training data is not cleared for redistribution, so no example image "
        f"is shipped. The GitHub repository has a [CORD example]({GITHUB}/tree/main/assets/examples/cord) "
        f"with the same input format, and [`assets/examples/{ds}/`]({GITHUB}/tree/main/assets/examples/{ds}) "
        f"holds this model's output on a selected test document with the text removed.")

    return f"""---
license: cc-by-nc-sa-4.0
language:
- en
library_name: transformers
pipeline_tag: token-classification
base_model: microsoft/layoutlmv3-large
base_model_relation: finetune
{yaml_ds}tags:
- document-ai
- document-understanding
- key-information-extraction
- token-classification
- conformal-prediction
- uncertainty-quantification
- calibration
- layoutlmv3
- wacv-2027
- calidoc
---

# CaliDoc: {meta['title']}

LayoutLMv3-Large fine-tuned for key information extraction on **{meta['title']}**,
plus the CaliDoc calibration artifacts fitted for this exact checkpoint. Paper:
*CaliDoc: Calibrated Field-Level Risk and Annotation Auditing for Document
Information Extraction*, WACV 2027 (Evaluations and Datasets Track).
Code: [{GITHUB}]({GITHUB}) · Supplement: [PDF]({SUPP}) · Collection:
[CaliDoc — WACV 2027]({collection})

> Token-level reliability is not complete-field reliability. CaliDoc calibrates
> the object a downstream system consumes: the complete field.

## Model Description

- **Extractor.** `LayoutLMv3ForTokenClassification`
  ({cfg['base_model']}), {num_labels} BIO labels, trained on
  {meta['about']}. This is a **dataset-specific** checkpoint, and CaliDoc does not provide a
  universal model. Each dataset has its own repository in the collection.
- **Checkpoint.** This repository contains the **seed-42 canonical checkpoint**,
  the one the paper uses for every ablation, SpanCRC and audit result. Headline WACV
  results report mean ± standard deviation over three training seeds
  (42, 123, 456). Those aggregates are not metrics of this single checkpoint.
- **Calibration (post hoc, no retraining).** Temperature scaling followed by
  field-conditional **SpanCRC**, a conformal risk control over max-token span scores.
  The neural extractor and the calibration are separate: the weights are a standard
  Transformers model, and the thresholds in `calibration/` turn its probabilities
  into calibrated label sets.

## Intended Use

- Research on uncertainty quantification, conformal prediction and annotation
  auditing for document information extraction.
- Reproducing and extending the WACV 2027 results.
- Prototyping human-in-the-loop review routing on documents **from the same
  distribution** as {meta['title']}.

## Out-of-Scope Use

- Commercial use: the weights inherit CC BY-NC-SA 4.0 from LayoutLMv3.
- Documents from other domains, layouts, languages or OCR engines. The calibration
  guarantee assumes exchangeability with the calibration documents and does not
  transfer to out-of-domain inputs.
- Autonomous decisions in consequential workflows (finance, legal, medical,
  identity). CaliDoc is meant to decide what a human should check, not to replace
  that check.
- Treating calibrated-set status as a per-field probability of correctness.

## How CaliDoc Works

1. **Input:** a document image, plus OCR words and word boxes on LayoutLMv3's
   0–1000 grid. **OCR is not included**: the paper uses the dataset-provided OCR.
2. **Extractor:** LayoutLMv3 token classification. Each word is scored by its
   first sub-word, and probabilities are `softmax(logits / T)` with T = {cfg['temperature']:.4f}.
3. **Calibration:** the calibrated set of word *i* is
   `C(x_i) = {{ y : 1 − p(y|x_i) ≤ λ_type(y) }}`. A gold field is fully covered
   exactly when its worst word clears the threshold, which makes this a max-token
   complete-field score.
4. **Reliability:** taking the document as the exchangeable unit, SpanCRC bounds
   the **expected per-document fraction of missed gold fields** by α = {cfg['alpha']}.
   The bound holds on fresh documents exchangeable with the calibration documents
   that contain at least one gold field.
5. **Audit (corpus level):** at the pooled threshold, a boundary-escape test asks
   whether threshold escapes concentrate at field edges, which is a symptom of
   alignment and annotation errors. It runs over a calibration corpus with Holm
   control and a document-blocked bootstrap. It is a dataset diagnostic, not a
   per-document error detector, and it is not part of single-document inference.

{note}

## Quick Start

```bash
pip install "calidoc @ git+{GITHUB}"
```

Plain Transformers (the extractor only):

```python
from transformers import AutoModelForTokenClassification, AutoProcessor

model_id = "MHamdan/{name}"
processor = AutoProcessor.from_pretrained(model_id, apply_ocr=False)
model = AutoModelForTokenClassification.from_pretrained(model_id)
print(model.config.id2label)
```

CaliDoc calibrated extraction, given an image and an OCR JSON
`{{"words": [...], "boxes": [[x0, y0, x1, y1], ...], "box_format": "pixel" | "normalized_1000"}}`:

```python
import json
from PIL import Image
from calidoc.inference import CaliDocPredictor, normalize_boxes

predictor = CaliDocPredictor.from_pretrained("MHamdan/{name}")
image = Image.open("document.png").convert("RGB")
ocr = json.load(open("ocr.json"))
boxes = ocr["boxes"] if ocr["box_format"] == "normalized_1000" else normalize_boxes(ocr["boxes"], *image.size)
result = predictor.predict(image, ocr["words"], boxes)
for field in result["fields"]:
    print(field["type"], "|", field["text"], "|", field["status"], field["set_sizes"])
print(result["summary"]["route"])
```

or from the command line:

```bash
python -m calidoc.predict --model MHamdan/{name} --image document.png --ocr ocr.json --out prediction.json
```

{example_note}

Each predicted field gets one of these statuses:

| status | meaning | suggested action |
|---|---|---|
| `covered` | every word's calibrated set is exactly the predicted label | pass downstream |
| `ambiguous_review` | the predicted label is in every set, but some set also admits other labels | validate / review |
| `conflict_review` | some word's set is non-empty but excludes the predicted label | review |
| `empty_abstain` | some word's set is empty | abstain / review |

`unassigned_field_candidates` lists words predicted `O` whose set still admits a
field label, which are possible missed fields. These statuses are properties of
calibrated prediction sets, not confidence scores.

## Calibration Artifacts

| file | content |
|---|---|
| `calidoc_config.json` | α = {cfg['alpha']}, m0 = {cfg['m0']}, T, pooled λ, supported/fallback field types, released-checkpoint test metrics |
| `calibration/temperature.json` | T = {cfg['temperature']:.6f}, fitted by NLL on the `{temp['fit_split']}` early-stopping split |
| `calibration/pooled_threshold.json` | pooled SpanCRC λ = {cfg['pooled_lambda']:.6f} and the token-marginal q̂ for comparison |
| `calibration/field_thresholds.json` | λ per field type with its calibration support |
| `calibration/split_manifest.json` | calibration document IDs (identifiers only, no data) |

The thresholds were fitted on {unit} (the paper's first calibration unit) and match
the committed paper artifacts exactly. Field types with fewer than m0 = {cfg['m0']}
calibration spans fall back to the pooled λ, so they inherit only the pooled
document-level bound and get no per-type guarantee.

| field type | λ | calibration spans | threshold |
|---|---|---|---|
{rows}

## Evaluation

**Paper: three training seeds** (headline, mean ± s.d. over seeds 42/123/456, full test split):

| metric | value |
|---|---|
| entity F1 | {P('F')} ± {P('FSd')} |

**Paper: CaliDoc on this seed-42 checkpoint** (90% target, α = 0.10, mean over three
calibration splits; the paper's SpanCRC table):

| | token-marginal threshold | field-conditional SpanCRC |
|---|---|---|
| realized field-miss risk (the bounded quantity) | {P('RiskTok')} | {P('RiskFld')} |
| full-span coverage (empirical) | {P('SpanCovTok')} | {P('SpanCovFld')} |
| average set size | {P('SizeTok')} | {P('SizeFld')} |

**This released checkpoint and calibration** (computed from the release artifacts):

| metric | value |
|---|---|
| test entity F1 / precision / recall | {100 * t['entity_f1']:.2f} / {100 * t['precision']:.2f} / {100 * t['recall']:.2f} ({t['n_test_docs']} test documents) |
| test ECE (after temperature) | {t['ece_percent']:.2f}% |
| SpanCRC on the evaluation side of the released calibration split: field-miss risk / span coverage / set size | {ev['field_crc_doc_risk']:.3f} / {ev['field_crc_span_coverage']:.3f} / {ev['field_crc_avg_set_size']:.2f} ({ev['n_ev_spans']} gold spans) |
| same split, token-marginal threshold: token / span coverage | {ev['token_marginal_token_coverage']:.3f} / {ev['token_marginal_span_coverage']:.3f} |

The released weights reproduce the paper pipeline's cached probabilities to within
3e-5 and the recorded test F1 exactly (see `docs/release/HF_MODEL_ARTIFACT_AUDIT.md`
in the GitHub repository).

## Limitations

- **Dataset and domain specific.** One checkpoint per benchmark. Results do not
  transfer to other document types, and the guarantee assumes exchangeability with the
  calibration documents.
- **English-script benchmarks only** (CORD receipts are Indonesian with mostly Latin-script
  menu text). One backbone family (LayoutLMv3).
- **OCR dependence.** Inputs are OCR words and boxes, and the paper uses dataset-provided
  OCR. A different OCR engine changes tokenization and boxes and voids the calibration.
- **Max-score conservatism.** The worst word decides a field, so long fields
  (FUNSD answers, SROIE addresses) need larger thresholds and larger sets.
- **Support-limited field conditioning.** Types below m0 = 25 calibration spans fall
  back to the pooled threshold and receive no per-type guarantee (most of CORD).
- **Short-span audit blind spot.** The boundary-escape test cannot see spans of two
  words or fewer, so a clean result on short fields means untested.
- **Audit independence assumption.** The randomization null assumes independence
  across spans, which the method states rather than derives. Findings are claimed only
  when a document-blocked bootstrap agrees.
- **The guarantee is in expectation**, over calibration draws and fresh documents. A
  single document, or a single calibration split, can exceed α.
- **Not a replacement for human review** in consequential workflows.

## License

- **Model weights: CC BY-NC-SA 4.0**, inherited from
  [`microsoft/layoutlmv3-large`](https://huggingface.co/microsoft/layoutlmv3-large)
  (CC BY-NC-SA 4.0). This repository fine-tunes that model for token classification and
  adds calibration files. Non-commercial use only, with attribution, under the same license.
- **CaliDoc source code:** Apache-2.0, in the [GitHub repository]({GITHUB}).
- **Training data:** {meta['data_terms']} Lineage used: {meta['lineage']}. Cite
  {meta['citation']}. No dataset images or annotations are included in this repository.

## Citation

WACV 2027 proceedings metadata (pages, DOI) will be added once published.

```bibtex
@inproceedings{{hamdan2027calidoc,
  author    = {{Hamdan, Mohammed and Cheriet, Mohamed}},
  title     = {{{{CaliDoc}}: Calibrated Field-Level Risk and Annotation Auditing for Document Information Extraction}},
  booktitle = {{Proceedings of the IEEE/CVF Winter Conference on Applications of Computer Vision (WACV)}},
  year      = {{2027}}
}}

@inproceedings{{huang2022layoutlmv3,
  author    = {{Huang, Yupan and Lv, Tengchao and Cui, Lei and Lu, Yutong and Wei, Furu}},
  title     = {{{{LayoutLMv3}}: Pre-training for Document {{AI}} with Unified Text and Image Masking}},
  booktitle = {{Proceedings of the 30th ACM International Conference on Multimedia}},
  year      = {{2022}}
}}
```
"""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--release", default="release/hf")
    ap.add_argument("--macros", default="docs/CaliDoc_WACV2027_camera_ready/results_macros.tex")
    ap.add_argument("--collection", required=True, help="Hugging Face collection URL")
    args = ap.parse_args()
    macros = dict(re.findall(r"\\newcommand\{\\(\w+)\}\{([^}]*)\}", Path(args.macros).read_text()))
    for ds in META:
        root = Path(args.release) / f"calidoc-layoutlmv3-large-{ds}"
        cfg = json.loads((root / "calidoc_config.json").read_text())
        fields = json.loads((root / "calibration" / "field_thresholds.json").read_text())["fields"]
        num_labels = len(json.loads((root / "config.json").read_text())["id2label"])
        temp = json.loads((root / "calibration" / "temperature.json").read_text())
        (root / "README.md").write_text(card(ds, cfg, fields, macros, num_labels, temp,
                                             args.collection))
        print("wrote", root / "README.md")


if __name__ == "__main__":
    main()
