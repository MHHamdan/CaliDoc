# Hugging Face model artifact audit

Date: 2026-10-17. Scope: the four CaliDoc model repositories published under
`MHamdan/` for the WACV 2027 release. Nothing was retrained, no checkpoint was
re-selected, and no threshold was re-fitted. Every number below was produced by
the tools named in this document.

## 1. Which checkpoints the camera-ready paper uses

Every camera-ready number comes from the local 2026-08-03/04 lineage:

- The camera-ready macros are emitted by `scripts/emit_result_macros.py`, which reads
  `results/local_20260803/` (F1 from `multiseed_summary.json`, SpanCRC from
  `span_crc_<ds>.json`, temperatures from `results/cache/<ds>_local_20260803.json`).
- The result JSONs record `meta.model_path`. For seed 42 this is
  `outputs/<ds>_retrain_s42_local_20260803/best` for all four datasets.
- No number from the earlier cluster lineage remains in the camera-ready paper.

**Canonical seed policy.** The paper fixes seed 42 in advance:

- `main.tex` (Implementation Details): "Headline metrics average over training seeds
  {42,123,456}; ablations use the seed-42 checkpoint".
- The supplement repeats this, and supplementary §S2 calls it "the committed seed-42
  lineage".
- SpanCRC, the boundary-escape audit, Mondrian calibration and the temperatures are all
  computed from the seed-42 probability caches.

Seed 42 is therefore released as the default. It was **not** chosen for performance. On
CORD it is the low tail of the three seeds (test F1 93.44 / 95.70 / 96.29).

**`best/`, not `final/`.** `scripts/run_local_phase2_queue.sh` sets
`CK=outputs/${DS}_retrain_s42_local_20260803/best`, and the trainer writes `best/` when
early-stopping-split F1 improves. `final/` holds different weights and is not used.

## 2. Inventory

| dataset | source checkpoint | source `pytorch_model.bin` sha256 | released `model.safetensors` sha256 | tensors (all bit-equal) |
|---|---|---|---|---|
| SROIE | `outputs/sroie_retrain_s42_local_20260803/best` | `2412b460dc6c574b558d52cb63bcfe9146d4f68234eeabda66baf031006ac5dd` | `96fee07e06edbf0a3a35175b19eaf9613ab0be6904986376ab355ec680b0b654` | 406 / 406 |
| CORD | `outputs/cord_retrain_s42_local_20260803/best` | `4969ba728c2ed748e257fa142588e18ddccafec495a70863d994f2f3dedcba8d` | `2afd73f698f4d48dfa5cc9c7aea4d9545de42195bde8f09492fe22b3017e5517` | 408 / 408 |
| FUNSD | `outputs/funsd_retrain_s42_local_20260803/best` | `de7873b649b9d0e578b4f528388c404cb5169272610aa4278b65f4b89b977f9a` | `eb7265f1be7626212ec31be4f69b82382df4e1a69d651fda07c894169d2116a0` | 406 / 406 |
| WildReceipt | `outputs/wildreceipt_retrain_s42_local_20260803/best` | `aadde05889d504eeddb9c2b4d17be03c6ee7fb5a0db1cd342a928b191d050edb` | `869bc4f949d2f963a18ae249b94cb2511ddec4c89b70261045b03bf3d1e5397b` | 408 / 408 |

The source state dicts hold fp32 tensors only. Their keys carry the CaliDoc wrapper
prefix `encoder.model.`. The prefix was stripped and the tensors were loaded **strictly**
(no missing or unexpected keys) into a stock `LayoutLMv3ForTokenClassification`. The
weights were written as safetensors and re-read, and every tensor was asserted
`torch.equal` to the source. Optimizer state (`training_state.pt`) is not released.

| | SROIE | CORD | FUNSD | WildReceipt |
|---|---|---|---|---|
| architecture | LayoutLMv3ForTokenClassification | same | same | same |
| pretrained parent | microsoft/layoutlmv3-large | same | same | same |
| training seed | 42 | 42 | 42 | 42 |
| labels | 9 (O at index 8) | 54 (no O) | 7 (O at index 0) | 49 (O at index 48) |
| epochs / lr / effective batch | 30 / 2e-5 / 32 | 30 / 2e-5 / 32 | 120 / 5e-5 / 32 | 30 / 2e-5 / 32 |
| early stopping (patience, split) | 5, val_es (60) | 5, val_es (50) | 15, es_holdout (15) | 5, val_es (100) |
| LLRD / label smoothing | 1.0 / 0 | 1.0 / 0 | 0.9 / 0.1 | 1.0 / 0 |
| training config | `configs/training/sroie_phase2.yaml` | `cord_phase2.yaml` | `funsd_phase2.yaml` | `wildreceipt_phase2.yaml` |

- **Labels.** `id2label`/`label2id` are taken from `data_bundle/<ds>/processed/label_list.json`.
  The tool asserts that this list equals the `id2label` saved in the checkpoint's
  `training_config.json`.
- **Processor.** The evaluation pipeline encodes inputs with the
  `microsoft/layoutlmv3-base` processor (`calidoc/data/bundle.py`) and `apply_ocr=False`.
  Its vocabulary, merges and image-processor config are byte-identical to -large; only
  `name_or_path` differs. The release ships exactly the processor that was used.
- **Saved-config caveat.** The checkpoint's `training_config.json` contains
  `"apply_augmentation": true`. That is the unused default of the `TrainingConfig`
  dataclass. The flag that takes effect is the YAML `robustness.augmentation.enabled: false`,
  so augmentation was off.

## 3. Calibration artifacts

`tools/export_hf_release.py` recomputes the SpanCRC thresholds from the paper's cached
temperature-scaled probabilities (α = 0.10, m0 = 25, CRC with B = 1, document as the
exchangeable unit). It uses the same functions that produced the paper (`scripts/span_crc.py`).

- **Calibration unit.** The tool uses the paper's first calibration unit: calibration split
  seed 42 for FUNSD, WildReceipt and SROIE, and the fixed 50-receipt validation split for CORD.
- **Equality check.** The tool **asserts equality** with the committed
  `results/local_20260803/span_crc_<ds>.json`: pooled λ, token q̂, and the λ and fallback
  flag of every evaluated field.
- **Fields outside the committed JSON.** Fields present only on the calibration side are
  absent from the committed JSON. Their λ is computed by the same rule, and fields that
  never occur in calibration get the pooled λ, exactly as the deployed rule does.
- **Naming caveat.** The paper's `seed_s42/s123/s456` SpanCRC units are *calibration-split*
  seeds applied to the seed-42 model, not training seeds.

| dataset | T (fit split) | pooled λ | token q̂ | own-threshold fields | fallback fields | calibration docs |
|---|---|---|---|---|---|---|
| SROIE | 1.591272 (val_es) | 0.076803 | 0.005438 | 4 | 0 | 242 (test-side cal, seed 42) |
| CORD | 0.894607 (val_es) | 0.189033 | 0.109659 | 9 | 20 (incl. types absent from calibration) | 50 (validation) |
| FUNSD | 1.259807 (es_holdout) | 0.951165 | 0.955434 | 3 | 0 | 35 (test-side cal, seed 42) |
| WildReceipt | 1.114367 (val_es) | 0.147488 | 0.171850 | 22 | 2 | 330 (test-side cal, seed 42) |

- **Files per repository:** `calidoc_config.json`, `calibration/{temperature,pooled_threshold,field_thresholds,split_manifest}.json`.
- **`split_manifest.json`** lists calibration document IDs only.
- **CORD count.** The paper counts 9 own / 16 fallback over the 25 field types present in
  CORD's evaluation side. The release lists all 29 types, so it reports 20 fallback.
- **Calibration documents.** For FUNSD, WildReceipt and SROIE the calibration documents
  are a document-level 70/30 split of the *test* set. This is valid split conformal
  because no test document influenced training, early stopping, temperature or model
  selection. A consequence: the released checkpoint's **test F1 is computed over the full
  test split, which includes these calibration documents**.

## 4. Parity with the paper pipeline

| dataset | released-model test F1 | recorded test F1 (`phase2_<ds>.json`) | max \|Δp\| vs cache, full test | words whose calibrated set differs |
|---|---|---|---|---|
| SROIE | 0.949249 | 0.949249 | 1.7e-5 | 2 / 40,401 |
| CORD | 0.934445 | 0.934445 | 2.9e-5 | 0 |
| FUNSD | 0.827038 | 0.827038 | 2.4e-5 | 1 |
| WildReceipt | 0.940379 | 0.940379 | 1.4e-5 | 13 |

Inputs were the full test split. Each `calidoc.inference.CaliDocPredictor`, loaded from
the export directory, was compared with `results/cache/<ds>_local_20260803.npz` (the
probabilities the paper used).

- **F1** is identical to every printed digit.
- **Set differences.** The 16 words whose set differs (out of roughly 100k) have a
  probability within float noise (~1e-5) of a threshold. They come from batch-1 versus
  batch-8 inference.
- **Attention.** The predictor loads the model with `attn_implementation="eager"`, as the
  original pipeline did (it requested attention maps).

## 5. Publication gates

Gate status per model:

- [x] exact checkpoint provenance established (§1)
- [x] seed declared (42, predeclared canonical)
- [x] label mapping verified against the bundle and the checkpoint
- [x] processor and tokenizer tested (§4)
- [x] inference parity before upload (§4), and after upload from the Hub (see `PUBLIC_RELEASE_SMOKE_TEST.md`)
- [x] calibration artifacts verified against the committed paper JSON (§3)
- [x] no dataset images or annotations in the weight repositories (only `*.json`, `model.safetensors`, `README.md`)
- [x] no secrets
- [x] weight license CC BY-NC-SA 4.0 (see `DATASET_LICENSE_AUDIT.md`)
- [x] base model attribution and dataset attribution in each card
- [x] cards separate the paper's three-seed mean from this checkpoint's metrics

## 6. Reproduce

```bash
python tools/export_hf_release.py --out release/hf          # weights + calibration, with all asserts
python tools/write_model_cards.py --release release/hf --collection <collection URL>
```

Inputs (`outputs/`, `results/`, `data_bundle/`) are local research artifacts that are
not under version control.
