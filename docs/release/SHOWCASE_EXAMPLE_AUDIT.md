# Showcase example audit

All examples in `assets/examples/` were produced by `tools/showcase_examples.py`,
running the **published** Hugging Face checkpoints (`--model_root MHamdan/`) on
benchmark test documents. Figures were drawn by `tools/render_showcase_examples.py`
from the resulting `prediction.json`. No prediction, threshold or status was typed
or edited by hand.

## Selection rule

The rule was fixed in code before any prediction was inspected.

1. **Eligible documents** are those never used to fit a threshold:
   - the CORD test split, since CORD calibrates on 50 validation receipts;
   - for FUNSD, WildReceipt and SROIE, the `evaluation` list of `data_bundle/splits/<ds>_calib_seed42.json`.
2. Sort eligible documents by (word count, doc_id).
3. **Representative:** the lower median, at index (n − 1) // 2.
4. **Diagnostic (CORD only):** the 90th-percentile word count, at index ⌊0.9 (n − 1)⌋. This
   gives a dense receipt that exercises CORD's long-tail schema and pooled fallback.

No example was chosen for the quality of its prediction.

## Examples

| dataset | doc id | split | role | rank / eligible | words | model | redistribution | output |
|---|---|---|---|---|---|---|---|---|
| CORD | `cord_test_0042` | test | representative | 49 / 100 | 20 | MHamdan/calidoc-layoutlmv3-large-cord | cleared (CC BY 4.0) | `assets/examples/cord/cord_test_0042/` (image, OCR, prediction, figure) |
| CORD | `cord_test_0014` | test | diagnostic | 89 / 100 | 40 | MHamdan/calidoc-layoutlmv3-large-cord | cleared (CC BY 4.0) | `assets/examples/cord/cord_test_0014/` |
| SROIE | `sroie_test_X51005568889` | test (evaluation side, cal seed 42) | representative | see JSON | see JSON | MHamdan/calidoc-layoutlmv3-large-sroie | not cleared, structural output only | `assets/examples/sroie/sroie_test_X51005568889/prediction.json` |
| FUNSD | `funsd_test_87093315_87093318` | test (evaluation side) | representative | see JSON | see JSON | MHamdan/calidoc-layoutlmv3-large-funsd | not cleared, structural output only | `assets/examples/funsd/funsd_test_87093315_87093318/prediction.json` |
| WildReceipt | `wildreceipt_test_32abb05dc239b237b39b094a784e23cd7f251dbb` | test (evaluation side) | representative | see JSON | see JSON | MHamdan/calidoc-layoutlmv3-large-wildreceipt | not cleared, structural output only | `assets/examples/wildreceipt/wildreceipt_test_32abb05dc239b237b39b094a784e23cd7f251dbb/prediction.json` |

- **OCR.** Every example uses the dataset-provided OCR words and boxes. No OCR engine is
  run.
- **CORD attribution.** CORD v2 (Park et al., 2019), CC BY 4.0,
  `naver-clova-ix/cord-v2` @ `7f0115a4b758a71d6473b8d085751692da2fef98`. Test-split indices
  are recorded in each JSON as `attribution.source_index_in_split`.

## What the examples show (from the JSON)

- **`cord_test_0042`.**
  - All 9 predicted fields match gold exactly, and all 9 gold fields are covered by their
    calibrated sets.
  - All 9 fields are `ambiguous_review`. `MENU.SUB.CNT` has 31 calibration spans and its own
    threshold, but that threshold is λ = 1.0, so its B/I labels enter every word's set.
  - This is the CORD set-size cost the paper reports. It is not an extraction error.
- **`cord_test_0014`.**
  - 25 fields predicted: 24 `ambiguous_review` and 1 `conflict_review`.
  - The conflict is *Coupon 100,000* (words 34–36). The model predicted `SUB_TOTAL.ETC`;
    gold is `TOTAL.TOTAL_ETC`. Its worst-word score is 0.80, against the pooled fallback
    λ = 0.189.
  - It is the only prediction that differs from gold, and the only gold field (1 of 25)
    not covered by the calibrated sets.
- **SROIE.** 4 fields, all `covered` (singleton sets); the document routes to `accept`.
- **FUNSD.** 100 fields: 15 `covered`, 85 `ambiguous_review`. The large thresholds
  (pooled λ = 0.951) come from long form fields and a 35-form calibration set.
- **WildReceipt.** 17 fields: 16 `covered` and 1 `empty_abstain`.

## Checklist (every example)

- [x] real benchmark document; exact document ID recorded
- [x] exact model repository recorded (the published Hub repo)
- [x] deterministic selection rule, declared in code and here; not chosen on prediction quality
- [x] inference generated programmatically; no manual alteration
- [x] ground truth from the same bundle lineage the model was evaluated on (CORD only; withheld elsewhere)
- [x] license permits redistribution (CORD); text, boxes and images withheld for the others
- [x] attribution shown in each figure and JSON
- [x] no privacy-sensitive material beyond what CORD distributes (the dataset authors blurred personal data)
- [x] figures and JSON agree: the renderer reads only `prediction.json`
