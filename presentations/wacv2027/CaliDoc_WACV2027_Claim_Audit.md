# CaliDoc WACV 2027 Talk: Claim Audit

This file maps every scientific claim, number, and caveat in `CaliDoc_WACV2027_Presentation.pptx` to the accepted camera-ready sources. It covers main slides 1–14 and backup slides B1–B10, which are deck slides 16–25.

## 1. Sources of truth

| Source | File used | SHA-256 | Pages |
|---|---|---|---|
| Main paper (camera-ready) | `docs/CaliDoc_WACV2027_camera_ready/main.pdf` | `02a649c4d7f4600787cb2a6ae63082c4d6c5f75221f17e504c76e9882d114d42` | 10 |
| Same bytes, WACV upload copy | `docs/camera_ready_audit/wacv_package/CaliDoc_WACV2027_1597_main.pdf` and `docs/CaliDoc_Submission/reference_pdfs/01597_main_reference.pdf` | identical | 10 |
| Supplement (final S12 version) | `docs/CaliDoc_WACV2027_supp_s12/01597_supp.pdf` (= `docs/final_submission/01597_supp.pdf`) | `048542cc8973f6d62ed1c2e1d13d33d1a67629874e608ea280dc7baab9283ede` | 22 |
| Public supplement | GitHub release `wacv2027-camera-ready`, asset `CaliDoc_WACV2027_Supplementary.pdf`, 1,084,455 bytes | same size as the local S12 file; replaced with the S12 version on 2026-10-25 (sha `048542cc…`) | 22 |
| Real-receipt example (slide 4) | `assets/examples/sroie/sroie_test_X51005568889/prediction.json` | `06529318a55103b4ea55583c1b5eb8316abc77702da68f4e8cdb680f719093d8` | — |
| CORD example image (B9) | `assets/examples/cord/cord_test_0014/showcase.png` (CORD v2, CC BY 4.0) | `15b1e163a85c0bdd4b7abb9b958cffb356dcdd076f5450ec947e4c58a36e0069` | — |

**Not used:** `docs/camera_ready_audit/github_release/CaliDoc_WACV2027_Supplementary.pdf` (sha `66589acf…`) is the superseded 19-page supplement.

**Not found locally:** the PDF eXpress-validated file `2026401734.pdf`. The deck uses the camera-ready `main.pdf` above, which is byte-identical to the copy prepared for WACV upload. See open item O1.

Page references are to PDF pages: main paper p.1–10, supplement p.1–22.

## 2. Automated numerical verification

`source/verify_numbers.py` extracts every numeric token from the built PPTX, covering text boxes, table cells, and chart data caches. It then searches for each token verbatim in the PyMuPDF text of the main paper and supplement. Trivial tokens are skipped: 0–5, 10, 100, axis ticks, 2027, and 1597.

Result on the final deck:

```
slide 4: not found verbatim -> ['0.0079', '0.0113', '0.0289', '0.0442']
checked 25 slides
```

The four slide-4 values are the per-field worst-word scores of the real SROIE test receipt. They come from the released prediction artifact, not from the papers, and were checked against the JSON:

| Field | Words | Worst-word score (JSON) | Field threshold (JSON) | Supp. Table S13, split 1 | > 0.0056 (marginal q̂, Table 1)? |
|---|---|---|---|---|---|
| COMPANY | 4 | 0.04417 → 0.0442 | 0.14021 → 0.140 | 0.1402 | yes, missed |
| ADDRESS | 11 | 0.02887 → 0.0289 | 0.10132 → 0.101 | 0.1013 | yes, missed |
| DATE | 1 | 0.00787 → 0.0079 | 0.02243 → 0.022 | 0.0224 | yes, missed |
| TOTAL | 1 | 0.01132 → 0.0113 | 0.07654 → 0.076 | 0.0765 | yes, missed |

Every other number on every slide was found verbatim in the accepted PDFs. The chart value 97.25 on B7 is the cumulative column of Supp. Table S24.

## 3. Claim ledger

Columns: claim · exact evidence · main paper · supplement · slide · caveat shown or spoken.

| # | Claim | Evidence | Main | Supp. | Slide | Caveat |
|---|---|---|---|---|---|---|
| C1 | Token coverage on SROIE is near nominal at the 90% target | 89.8% token coverage, LAC, mean over 3 calibration splits | Sec. 1 p.1; Sec. 4.4.2 + Table 1 p.5 | Table S6 p.4 (89.8 / 0.0) | 3, 14 | — |
| C2 | No complete field is covered | 0 of 420 gold spans; full-span coverage 0.000 | Sec. 1 p.1; Sec. 4.4.2 p.5; Table 2 p.6 | Table S6 | 3, 14 | — |
| C3 | Mechanism: the threshold is sized for O | q̂ = 0.0056; O = 84% of tokens; 6/9 classes at 0%, including all four B- classes; every span starts with B- | Sec. 1, Sec. 4.4.2, Table 1 | Table S6 caption | 3 | I-COMPANY is 2.5%, not 0, so the slide says "6 of 9" |
| C4 | The extractor is competent | SROIE entity F1 94.54 ± 0.33 (3 training seeds) | Sec. 4.3 p.4 | — | 3 | — |
| C5 | Real receipt: all 4 fields missed under the marginal threshold and covered under SpanCRC | prediction.json values (Sec. 2 above) | Table 1 (q̂) | Table S13 (λ_t) | 4 | Spoken and on the slide: one illustrative receipt; text withheld because the SROIE licence is unclear |
| C6 | No retraining, no constrained decoding, no sequence search | "no sequence search, constrained decoding or retraining" | Sec. 3.3 p.3; Sec. 2.2 p.2 | — | 5 | — |
| C7 | Temperature fit on the early-stopping split | "T fitted by negative log-likelihood on the early-stopping split" | Sec. 3.1 p.2 | S1 p.1 (T values) | 5 | — |
| C8 | Max-token span score; full coverage iff S(s) ≤ λ; the max is forced | Eq. 1 and following text | Sec. 3.3 p.3 | S4.1–S4.2 | 6, B2 | Conservatism grows with span length |
| C9 | The document is the exchangeable unit; risk is the fraction of fields missed; CRC | Eqs. 2–3 | Sec. 3.3 p.3 | — | 6 | — |
| C10 | Guarantee: E[R_{n+1}(λ̂)] ≤ α on a fresh document with ≥ 1 gold field | Verbatim | Sec. 3.3 p.3 | S4 p.3 | 6 | "Not guaranteed" box; Sec. 5.2 p.8 (not per document, not per field type); empirical full-span coverage is not bounded |
| C11 | Candidate-label rule; calibrated and deployed rules coincide | Verbatim | Sec. 3.3 p.3 | S4 p.3 | 7 | — |
| C12 | m0 = 25 fallback to the pooled threshold, mirroring clustered CP | Verbatim | Sec. 3.3 p.3 | S4 | 7 | Fallback types inherit only the pooled bound; aggregate full-span coverage under the mixed rule is empirical (Sec. 3.3 "Two caveats") |
| C13 | SROIE: all 4 types supported, 242 calibration spans each | n_cal 242 per field | — | Table S13 p.7 | 7 | — |
| C14 | CORD support chart (24 field types, 9 ≥ 25) | n_cal per field | — | Table S14 p.8 | 7 | See D1 |
| C15 | 19.3% of CORD evaluation spans are in fallback types | Verbatim | — | S4.3 p.5 | 7 | — |
| C16 | SROIE risk 1.000 → 0.091 | Table 2 | Table 2 p.6; Sec. 4.5 p.5 | — | 8 | Realized risk; the theorem bounds its expectation |
| C17 | SROIE full-span coverage 0.000 → 0.909 ± 0.011; CovGap 90.0 → 2.1 pp | Table 2 | Table 2, Fig. 1(a) | — | 8 | Empirical |
| C18 | SROIE set size +10% | "+10% set-size increase" | Sec. 4.5 p.5 | Fig. S1 p.5; Table S3 (0.90 → 0.99) | 8 | — |
| C19 | SROIE-rep. 0.749 → 0.110; 0.257 → 0.894 ± 0.020; 64.2 → 2.9; +11% | Table 2; Fig. S1 | Table 2 | Fig. S1 | 8 | Separate annotation lineage |
| C20 | WildReceipt 0.093 → 0.100; 0.918 → 0.913 ± 0.004; 13.5 → 8.8; +1% | Table 2; Fig. S1 | Table 2 | Fig. S1 | 8 | Spoken: coverage dips slightly |
| C21 | FUNSD 0.054 → 0.058; 0.950 → 0.944 ± 0.005; 6.4 → 5.0; +23% | Table 2; Fig. S1 | Table 2; Sec. 4.5 | Fig. S1 | 8 | Spoken: coverage dips slightly; no universal improvement |
| C22 | CORD 0.115 → 0.108; 0.872 → 0.886; 42.3 → 33.3 pp; +224% (0.90 → 2.92, 3.2×) | Table 2; Sec. 4.5 | Table 2; Sec. 4.5 p.5; Sec. 6 | Fig. S1; Table S12 | 8, 9, 13 | Weakest case, stated explicitly |
| C23 | CORD realized risk 0.108 > α on a single split, consistent with an expectation bound | Verbatim | Sec. 4.5 p.5 | S4 p.3 | 9, 13 | — |
| C24 | 100% of CORD tokens carry multi-label sets (p90 size 3) | Verbatim | Sec. 5.2 p.8 | S8.1 p.10 | 9, B8 | — |
| C25 | MENU.SUB.CNT λ̂ = 1.0 | Table S14 (1.0000); S12 prose | — | Table S14 p.8; S12 p.20 | 9 | — |
| C26 | Level-calibrated interpolation τ = 0.5: risk 0.035, CovGap 18.4, set size 1.12, 0% empty | Table S12 | Sec. 4.5 p.5 | Prop. 3, Table S12 pp.4–5 | 9 | Pooled bound only, not field-conditional; over-covers (μ̂ ≈ 0.92); FUNSD inflates to 4.69 |
| C27 | Escape, informative span, edge statistic W_t, conditional-randomization null, B = 5,000 | Eq. 4, Sec. 3.4 | Sec. 3.4 pp.3–4 | S5 | 10 | — |
| C28 | Holm controls FWER across fields under arbitrary dependence; it does not repair raw-p validity; the bootstrap corroborates but does not restore validity | Verbatim | Sec. 3.4 p.4 | S5 | 10, 13 | Cross-span independence is an assumption |
| C29 | ≤ 2-word spans are a structural blind spot; ineligible fields are untested, not clean | Verbatim | Sec. 3.4 p.4; Sec. 6 | S5 | 10, 13 | — |
| C30 | ADDRESS enrichment 4.07, COMPANY 1.80, Holm p 0.0008, at the MC floor | Sec. 4.6 | Sec. 4.6 p.6; Fig. 1(b) | Table S17 p.11 | 11 | — |
| C31 | Bootstrap intervals [3.37, 4.91] and [1.66, 1.94] | Verbatim | Sec. 4.6 p.6 | Table S17 | 11 | Independence assumed only across documents |
| C32 | 0 of 16 eligible clean-benchmark fields rejected; Holm, BH and BY agree on all 3 splits | Verbatim | Sec. 4.6 p.6 | Tables S19–S20 | 11 | Eligibility: untested ≠ clean |
| C33 | CORD SUB_TOTAL.SERVICE_PRICE enrichment 2.00, 2 informative spans, p = 0.2549 | Table S18 | Fig. 1(b) | Table S18 p.12 | 11 | — |
| C34 | Other Fig. 1(b) bars: FUNSD QUESTION 1.28, SROIE-rep. COMPANY 1.15, WildReceipt Prod_item_value 1.00 | Tables S17–S18 | Fig. 1(b) p.6 | Tables S17–S18 | 11 | — |
| C35 | Modal prediction over ADDRESS escapes is B-ADDRESS at span edges | Verbatim | Sec. 4.6 p.6 | — | 11 (spoken) | — |
| C36 | Repair: ADDRESS 0.79, p 0.8666, CI [0.41, 1.10]; COMPANY 1.15, p 0.2204, CI [0.93, 1.38]; nothing rejected | Verbatim / Table S17 | Sec. 4.6, Sec. 4.9 p.7 | Table S17 | 11 | Separate lineage; the repair is re-alignment, not calibration |
| C37 | Rombach & Mehdiyev: first-token score, marginal coverage over entity instances, WildReceipt, per-field calibration as future work | Sec. 2.2 | Sec. 2.2 p.2 | — | 12 | Cited as the closest prior work; no "first" claim |
| C38 | Their operating point: 98.3%, 1.38, 70%; ours 0.980 ± 0.002, 1.16, 88.8% | Verbatim | Sec. 4.4.1 pp.4–5 | Table S9 p.6 | 12 | "Stated, not ranked" on slide and in narration |
| C39 | SROIE α = 0.10: first-token 0.897 vs full-span 0.856 | Verbatim | Sec. 4.4.1 p.5 | Table S9 | 12 | Proxy gap depends on dataset and operating point |
| C40 | Six limitations | Sec. 6 "Limitations" (five items) + Sec. 3.4 dependence + S2.1 Base scope | Sec. 6 p.8 | S2.1 | 13 | Reproduces the paper's list without adding or softening |
| C41 | NSERC Discovery Grant RGPIN-2025-05449 | Acknowledgment | p.8 | — | 14 | — |
| B1 | Table 1 values; Mondrian CovGap 71.1 → 3.4 pp; set size 0.90; quantile factor 367 (0.0027 → 0.9996) | Verbatim | Table 1, Sec. 4.4.2 p.5 | — | B1 | Screening signal, not evidence |
| B2 | Own-event vs full-span (0.886/0.886, 0.898/0.762, 0.886/0.698, 0.893/0.840); risks 0.114/0.238/0.302/0.160; Spearman 0.59 vs 0.04 | Table S11; Sec. 4.5 | Sec. 4.5 p.5 | Tables S10–S11 p.6 | B2 | Diagnostic ablation |
| B3 | Matched-n 0.810/0.910/0.954/0.922/0.905; CovGap 45.8/45.9/32.2/16.4; set size 0.90 → 2.49; bound n ≥ 8; 24 classes below the bound | Verbatim | Sec. 4.7 p.7 | Tables S28–S29 | B3 | — |
| B4 | Size (Holm) 0.025 / 0.000 / 0.000; uncorrected 0.140 / 0.000 / 0.045; power 0.53 / 0.73 / 0.93; start-shift ≤ 0.20; FUNSD QUESTION raw p 0.0362, Holm 0.1086; 36 (field, mask) pairs | Verbatim | Sec. 4.6 pp.6–7 | Tables S16, S21–S23 | B4 | Uncorrected WildReceipt/CORD values from Table S21 |
| B5 | AUPRC 0.007–0.169 vs 0.047–0.951 (Cleanlab) and 0.117–0.859 (AUM); ranks 6th–12th vs 21st–46th; 48th, 38th | Verbatim | Sec. 4.8 p.7 | Table S26 p.17 | B5 | "Not claimed to outperform" |
| B6 | Base and clean-cal rows | Table S3 | Sec. 4.5 p.5 | S2.1, Table S3 pp.2–4 | B6 | Two scales of one family |
| B7 | Gate 94.68% (2,369/2,502), 94.02% (1,305/1,388), pooled 94.45%; +2.80 → 97.25 → 97.38%; 0.10% / 2.52%; address 85.6% vs 95.1–99.1%; B-/I-ADDRESS 0.000 → 0.444 / 0.959 → 0.261; CovGap 71.1 → 52.7 | Verbatim | Sec. 4.9 p.7 | S6 p.10; Tables S24–S25 p.16 | B7 | The repair does not retroactively pass the gate |
| B8 | α reading, 0.4 expected misses per receipt, review burden; 99.0% / 92.9% singleton; 6.3% empty | Verbatim | Sec. 5.2 p.8 | S8.1 p.10 | B8 | — |
| B9 | cord_test_0014: 25 fields, 24 correct, 25 review, 1 conflict; 0.80 vs 0.189 | S12 prose and `usecase_macros.tex` | — | S12 p.20 | B9 | "One illustrative case, not a sensitivity result" |
| B10 | ECE 1.27 ± 0.22 / 1.81 ± 0.08 / 6.37 ± 0.76; F1 drop 0.56 pp; coverage drop < 0.15 pp (worst 1.53); MC Dropout 1.57 → 2.77, 87.9 → 78.6, 20×; 9.9% empty, 68.2% | Verbatim | Sec. 4.4 p.4; Sec. 4.10 p.7 | S8–S9 | B10 | Coverage under perturbation is empirical |

## 4. Guarantee-language audit

These searches were run over the slide text and the notes:

- **"guarantee"**: used only for (a) the expected per-document field-miss bound (slide 6, takeaways), (b) the pooled bound of level-calibrated interpolation (slide 9), and (c) negative statements ("not guaranteed", "a guarantee we do not have").
- **"coverage" next to "guarantee"**: never asserted as guaranteed. Slide 6 explicitly lists empirical full-span coverage under "Not guaranteed".
- **"first" / "state of the art" / "outperform"**: none claimed. B5 states "not claimed to outperform"; slide 12 notes say "conformal KIE isn't new, and we don't claim it is".
- **Per-field-type guarantee**: slide 7 limits per-type bounds to supported types and says that fallback types inherit only the pooled bound.
- **The audit**: always described as a positional test under a stated null. It is never called a general label-error detector (slide 10, B5), and the repair is credited to re-alignment, not calibration (slide 11).

## 5. Discrepancies and caveats found during the audit

- **D1 (CORD field-type count).** Supp. Table S14 lists 24 CORD field types for the first calibration split: 9 supported and 15 below m0. Supp. S4.3 prose says "16 of 25 CORD field types sit below m0 = 25". The slides use the main-paper wording ("most field types") and the Table S14 count ("9 of 24 listed"). If asked, say that S4.3 counts 25 types across the schema while Table S14 lists the 24 present in the first split. That is the likely reading, but it is not verified; see open item O2.
- **D2 (mixed thresholds on slide 4).** The marginal threshold 0.0056 is the Table 1 mean over three calibration splits. The field thresholds are the released first-split values. Every worst-word score exceeds 0.0056 by at least 40%, so the field verdicts do not depend on the split. The slide labels the example as an illustrative single document.
- **D3 (WildReceipt set size).** Table S3 shows 0.93 → 0.93 at two decimals; Fig. S1 annotates +1%. The slide uses Fig. S1.
- **D4 (Fig. 1(a) labels).** The paper's Fig. 1(a) prints only four bar labels. The slide-8 chart shows all ten bars with values taken from Table 2.

## 6. Visual provenance

| Visual | Slide | Status |
|---|---|---|
| Receipt with field boxes; 11-chip ADDRESS strip | 2 | Conceptual, labelled "Conceptual illustration" |
| 100-cell token waffle and 420-cell field waffle | 3 | Faithful encoding of 89.8% and 0/420 (one cell ≈ 1% of tokens; one outline = one gold field) |
| Worst-word dot plot | 4 | Real data from the released checkpoint (log axis; values printed) |
| Framework diagram | 5 | Schematic of Sec. 3 |
| Bar diagram for the max score | 6 | Conceptual, labelled "Conceptual bars" |
| CORD support chart | 7 | Native chart, Supp. Table S14 |
| Full-span coverage chart | 8 | Native chart, Table 2 |
| Escape-pattern spans | 10 | Conceptual, labelled "Conceptual spans" |
| Enrichment chart and repair dumbbell | 11 | Native chart (Fig. 1(b) values); dumbbell from Table S17 values and CIs |
| B1–B7 charts | B1–B7 | Native charts from the cited tables |
| CORD receipt routing figure | B9 | Released artifact `showcase.png`; CORD v2 is CC BY 4.0; personal information is blurred by the dataset authors |

No SROIE, FUNSD, or WildReceipt images appear anywhere in the deck.

## 7. Open items needing the author's decision

- **O1.** Confirm that the PDF eXpress-validated `2026401734.pdf` is content-identical to `main.pdf` (sha `02a649c4…`). If it differs, re-run `source/verify_numbers.py` against it.
- **O2.** Choose the CORD wording if you prefer the S4.3 count: "16 of 25 types fall back" versus Table S14's 9 of 24 supported. The current slides avoid stating "16 of 25".
