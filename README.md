# CaliDoc

**Calibrated Field-Level Risk and Annotation Auditing for Document Information Extraction**
Mohammed Hamdan and Mohamed Cheriet — Synchromedia Laboratory, École de technologie supérieure (ÉTS), Montréal.

**Accepted at WACV 2027** (IEEE/CVF Winter Conference on Applications of Computer Vision, Evaluations and Datasets Track).

| Resource | Link |
|---|---|
| Paper | camera-ready version; proceedings link to follow |
| Supplementary material | [CaliDoc_WACV2027_Supplementary.pdf](https://github.com/MHHamdan/CaliDoc/releases/download/wacv2027-camera-ready/CaliDoc_WACV2027_Supplementary.pdf) (release `wacv2027-camera-ready`) |
| Code | this repository (reproduction path below) |

Token-level conformal prediction certifies the wrong object: a practitioner
consumes a complete field, and marginal coverage over tokens says nothing about
whether every token of a delivered BIO span is covered. CaliDoc makes the
complete field the unit of reliability, and reuses the resulting calibrated
threshold as an annotation diagnostic.

## CaliDoc at a Glance

![CaliDoc overview: a fixed LayoutLMv3 extractor feeds max-token span scores into document-level conformal risk control (SpanCRC); the same pooled threshold drives a boundary-escape annotation audit with a conditional randomization null, Holm control, and document-blocked bootstrap corroboration, validated by independent re-alignment and repair on SROIE.](assets/architecture/calidoc_architecture.svg)

Conformal prediction for document information extraction is usually calibrated
and evaluated on tokens, or on one representative token per entity. Nominal
coverage at that level does not certify the object a downstream system
consumes: a field is recovered only if every token of its span is covered, so
token-marginal coverage of 1 − α implies nothing about full-span coverage. On
SROIE, a LayoutLMv3-Large tagger attains 89.8% token coverage at a 90% target
while none of the 420 gold field spans is fully covered, because the single
marginal threshold is sized for the background class that holds 84% of tokens.
CaliDoc therefore shifts the unit of reliability from the token to the complete
field. Each gold span *s* is scored by its worst token,
*S(s) = max<sub>i∈s</sub> (1 − p(g<sub>i</sub> | x<sub>i</sub>))*, so that a
span is fully covered at threshold λ exactly when *S(s) ≤ λ*; full-span
coverage becomes a one-dimensional conformal problem with no retraining,
sequence search, or constrained decoding. Because spans within a page are not
exchangeable, the document is taken as the exchangeable unit and the risk
is the per-document fraction of missed gold fields, *R<sub>j</sub>(λ)*.
Conformal risk control over this monotone loss (SpanCRC) selects a threshold
λ̂ with the finite-sample guarantee *E[R<sub>n+1</sub>(λ̂)] ≤ α* on a fresh
document containing at least one gold field. Field types with at least
*m<sub>0</sub> = 25* calibration spans receive their own threshold λ̂<sub>t</sub>
and the corresponding per-type bound; lower-support types fall back to the
pooled threshold and inherit only the pooled document-level bound, and the
aggregate full-span coverage reported under this mixed rule is an empirical
quantity, not the guarantee.

The same calibrated threshold supports an annotation audit. At the pooled
reference threshold τ = λ̂, the tokens of a span whose gold-label score exceeds
τ form its escape set; for spans in which some but not all tokens escape, the
boundary-escape statistic *W<sub>t</sub>* counts escapes landing on the two
span edges, where alignment and BIO annotation errors tend to appear. Its null
is a conditional randomization null (escape positions uniform within each span
given the observed span length and escape count, independently across spans),
evaluated by Monte-Carlo draws, with Holm familywise-error control across field
types and a document-blocked bootstrap of the enrichment ratio as
corroboration. The cross-span independence assumption is stated rather than
derived, and the bootstrap corroborates under weaker assumptions without
repairing the validity of the raw randomization p-value; a finding is claimed
only when both agree. On SROIE the audit flags ADDRESS and COMPANY, an
independent re-alignment of the official Task-3 strings reaches only 85.6%
strict match on ADDRESS, and the positional signal disappears after a repaired
labeling pipeline is run as a separate lineage. The method is support-limited
(most CORD field types fall back to the pooled threshold at a set-size cost),
blind to spans of two tokens or fewer, validated on one real defective corpus,
and evaluated on four English form and receipt datasets with one backbone
family.

## What is here

- **SpanCRC** — conformal risk control over field-conditional max-token span
  scores. Treats the document as the exchangeable unit and bounds the expected
  per-document fraction of missed fields. The extractor is never retrained; the
  method fits one scalar threshold per field type on cached probabilities.
- **Boundary-escape test** — asks whether threshold escapes concentrate at field
  edges, under an explicit conditional-randomization null with Holm familywise
  error control and a document-blocked cluster bootstrap.
- **Conformal baselines** — LAC, APS, RAPS, Mondrian (class-conditional),
  clustered CP, long-tailed interpolation, cross-conformal validation.
- **Label-error baselines** — Cleanlab token classification, AUM, data
  cartography, inductive-CP p-values, max-softmax, under planted noise.
- **Robustness grids** — 9 image perturbations x 3 severities with thresholds
  held at clean values.

Backbone is LayoutLMv3 (Base and Large) under a token-classification head.
Datasets: FUNSD, CORD, WildReceipt, SROIE.

## Install

```bash
pip install -e .
```

Python 3.10+ (the reported runs used 3.11 and PyTorch 2.9.1). Full training
dependencies are in `requirements.txt`; the post-hoc analysis path, which needs
neither a GPU nor PyTorch, only needs `requirements-analysis.txt`.

## Layout

```
calidoc/              library: data, models, training, evaluation, utils
scripts/              experiment entry points, analysis, artifact emitters
configs/              Hydra configs per dataset, model and training strategy
tests/                unit tests
assets/architecture/  architecture diagram (SVG) and its editable source
```

Experiment outputs, dataset bundles, checkpoints, result artifacts and cluster
job scripts are kept outside version control (see `.gitignore`).

## Reproducing an experiment

Training, probability caching, and field-level calibration for one dataset:

```bash
python scripts/train.py training=cord_phase2 dataset=cord
python scripts/cache_probs.py --dataset cord
python scripts/span_crc.py --dataset cord
```

Every table and figure in the paper is computed post hoc from the cached,
temperature-scaled probabilities, on CPU:

```bash
python scripts/span_crc.py --all
python scripts/multiplicity_analysis.py
python scripts/spancrc_sensitivity.py
python scripts/boundary_escape_power.py
python scripts/first_token_entity_cp.py
python scripts/emit_result_macros.py
```

Seeds, split definitions, and determinism settings are recorded per run in the
emitted result JSON.

## Citation

Accepted at WACV 2027 (Evaluations and Datasets Track). Final proceedings
metadata (pages, DOI) will be added once published.

```bibtex
@inproceedings{hamdan2027calidoc,
  author    = {Hamdan, Mohammed and Cheriet, Mohamed},
  title     = {{CaliDoc}: Calibrated Field-Level Risk and Annotation Auditing for Document Information Extraction},
  booktitle = {Proceedings of the IEEE/CVF Winter Conference on Applications of Computer Vision (WACV)},
  year      = {2027},
  note      = {Accepted at WACV 2027 (Evaluations and Datasets Track)}
}
```

## License

Apache-2.0. See `setup.py`.
