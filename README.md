# CaliDoc

Field-level uncertainty quantification and annotation auditing for document
information extraction.

Token-level conformal prediction certifies the wrong object: a practitioner
consumes a complete field, and marginal coverage over tokens says nothing about
whether every token of a delivered BIO span is covered. CaliDoc makes the
complete field the unit of reliability, and reuses the resulting calibrated
threshold as an annotation diagnostic.

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

Python 3.11+, PyTorch 2.12. Pinned versions in `requirements.txt`;
analysis-only extras in `requirements-analysis.txt`.

## Layout

```
calidoc/          library: data, models, training, evaluation, utils
scripts/          experiment entry points, analysis, artifact emitters
configs/          YAML configs per dataset and training strategy
slurm/            cluster job scripts
tests/            unit tests
```

Experiment outputs, dataset bundles, checkpoints, and result artifacts are
written outside version control (see `.gitignore`).

## Reproducing an experiment

```bash
python scripts/train.py --config configs/training/cord_phase2.yaml
python scripts/cache_probs.py --dataset cord
python scripts/span_crc.py --dataset cord
```

Seeds, split definitions, and determinism settings are recorded per run in the
emitted result JSON.

## License

Apache-2.0. See `setup.py`.
