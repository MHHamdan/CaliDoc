#!/usr/bin/env python3
"""Package the canonical seed-42 CaliDoc checkpoints as Hugging Face repos.

For each dataset this tool

1. loads the committed checkpoint `outputs/<ds>_retrain_s42_local_20260803/best`
   (the one every camera-ready SpanCRC/audit number was computed from),
2. strips the `encoder.model.` wrapper prefix and loads the tensors STRICTLY into
   a stock `LayoutLMv3ForTokenClassification`, then writes safetensors and
   re-reads them to assert bit-equality with the original pickle,
3. saves the processor the evaluation pipeline actually used
   (`microsoft/layoutlmv3-base`, `apply_ocr=False`; tokenizer and image
   processor are identical to -large apart from `name_or_path`),
4. recomputes the SpanCRC thresholds at alpha=0.10, m0=25 on the seed-42
   calibration unit from the paper's cached temperature-scaled probabilities and
   asserts they equal the committed `span_crc_<ds>.json`, then writes them out.

Nothing is retrained and no threshold is re-chosen. Inputs (outputs/, results/,
data_bundle/) are local research artifacts that are not under version control.

    python tools/export_hf_release.py --out release/hf
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from safetensors.torch import load_file
from transformers import (LayoutLMv3Config, LayoutLMv3ForTokenClassification,
                          LayoutLMv3Processor)

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "scripts"))
import span_crc  # noqa: E402

ALPHA = 0.10
M0 = 25
TAG = "local_20260803"
PARENT = "microsoft/layoutlmv3-large"
PROCESSOR = "microsoft/layoutlmv3-base"

DATASETS = {
    # ds: (hf repo name, temperature-fit split, calibration unit)
    "sroie": ("calidoc-layoutlmv3-large-sroie", "val_es", "seed_s42"),
    "cord": ("calidoc-layoutlmv3-large-cord", "val_es", "fixed"),
    "funsd": ("calidoc-layoutlmv3-large-funsd", "es_holdout", "seed_s42"),
    "wildreceipt": ("calidoc-layoutlmv3-large-wildreceipt", "val_es", "seed_s42"),
}


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def cal_doc_ids(ds: str) -> tuple[list, str]:
    """Calibration doc ids in cache order (parquet order filtered by the split)."""
    splits = REPO / "data_bundle" / "splits"
    if ds == "cord":
        ids = set(json.loads((splits / "cord_cal.json").read_text())["doc_ids"])
        src = "validation"
    else:
        ids = set(json.loads((splits / f"{ds}_calib_seed42.json").read_text())["calibration"])
        src = "test"
    df = pd.read_parquet(REPO / "data_bundle" / ds / "processed" / f"{src}.parquet",
                         columns=["doc_id"])
    ordered = [d for d in df["doc_id"] if d in ids]
    assert len(ordered) == len(ids), (ds, len(ordered), len(ids))
    return ordered, src


def export_model(ds: str, out: Path) -> dict:
    ckpt = REPO / "outputs" / f"{ds}_retrain_s42_{TAG}" / "best"
    tcfg = json.loads((ckpt / "training_config.json").read_text())
    labels = json.loads((REPO / "data_bundle" / ds / "processed" / "label_list.json")
                        .read_text())["labels"]
    id2label = {i: l for i, l in enumerate(labels)}
    assert {int(k): v for k, v in tcfg["id2label"].items()} == id2label, ds

    raw = torch.load(ckpt / "pytorch_model.bin", map_location="cpu", weights_only=False)
    assert all(k.startswith("encoder.model.") for k in raw), ds
    sd = {k[len("encoder.model."):]: v for k, v in raw.items()}

    config = LayoutLMv3Config.from_pretrained(
        PARENT, num_labels=len(labels), id2label=id2label,
        label2id={l: i for i, l in id2label.items()})
    model = LayoutLMv3ForTokenClassification(config)
    missing, unexpected = model.load_state_dict(sd, strict=True)
    assert not missing and not unexpected
    model.save_pretrained(out, safe_serialization=True)

    reloaded = load_file(out / "model.safetensors")
    ref = model.state_dict()
    assert set(reloaded) <= set(ref), set(reloaded) - set(ref)
    n_eq = 0
    for k, v in reloaded.items():
        assert torch.equal(v, sd[k]), k
        n_eq += 1
    # tied/aliased tensors may be stored once; every original tensor must be covered
    assert set(sd) == set(ref), set(sd) ^ set(ref)

    LayoutLMv3Processor.from_pretrained(PROCESSOR, apply_ocr=False).save_pretrained(out)
    return {
        "source_checkpoint": str(ckpt.relative_to(REPO)),
        "source_sha256": sha256(ckpt / "pytorch_model.bin"),
        "safetensors_sha256": sha256(out / "model.safetensors"),
        "tensors_verified_equal": n_eq,
        "tensors_in_source": len(sd),
        "num_labels": len(labels),
        "id2label": id2label,
    }


def export_calibration(ds: str, unit_name: str, es_split: str, out: Path) -> dict:
    z, meta, id2label = span_crc.load_cache(ds, str(REPO / "results" / "cache"), TAG)
    unit = {u[0]: u for u in span_crc.calibration_units(meta)}[unit_name]
    _, cal_key, ev_key = unit
    cal_spans, cal_lac = span_crc.build_spans(z[f"{cal_key}_probs"], z[f"{cal_key}_labels"],
                                              z[f"{cal_key}_docstart"], id2label)
    ev_spans, ev_lac = span_crc.build_spans(z[f"{ev_key}_probs"], z[f"{ev_key}_labels"],
                                            z[f"{ev_key}_docstart"], id2label)
    res = span_crc.span_crc_unit(cal_spans, ev_spans, z[f"{ev_key}_probs"],
                                 z[f"{ev_key}_labels"], cal_lac, ev_lac,
                                 ALPHA, M0, id2label)

    committed = json.loads((REPO / "results" / "local_20260803" / f"span_crc_{ds}.json")
                           .read_text())
    cu = next(u for u in committed["per_unit"] if u["unit"] == unit_name)
    assert cu["pooled_crc"]["lambda"] == res["pooled_crc"]["lambda"], ds
    assert cu["token_marginal"]["qhat"] == res["token_marginal"]["qhat"], ds
    for f, v in cu["per_field"].items():
        assert v["lambda"] == res["per_field"][f]["lambda"], (ds, f)
        assert v["fallback"] == res["per_field"][f]["fallback"], (ds, f)

    lam0 = res["pooled_crc"]["lambda"]
    cal_count: dict = {}
    for s in cal_spans:
        cal_count[s["field"]] = cal_count.get(s["field"], 0) + 1
    by_field: dict = {}
    for s in cal_spans:
        by_field.setdefault(s["field"], []).append(s)
    all_fields = sorted({l[2:] for l in id2label.values() if l[:2] in ("B-", "I-")})
    fields = {}
    for f in all_fields:
        n = cal_count.get(f, 0)
        if n >= M0:
            lam, _ = span_crc.crc_lambda(span_crc._by_doc(by_field[f]), ALPHA)
        else:
            lam = lam0
        if f in res["per_field"]:
            assert res["per_field"][f]["lambda"] == lam, (ds, f)
        fields[f] = {"lambda": lam, "n_cal_spans": n, "fallback": n < M0}

    ids, src = cal_doc_ids(ds)
    assert len(ids) == len(np.asarray(z[f"{cal_key}_docstart"])) - 1, ds
    es_file = {"cord": "cord_val_es.json", "funsd": "funsd_es.json"}.get(ds, f"{ds}_val_es.json")

    cdir = out / "calibration"
    cdir.mkdir(parents=True, exist_ok=True)
    (cdir / "temperature.json").write_text(json.dumps({
        "temperature": meta["temperature"],
        "fit_split": es_split,
        "fit_objective": "NLL, LBFGS on log T (scripts/evaluate_conformal_phase2.py:fit_temperature)",
        "applied_as": "softmax(logits / T) over first-subword logits of each OCR word",
    }, indent=2) + "\n")
    (cdir / "pooled_threshold.json").write_text(json.dumps({
        "alpha": ALPHA,
        "pooled_lambda": lam0,
        "calibration_risk_at_lambda": res["pooled_crc"]["cal_risk"],
        "token_marginal_qhat": res["token_marginal"]["qhat"],
        "n_calibration_documents": len(ids),
        "n_calibration_documents_with_fields": res["n_cal_docs_with_spans"],
        "n_calibration_spans": res["n_cal_spans"],
    }, indent=2) + "\n")
    (cdir / "field_thresholds.json").write_text(json.dumps({
        "alpha": ALPHA,
        "m0": M0,
        "rule": "field types with >= m0 calibration spans get their own CRC threshold; "
                "others fall back to the pooled threshold",
        "fields": fields,
    }, indent=2) + "\n")
    (cdir / "split_manifest.json").write_text(json.dumps({
        "dataset": ds,
        "calibration_unit": unit_name,
        "calibration_source_split": src,
        "calibration_doc_ids": ids,
        "temperature_fit_split": es_split,
        "temperature_fit_split_file": f"data_bundle/splits/{es_file}",
        "note": "Document identifiers only; no dataset images or annotations are included.",
    }, indent=2) + "\n")

    return {
        "alpha": ALPHA,
        "m0": M0,
        "temperature": meta["temperature"],
        "pooled_lambda": lam0,
        "supported_field_types": [f for f, v in fields.items() if not v["fallback"]],
        "fallback_field_types": [f for f, v in fields.items() if v["fallback"]],
        "calibration_unit": unit_name,
        "evaluation_side_of_unit": {
            "field_crc_span_coverage": res["field_crc"]["span_coverage"],
            "field_crc_doc_risk": res["field_crc"]["ev_doc_risk"],
            "field_crc_avg_set_size": res["field_crc"]["avg_set_size"],
            "pooled_crc_span_coverage": res["pooled_crc"]["span_coverage"],
            "token_marginal_span_coverage": res["token_marginal"]["span_coverage"],
            "token_marginal_token_coverage": res["token_marginal"]["token_coverage"],
            "n_ev_spans": res["n_ev_spans"],
        },
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(REPO / "release" / "hf"))
    ap.add_argument("--datasets", nargs="*", default=list(DATASETS))
    args = ap.parse_args()

    manifest = {}
    for ds in args.datasets:
        name, es_split, unit = DATASETS[ds]
        out = Path(args.out) / name
        out.mkdir(parents=True, exist_ok=True)
        m = export_model(ds, out)
        c = export_calibration(ds, unit, es_split, out)
        phase2 = json.loads((REPO / "results" / "local_20260803" / f"phase2_{ds}.json")
                            .read_text())
        test = phase2["results"]["test_metrics"]
        cfg = {
            "calidoc_version": "wacv2027",
            "dataset": ds,
            "training_seed": 42,
            "base_model": PARENT,
            "processor": PROCESSOR,
            "max_length": 512,
            "label_alignment": "first subword of each OCR word",
            "bbox_format": "OCR word boxes normalized to 0-1000",
            "score": "max-token",
            "nonconformity": "1 - p(y | x_i), temperature-scaled",
            "exchangeable_unit": "document",
            "risk": "per-document missed-field fraction",
            "set_rule": "candidate-label: y in C(x_i) iff 1 - p(y|x_i) <= lambda_{type(y)}; "
                        "non-field labels (O) use the pooled lambda",
            **c,
            "released_checkpoint_test": {
                "entity_f1": test["f1"],
                "precision": test["precision"],
                "recall": test["recall"],
                "ece_percent": test["ece"],
                "n_test_docs": test["n_docs"],
                "source": f"results/local_20260803/phase2_{ds}.json",
            },
        }
        (out / "calidoc_config.json").write_text(json.dumps(cfg, indent=2) + "\n")
        manifest[ds] = {"repo": name, **{k: v for k, v in m.items() if k != "id2label"},
                        "entity_f1_test": test["f1"], "pooled_lambda": c["pooled_lambda"],
                        "temperature": c["temperature"],
                        "n_fallback": len(c["fallback_field_types"]),
                        "n_supported": len(c["supported_field_types"])}
        print(json.dumps({ds: manifest[ds]}, indent=2), flush=True)
    (Path(args.out) / "export_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")


if __name__ == "__main__":
    main()
