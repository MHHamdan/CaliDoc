"""Field-level inference with the released CaliDoc checkpoints.

A released model repository (e.g. ``MHamdan/calidoc-layoutlmv3-large-sroie``)
holds a stock ``LayoutLMv3ForTokenClassification`` plus the CaliDoc calibration
artifacts fitted for that exact checkpoint:

    calidoc_config.json               alpha, m0, temperature, supported/fallback fields
    calibration/field_thresholds.json lambda_t per field type (pooled lambda on fallback)
    calibration/pooled_threshold.json pooled SpanCRC lambda
    calibration/temperature.json      temperature and the split it was fitted on

Inference reproduces the evaluation pipeline of the paper: the processor runs
with the caller's OCR words and 0-1000 boxes (no OCR is performed here), the
first sub-word of every word carries that word's logits, and probabilities are
``softmax(logits / T)``. The calibrated set of word i is the candidate-label
rule SpanCRC is calibrated for,

    C(x_i) = { y : 1 - p(y | x_i) <= lambda_{type(y)} },

with non-field labels (``O``) at the pooled lambda. A gold field is covered
exactly when every one of its words keeps its gold label in C, and SpanCRC
bounds the expected per-document fraction of missed gold fields by alpha on
documents exchangeable with the calibration documents. Nothing here is a
per-field probability of correctness.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Union

import numpy as np
import torch
from PIL import Image

STATUS_COVERED = "covered"
STATUS_AMBIGUOUS = "ambiguous_review"
STATUS_CONFLICT = "conflict_review"
STATUS_EMPTY = "empty_abstain"

STATUS_HELP = {
    STATUS_COVERED: "every word's calibrated set is exactly the predicted label",
    STATUS_AMBIGUOUS: "every word keeps the predicted label, but some word's set "
                      "also admits other labels",
    STATUS_CONFLICT: "some word's calibrated set is non-empty but excludes the "
                     "predicted label",
    STATUS_EMPTY: "some word's calibrated set is empty: no label clears its threshold",
}


def _resolve(model_id: str, filename: str, revision: Optional[str]) -> Path:
    local = Path(model_id) / filename
    if local.exists():
        return local
    from huggingface_hub import hf_hub_download
    return Path(hf_hub_download(model_id, filename, revision=revision))


def _pred_spans(tags: Sequence[str]) -> List[tuple]:
    """(start, end, field) for predicted BIO tags; a type change starts a new span."""
    out, start, field = [], None, None
    for i, t in enumerate(tags):
        if t.startswith("B-") or (t.startswith("I-") and t[2:] != field):
            if start is not None:
                out.append((start, i, field))
            start, field = i, t[2:]
        elif not t.startswith("I-"):
            if start is not None:
                out.append((start, i, field))
            start, field = None, None
    if start is not None:
        out.append((start, len(tags), field))
    return out


def normalize_boxes(boxes: Sequence[Sequence[float]], width: int, height: int) -> List[List[int]]:
    """Pixel [x0, y0, x1, y1] boxes to LayoutLMv3's 0-1000 grid."""
    out = []
    for x0, y0, x1, y1 in boxes:
        out.append([
            int(min(max(1000 * x0 / width, 0), 1000)),
            int(min(max(1000 * y0 / height, 0), 1000)),
            int(min(max(1000 * x1 / width, 0), 1000)),
            int(min(max(1000 * y1 / height, 0), 1000)),
        ])
    return out


class CaliDocPredictor:
    """LayoutLMv3 extractor plus the SpanCRC thresholds fitted for it."""

    def __init__(self, model, processor, config: dict, field_thresholds: dict,
                 device: Union[str, torch.device] = "cpu", model_id: str = ""):
        self.model = model.eval().to(device)
        self.processor = processor
        self.config = config
        self.device = torch.device(device)
        self.model_id = model_id
        self.id2label = {int(k): v for k, v in model.config.id2label.items()}
        self.temperature = float(config["temperature"])
        self.pooled_lambda = float(config["pooled_lambda"])
        self.fields = field_thresholds["fields"]
        thr = np.full(len(self.id2label), self.pooled_lambda, dtype=np.float64)
        for j, lab in self.id2label.items():
            if lab.startswith(("B-", "I-")) and lab[2:] in self.fields:
                thr[j] = self.fields[lab[2:]]["lambda"]
        self.label_thresholds = thr

    @classmethod
    def from_pretrained(cls, model_id: str, device: Optional[str] = None,
                        revision: Optional[str] = None) -> "CaliDocPredictor":
        from transformers import AutoModelForTokenClassification, AutoProcessor

        device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        # eager attention matches the pipeline the calibration was fitted on
        model = AutoModelForTokenClassification.from_pretrained(
            model_id, revision=revision, attn_implementation="eager")
        processor = AutoProcessor.from_pretrained(model_id, revision=revision, apply_ocr=False)
        config = json.loads(_resolve(model_id, "calidoc_config.json", revision).read_text())
        fields = json.loads(_resolve(model_id, "calibration/field_thresholds.json",
                                     revision).read_text())
        return cls(model, processor, config, fields, device=device, model_id=model_id)

    @torch.no_grad()
    def word_probabilities(self, image: Image.Image, words: Sequence[str],
                           boxes: Sequence[Sequence[int]]) -> np.ndarray:
        """Temperature-scaled label probabilities, one row per word kept after
        the 512-token truncation (words past the limit are dropped)."""
        enc = self.processor(image.convert("RGB"), list(words), boxes=[list(map(int, b)) for b in boxes],
                             max_length=512, padding="max_length", truncation=True,
                             return_tensors="pt")
        word_ids = enc.word_ids(0)
        first = {}
        for pos, w in enumerate(word_ids):
            if w is not None and w not in first:
                first[w] = pos
        idx = [first[w] for w in sorted(first)]
        inputs = {k: v.to(self.device) for k, v in enc.items()}
        logits = self.model(**inputs).logits[0, idx].float()
        return torch.softmax(logits / self.temperature, dim=-1).cpu().numpy()

    def calibrated_sets(self, probs: np.ndarray) -> np.ndarray:
        """Boolean [n_words, n_labels] membership of the candidate-label sets."""
        return (1.0 - probs) <= self.label_thresholds[None, :]

    def predict(self, image: Union[str, Path, Image.Image], words: Sequence[str],
                boxes: Sequence[Sequence[int]]) -> Dict:
        """Extract fields and their calibrated-set status.

        ``boxes`` must already be on the 0-1000 grid (see ``normalize_boxes``).
        """
        if not isinstance(image, Image.Image):
            image = Image.open(image)
        if len(words) != len(boxes):
            raise ValueError(f"{len(words)} words but {len(boxes)} boxes")
        probs = self.word_probabilities(image, words, boxes)
        n = probs.shape[0]
        sets = self.calibrated_sets(probs)
        pred = probs.argmax(1)
        tags = [self.id2label[int(p)] for p in pred]

        fields = []
        for a, b, field in _pred_spans(tags):
            sizes = sets[a:b].sum(1)
            keeps_pred = sets[np.arange(a, b), pred[a:b]]
            if (sizes == 0).any():
                status = STATUS_EMPTY
            elif not keeps_pred.all():
                status = STATUS_CONFLICT
            elif (sizes > 1).any():
                status = STATUS_AMBIGUOUS
            else:
                status = STATUS_COVERED
            alt = sorted({self.id2label[j] for i in range(a, b)
                          for j in np.flatnonzero(sets[i]) if j != pred[i]})
            fmeta = self.fields.get(field, {"lambda": self.pooled_lambda, "fallback": True,
                                            "n_cal_spans": 0})
            box = np.asarray([boxes[i] for i in range(a, b)])
            fields.append({
                "type": field,
                "text": " ".join(words[a:b]),
                "word_span": [a, b],
                "bbox": [int(box[:, 0].min()), int(box[:, 1].min()),
                         int(box[:, 2].max()), int(box[:, 3].max())],
                "labels": tags[a:b],
                "set_sizes": [int(s) for s in sizes],
                "max_set_size": int(sizes.max()),
                "other_labels_in_set": alt,
                "max_token_score": float((1.0 - probs[np.arange(a, b), pred[a:b]]).max()),
                "threshold": float(fmeta["lambda"]),
                "threshold_source": "pooled_fallback" if fmeta["fallback"] else "field",
                "status": status,
            })

        # words the extractor left as O whose calibrated set still admits a field
        o_ids = {j for j, l in self.id2label.items() if not l.startswith(("B-", "I-"))}
        field_ids = [j for j in self.id2label if j not in o_ids]
        missed = []
        for i in range(n):
            if int(pred[i]) in o_ids:
                cands = [self.id2label[j] for j in field_ids if sets[i, j]]
                if cands:
                    missed.append({"word_index": i, "text": words[i],
                                   "candidate_labels": cands})

        counts = {s: sum(f["status"] == s for f in fields) for s in STATUS_HELP}
        return {
            "model": self.model_id,
            "n_words": len(words),
            "n_words_scored": n,
            "n_words_truncated": len(words) - n,
            "fields": fields,
            "unassigned_field_candidates": missed,
            "summary": {
                "n_fields": len(fields),
                "status_counts": counts,
                "route": "accept" if counts[STATUS_COVERED] == len(fields) and not missed
                         and n == len(words) else "review",
            },
            "calibration": {
                "alpha": self.config["alpha"],
                "m0": self.config["m0"],
                "temperature": self.temperature,
                "pooled_lambda": self.pooled_lambda,
                "threshold_mode": "field-conditional SpanCRC with pooled fallback",
                "fallback_fields": self.config["fallback_field_types"],
                "guarantee": "E[per-document fraction of missed gold fields] <= alpha, "
                             "for documents exchangeable with the calibration documents",
                "status_definitions": STATUS_HELP,
            },
        }
