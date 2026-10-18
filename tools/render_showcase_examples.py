#!/usr/bin/env python3
"""Render showcase figures from machine-generated prediction JSON.

Every value drawn comes from `assets/examples/<ds>/<id>/prediction.json`
(written by tools/showcase_examples.py); nothing is typed by hand. Only
examples whose dataset is cleared for redistribution carry a `document.png`
and are rendered.

Layout: original document | document with predicted field boxes, coloured by
calibrated-set status | field table. Status colours deliberately avoid a
green/red "correct/wrong" reading: the statuses describe calibrated prediction
sets, not verified correctness.

    python tools/render_showcase_examples.py assets/examples
"""

from __future__ import annotations

import json
import sys
import textwrap
from pathlib import Path

from matplotlib import font_manager
from PIL import Image, ImageDraw, ImageFont

STATUS_STYLE = {
    "covered": ("#2563eb", "Covered: singleton set = prediction"),
    "ambiguous_review": ("#d97706", "Ambiguous set: review"),
    "conflict_review": ("#a21caf", "Set excludes prediction: review"),
    "empty_abstain": ("#6b7280", "Empty set: abstain"),
}
INK, MUTED, RULE, BG = "#111827", "#4b5563", "#d1d5db", "#ffffff"


def font(size: int, bold: bool = False):
    path = font_manager.findfont(font_manager.FontProperties(
        family="DejaVu Sans", weight="bold" if bold else "normal"))
    return ImageFont.truetype(path, size)


def to_px(box, w, h):
    x0, y0, x1, y1 = box
    return [x0 * w / 1000, y0 * h / 1000, x1 * w / 1000, y1 * h / 1000]


def render(example_dir: Path) -> Path | None:
    ex = json.loads((example_dir / "prediction.json").read_text())
    if not ex.get("redistribution_cleared") or "image" not in ex:
        return None
    doc = Image.open(example_dir / ex["image"]).convert("RGB")
    pred = ex["prediction"]
    fields = pred["fields"]
    # crop to the OCR word region (plus margin) so dense receipts stay legible;
    # boxes are re-expressed on the cropped 0-1000 grid
    ocr = json.loads((example_dir / "ocr.json").read_text())
    xs0, ys0, xs1, ys1 = zip(*ocr["boxes"])
    mx, my = 40, 40
    cx0, cy0 = max(min(xs0) - mx, 0), max(min(ys0) - my, 0)
    cx1, cy1 = min(max(xs1) + mx, 1000), min(max(ys1) + my, 1000)
    w0, h0 = doc.size
    doc = doc.crop((round(cx0 * w0 / 1000), round(cy0 * h0 / 1000),
                    round(cx1 * w0 / 1000), round(cy1 * h0 / 1000)))
    scale = 1000 / doc.height
    doc = doc.resize((round(doc.width * scale), round(doc.height * scale)), Image.LANCZOS)
    W, H = doc.size

    def crop_box(b):
        return [(b[0] - cx0) * 1000 / (cx1 - cx0), (b[1] - cy0) * 1000 / (cy1 - cy0),
                (b[2] - cx0) * 1000 / (cx1 - cx0), (b[3] - cy0) * 1000 / (cy1 - cy0)]

    overlay = doc.copy()
    d = ImageDraw.Draw(overlay, "RGBA")
    f_tag = font(15, bold=True)
    for k, f in enumerate(fields, 1):
        color = STATUS_STYLE[f["status"]][0]
        x0, y0, x1, y1 = to_px(crop_box(f["bbox"]), W, H)
        d.rectangle([x0 - 2, y0 - 2, x1 + 2, y1 + 2], outline=color, width=3)
        tag = str(k)
        tw = d.textlength(tag, font=f_tag)
        # tag sits left of the box so stacked receipt lines do not hide each other
        d.rectangle([x0 - tw - 12, y0 - 2, x0 - 3, y0 + 18], fill=color)
        d.text((x0 - tw - 8, y0 - 1), tag, fill="white", font=f_tag)

    # field table
    f_h, f_b, f_s = font(30, bold=True), font(21), font(19)
    col_w = 1080
    rows = []
    for k, f in enumerate(fields, 1):
        text = textwrap.shorten(f["text"], width=30, placeholder="…")
        sizes = ",".join(map(str, f["set_sizes"]))
        rows.append((k, f["type"], text, sizes, f["threshold"], f["threshold_source"], f["status"]))
    row_h = 32
    gt = ex.get("ground_truth")
    head_h = 120
    legend_h = 36 * len(STATUS_STYLE) + 20
    table_h = head_h + row_h * (len(rows) + 1) + legend_h + 220
    canvas_h = max(H + 190, table_h)
    canvas = Image.new("RGB", (W * 2 + col_w + 80, canvas_h), BG)
    c = ImageDraw.Draw(canvas)

    c.text((20, 18), f"CaliDoc · {ex['attribution']['dataset']} · {ex['doc_id']} "
                     f"({ex['split']}, {ex['role']})", fill=INK, font=f_h)
    cal = pred["calibration"]
    c.text((20, 56), f"Model: {pred['model']}   alpha = {cal['alpha']}   m0 = {cal['m0']}   "
                     f"pooled lambda = {cal['pooled_lambda']:.4f}   T = {cal['temperature']:.4f}",
           fill=MUTED, font=f_b)
    c.text((20, 86), "Left: benchmark document.  Middle: predicted fields, numbered, coloured by "
                     "calibrated-set status.  Right: values from prediction.json.",
           fill=MUTED, font=f_s)
    top = 120
    canvas.paste(doc, (20, top))
    canvas.paste(overlay, (40 + W, top))
    c.text((20, top + H + 40), "Image cropped to the OCR word region; boxes are the dataset's "
                               "OCR word boxes merged per predicted field.", fill=MUTED, font=f_s)
    c.rectangle([20, top, 20 + W, top + H], outline=RULE)
    c.rectangle([40 + W, top, 40 + 2 * W, top + H], outline=RULE)

    x = 60 + 2 * W
    y = top
    cols = [(0, "#"), (44, "field type"), (400, "predicted text"), (760, "set sizes"),
            (900, "λ (source)")]
    for dx, name in cols:
        c.text((x + dx, y), name, fill=MUTED, font=font(19, bold=True))
    y += row_h
    c.line([x, y - 4, x + col_w - 20, y - 4], fill=RULE)
    for k, ftype, text, sizes, lam, src, status in rows:
        color = STATUS_STYLE[status][0]
        c.rectangle([x, y + 3, x + 32, y + 27], fill=color)
        c.text((x + 4, y + 4), str(k), fill="white", font=font(17, bold=True))
        c.text((x + 44, y), textwrap.shorten(ftype, 30, placeholder="…"), fill=INK, font=f_s)
        c.text((x + 400, y), text, fill=INK, font=f_s)
        c.text((x + 760, y), textwrap.shorten(sizes, 14, placeholder="…"), fill=INK, font=f_s)
        c.text((x + 900, y), f"{lam:.3f} ({'pool' if src == 'pooled_fallback' else 'field'})",
               fill=INK, font=f_s)
        y += row_h
    y += 14
    for status, (color, label) in STATUS_STYLE.items():
        n = pred["summary"]["status_counts"][status]
        c.rectangle([x, y + 3, x + 24, y + 27], fill=color)
        c.text((x + 36, y + 2), f"{label}  ({n})", fill=INK, font=f_b)
        y += 36
    y += 8
    lines = [f"Document route: {pred['summary']['route']}"]
    if pred["unassigned_field_candidates"]:
        lines.append(f"Words predicted O whose set admits a field: "
                     f"{len(pred['unassigned_field_candidates'])}")
    if gt:
        lines.append(f"Ground truth: {gt['n_gold_fields'] - gt['n_missed']}/{gt['n_gold_fields']} "
                     f"gold fields covered by the calibrated sets")
    lines.append("Guarantee: expected per-document field-miss fraction ≤ alpha on documents")
    lines.append("exchangeable with the calibration set. Statuses are not correctness labels.")
    for line in lines:
        c.text((x, y), line, fill=INK if not line.startswith(("Guarantee", "exchange")) else MUTED,
               font=f_b)
        y += 30
    att = ex["attribution"]
    c.text((20, top + H + 14),
           f"Image and annotations: {att['dataset']} ({att['citation'].split(',')[0]}), "
           f"{att['license']}, {att['source']}", fill=MUTED, font=f_s)
    out = example_dir / "showcase.png"
    canvas.save(out, optimize=True)
    return out


def main():
    root = Path(sys.argv[1] if len(sys.argv) > 1 else "assets/examples")
    for p in sorted(root.glob("*/*/prediction.json")):
        out = render(p.parent)
        print(out if out else f"skipped (not cleared): {p.parent}")


if __name__ == "__main__":
    main()
