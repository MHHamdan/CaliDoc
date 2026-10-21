"""CaliDoc interactive demo (WACV 2027).

Two workflows:

* **Try a Real Example** (default): two CORD v2 test receipts fetched from the
  public dataset repository ``MHamdan/calidoc-demo-examples`` and run through the
  released CORD checkpoint. No upload or token needed.
* **Upload Your Own**: a document image plus OCR words and boxes, with any of
  the four released dataset-specific checkpoints.

All extraction and calibration is done by ``calidoc.inference.CaliDocPredictor``;
this file only fetches inputs and presents the output. Run with

    python demo/gradio/app.py
"""

from __future__ import annotations

import html
import json
from functools import lru_cache

import gradio as gr
from huggingface_hub import hf_hub_download
from PIL import Image, ImageDraw, ImageFont

from calidoc.inference import CaliDocPredictor, normalize_boxes

EXAMPLES_REPO = "MHamdan/calidoc-demo-examples"
GITHUB = "https://github.com/MHHamdan/CaliDoc"
SUPPLEMENT = f"{GITHUB}/releases/download/wacv2027-camera-ready/CaliDoc_WACV2027_Supplementary.pdf"
COLLECTION = "https://huggingface.co/collections/MHamdan/calidoc-wacv-2027-6ad41a62c6a81846dfeb44b4"

MODELS = {
    "CORD (receipts)": "MHamdan/calidoc-layoutlmv3-large-cord",
    "FUNSD (forms)": "MHamdan/calidoc-layoutlmv3-large-funsd",
    "WildReceipt (receipts)": "MHamdan/calidoc-layoutlmv3-large-wildreceipt",
    "SROIE (receipts, case study)": "MHamdan/calidoc-layoutlmv3-large-sroie",
}

EXAMPLES = {
    "CORD: Representative receipt": {
        "doc_id": "cord_test_0042",
        "model": MODELS["CORD (receipts)"],
        "blurb": "Deterministically selected from the CORD test set (median word count) to "
                 "illustrate the standard CaliDoc workflow.",
    },
    "CORD: Review-routing case": {
        "doc_id": "cord_test_0014",
        "model": MODELS["CORD (receipts)"],
        "blurb": "A dense receipt (90th-percentile word count) containing a real extraction "
                 "disagreement, which shows why field-level review routing matters.",
    },
}
DEFAULT_EXAMPLE = "CORD: Representative receipt"

# raw calibrated-set status from calidoc.inference -> (state, action) shown to people
DISPLAY = {
    "covered": ("singleton", "pass"),
    "ambiguous_review": ("ambiguous", "review"),
    "conflict_review": ("conflict", "review"),
    "empty_abstain": ("empty", "abstain"),
}
COLORS = {"pass": "#2563eb", "ambiguous": "#d97706", "conflict": "#9333ea", "abstain": "#6b7280"}
TABLE_HEADERS = ["#", "Field", "Extracted value", "Set size", "Calibration", "State", "Action"]

SCOPE_HTML = """
<div style="border:1px solid #cbd5e1;border-left:4px solid #2563eb;border-radius:6px;padding:10px 14px;margin:4px 0 8px 0">
<b>Calibration scope</b>
<ul style="margin:6px 0 0 0">
<li>CaliDoc's finite-sample guarantee assumes exchangeability with the calibration distribution.
Each released checkpoint is calibrated for its own benchmark, so results on unrelated document
domains do not carry the same guarantee.</li>
<li>OCR is an external input. CaliDoc does not run an OCR engine.</li>
<li>The boundary-escape annotation audit is a corpus-level diagnostic, not a per-document error
detector, and it is not run here.</li>
</ul></div>
"""

LEGEND_HTML = "".join(
    f'<span style="display:inline-block;margin-right:14px"><span style="display:inline-block;'
    f'width:12px;height:12px;border:3px solid {c};margin-right:5px;vertical-align:-2px"></span>{t}</span>'
    for c, t in [(COLORS["pass"], "pass / singleton"), (COLORS["ambiguous"], "review / ambiguous"),
                 (COLORS["conflict"], "review / conflict: prediction excluded from its calibrated set"),
                 (COLORS["abstain"], "abstain / empty set")])
LEGEND_HTML = (f'<div style="font-size:0.9em"><b>Operational routing, not correctness probability.</b>'
               f'<br>{LEGEND_HTML}</div>')

OCR_SCHEMA_MD = """
CaliDoc accepts exactly this JSON (the format read by `python -m calidoc.predict`):

```json
{
  "words": ["Total", "16,500"],
  "boxes": [[435, 536, 520, 567], [600, 536, 686, 567]],
  "box_format": "pixel"
}
```

- `words`: OCR words in reading order.
- `boxes`: one `[x0, y0, x1, y1]` box per word.
- `box_format`: `"pixel"` (image pixel coordinates, the default when omitted) or
  `"normalized_1000"` (LayoutLMv3's 0 to 1000 grid).

Only the first 512 sub-word tokens are scored. Words past that limit are reported as truncated.
"""


# --------------------------------------------------------------------------- inputs

@lru_cache(maxsize=None)
def predictor(model_id: str) -> CaliDocPredictor:
    return CaliDocPredictor.from_pretrained(model_id)


def example_file(doc_id: str, name: str) -> str:
    return hf_hub_download(EXAMPLES_REPO, f"{doc_id}/{name}", repo_type="dataset")


@lru_cache(maxsize=None)
def example_inputs(doc_id: str):
    image = Image.open(example_file(doc_id, "image.png")).convert("RGB")
    ocr = json.load(open(example_file(doc_id, "ocr.json")))
    meta = json.load(open(example_file(doc_id, "metadata.json")))
    expected = json.load(open(example_file(doc_id, "expected_prediction.json")))
    return image, ocr, meta, expected


def boxes_for(ocr: dict, image: Image.Image):
    fmt = ocr.get("box_format", "pixel")
    if fmt == "pixel":
        return normalize_boxes(ocr["boxes"], *image.size)
    if fmt == "normalized_1000":
        return ocr["boxes"]
    raise gr.Error(f'Unsupported box_format {fmt!r}: use "pixel" or "normalized_1000".')


# --------------------------------------------------------------------------- presentation

def display_image(image: Image.Image, max_h: int = 1100) -> Image.Image:
    if image.height <= max_h:
        return image.copy()
    s = max_h / image.height
    return image.resize((round(image.width * s), max_h), Image.LANCZOS)


def overlay(image: Image.Image, result: dict, boxes) -> Image.Image:
    """Thin outlines around each predicted field, numbered tags left of the box.

    The view is cropped to the OCR word region (plus a margin) so that dense
    receipts photographed on a table stay legible; boxes are mapped accordingly.
    """
    xs0, ys0, xs1, ys1 = zip(*boxes)
    m = 40
    cx0, cy0 = max(min(xs0) - m, 0), max(min(ys0) - m, 0)
    cx1, cy1 = min(max(xs1) + m, 1000), min(max(ys1) + m, 1000)
    w0, h0 = image.size
    vis = display_image(image.crop((round(cx0 * w0 / 1000), round(cy0 * h0 / 1000),
                                    round(cx1 * w0 / 1000), round(cy1 * h0 / 1000))), max_h=1000)
    d = ImageDraw.Draw(vis)
    W, H = vis.size
    sx, sy = W / (cx1 - cx0), H / (cy1 - cy0)
    lw = max(2, round(W / 350))
    try:
        font = ImageFont.load_default(size=max(13, round(W / 45)))
    except TypeError:  # Pillow < 10.1
        font = ImageFont.load_default()
    th = getattr(font, "size", 11)
    for k, f in enumerate(result["fields"], 1):
        state, action = DISPLAY[f["status"]]
        color = COLORS["conflict" if state == "conflict" else action if action != "review" else "ambiguous"]
        x0, y0 = (f["bbox"][0] - cx0) * sx, (f["bbox"][1] - cy0) * sy
        x1, y1 = (f["bbox"][2] - cx0) * sx, (f["bbox"][3] - cy0) * sy
        d.rectangle([x0 - lw, y0 - lw, x1 + lw, y1 + lw], outline=color, width=lw)
        tag = str(k)
        tw = d.textlength(tag, font=font)
        d.rectangle([x0 - tw - 3 * lw - 6, y0 - lw, x0 - 2 * lw, y0 + th + 2], fill=color)
        d.text((x0 - tw - 2 * lw - 3, y0 - lw + 1), tag, fill="white", font=font)
    return vis


def table_rows(result: dict):
    rows = []
    for k, f in enumerate(result["fields"], 1):
        state, action = DISPLAY[f["status"]]
        sizes = f["set_sizes"]
        size = str(sizes[0]) if len(set(sizes)) == 1 else f"{min(sizes)}–{max(sizes)}"
        src = "pooled" if f["threshold_source"] == "pooled_fallback" else "field"
        rows.append([k, f["type"], f["text"], size, f"{src} λ={f['threshold']:.3f}", state, action])
    return rows


def counts(result: dict) -> dict:
    actions = [DISPLAY[f["status"]][1] for f in result["fields"]]
    return {
        "fields": len(result["fields"]),
        "pass": actions.count("pass"),
        "review": actions.count("review"),
        "abstain": actions.count("abstain"),
        "fallback": sum(f["threshold_source"] == "pooled_fallback" for f in result["fields"]),
        "conflict": sum(f["status"] == "conflict_review" for f in result["fields"]),
        "ambiguous": sum(f["status"] == "ambiguous_review" for f in result["fields"]),
        "unassigned": len(result["unassigned_field_candidates"]),
        "truncated": result["n_words_truncated"],
    }


def summary_html(result: dict, dataset: str | None) -> str:
    c = counts(result)
    cell = lambda label, v, color="#111827": (
        f'<div style="padding:6px 14px;border:1px solid #e2e8f0;border-radius:6px;text-align:center">'
        f'<div style="font-size:1.6em;font-weight:600;color:{color}">{v}</div>'
        f'<div style="font-size:0.85em;color:#475569">{label}</div></div>')
    cards = "".join([
        cell("Fields extracted", c["fields"]), cell("Pass through", c["pass"], COLORS["pass"]),
        cell("Review", c["review"], COLORS["ambiguous"]), cell("Abstain", c["abstain"], COLORS["abstain"]),
        cell("Pooled-fallback fields", c["fallback"])])
    route = result["summary"]["route"]
    out = (f'<div><b>CaliDoc decision summary</b> · document route: <b>{route}</b>'
           f'<div style="display:flex;flex-wrap:wrap;gap:8px;margin-top:6px">{cards}</div>')
    if dataset == "cord":
        out += ('<p style="font-size:0.9em;color:#475569;margin-top:8px">CORD is intentionally a '
                'difficult CaliDoc operating regime. Many field types have limited calibration '
                'support and fall back to the pooled threshold, and a few supported long-tail types '
                'need very large thresholds, which produces larger prediction sets. The WACV paper '
                'documents this behavior, and the demo does not hide it.</p>')
    return out + "</div>"


def _n(k: int, one: str, many: str) -> str:
    return f"{k} {one if k == 1 else many}"


def explanation(result: dict, dataset: str | None, field_thresholds: dict | None = None) -> str:
    """Deterministic 'What happened?' text built only from the output's status fields."""
    c = counts(result)
    if c["fields"] == 0:
        return "### What happened?\nThe extractor predicted no fields on this document."
    head = f"CaliDoc extracted {_n(c['fields'], 'field', 'fields')}."
    if c["pass"] == c["fields"] and not c["unassigned"]:
        head += (" Every field's calibrated set is a singleton equal to the prediction, so all "
                 "fields can pass through.")
    else:
        parts = []
        if c["pass"]:
            parts.append(f"{_n(c['pass'], 'field passes', 'fields pass')} through with singleton sets")
        if c["ambiguous"]:
            parts.append(f"{_n(c['ambiguous'], 'field is', 'fields are')} represented by "
                         f"multi-label prediction sets (ambiguous) and routed for review")
        if c["conflict"]:
            parts.append(f"{_n(c['conflict'], 'field has', 'fields have')} a calibrated set that "
                         f"excludes the predicted label (conflict) and is routed for review")
        if c["abstain"]:
            parts.append(f"{_n(c['abstain'], 'field has', 'fields have')} an empty calibrated set, "
                         f"so CaliDoc abstains")
        head += " " + "; ".join(parts) + "."
    lines = [head]

    ambiguous = [f for f in result["fields"] if f["status"] == "ambiguous_review"]
    if ambiguous:
        common = set.intersection(*({l[2:] for l in f["other_labels_in_set"]} for f in ambiguous))
        if common and field_thresholds:
            desc = ", ".join(f"`{t}` (λ = {field_thresholds[t]['lambda']:.3f}, "
                             f"{'pooled fallback' if field_thresholds[t]['fallback'] else 'own threshold'})"
                             for t in sorted(common) if t in field_thresholds)
            reach = ("every word's set" if all(field_thresholds[t]["lambda"] >= 1.0 for t in common
                                               if t in field_thresholds) else "most words' sets")
            lines.append(f"Every ambiguous set also admits {desc}. A threshold this large admits "
                         f"that label into {reach}, so the ambiguity reflects the calibrated "
                         f"threshold, not a disagreement specific to this document.")
        if dataset == "cord":
            lines.append(f"This is the CORD regime the paper documents: few field types have enough "
                         f"calibration support, and {_n(c['fallback'], 'extracted field uses', 'extracted fields use')} "
                         f"the pooled fallback threshold here.")
    for f in result["fields"]:
        if f["status"] == "conflict_review":
            lines.append(f"The **'{f['text']}'** extraction (predicted `{f['type']}`) is not presented "
                         f"as a confident final field: its worst word scores "
                         f"{f['max_token_score']:.2f} against a threshold of {f['threshold']:.3f}, "
                         f"so its calibrated set excludes the prediction and CaliDoc routes it for "
                         f"review.")
        elif f["status"] == "empty_abstain":
            lines.append(f"For **'{f['text']}'** (predicted `{f['type']}`), no label clears its "
                         f"threshold on at least one word, so CaliDoc abstains.")
    if c["unassigned"]:
        lines.append(f"{_n(c['unassigned'], 'word', 'words')} predicted as background still admit a "
                     f"field label in their calibrated set: possible missed fields to check.")
    if c["truncated"]:
        lines.append(f"{_n(c['truncated'], 'word', 'words')} beyond the 512-token limit were not scored.")
    return "### What happened?\n" + "\n\n".join(lines)


def ground_truth_md(result: dict, expected: dict) -> str:
    """Separate, explicitly labelled comparison with CORD gold labels (examples only)."""
    gt = expected.get("ground_truth")
    if not gt:
        return ""
    gold = {(tuple(f["word_span"]), f["type"]): f for f in gt["fields"]}
    pred = {(tuple(f["word_span"]), f["type"]): f for f in result["fields"]}
    gold_by_span = {tuple(f["word_span"]): f for f in gt["fields"]}
    wrong = [f for k, f in pred.items() if k not in gold]
    lines = ["### Ground-truth check (CORD gold labels, shown separately from routing)",
             f"- Gold fields: {gt['n_gold_fields']}; covered by their calibrated sets: "
             f"{gt['n_gold_fields'] - gt['n_missed']}.",
             f"- Predicted fields that exactly match a gold field: {len(pred) - len(wrong)} of {len(pred)}."]
    for f in wrong:
        g = gold_by_span.get(tuple(f["word_span"]))
        gl = f"gold `{g['type']}`" if g else "no gold field with the same span"
        state, action = DISPLAY[f["status"]]
        lines.append(f"- '{html.escape(f['text'])}': predicted `{f['type']}`, {gl}. CaliDoc state: "
                     f"**{state}**, action: **{action}**.")
    return "\n".join(lines)


def check_against_expected(result: dict, expected: dict) -> str:
    keys = ("type", "text", "labels", "set_sizes", "status")
    ref = expected["prediction"]["fields"]
    same = len(ref) == len(result["fields"]) and all(
        all(a[k] == b[k] for k in keys) for a, b in zip(result["fields"], ref))
    return ("matches the published expected output (fields, labels, set sizes, statuses)"
            if same else "**differs from the published expected output.** Please report this.")


# --------------------------------------------------------------------------- handlers

def load_example(name: str):
    ex = EXAMPLES[name]
    image, ocr, meta, _ = example_inputs(ex["doc_id"])
    info = (f"**Dataset:** CORD v2 · **Document ID:** `{meta['document_id']}` · **Split:** "
            f"{meta['split']} · **Released model:** "
            f"[`{ex['model']}`](https://huggingface.co/{ex['model']})\n\n{ex['blurb']}")
    tech = {"source": f"{meta['source']} @ {meta['source_revision']}, test index "
                      f"{meta['source_index_in_split']}",
            "license": meta["license"], "citation": meta["citation"],
            "selection": meta["selection"], "ocr": meta["ocr"],
            "n_words": len(ocr["words"]), "box_format": ocr["box_format"],
            "files": f"https://huggingface.co/datasets/{EXAMPLES_REPO}/tree/main/{ex['doc_id']}"}
    return display_image(image), info, tech, None, "", [], "", "", None


def run_example(name: str):
    ex = EXAMPLES[name]
    image, ocr, _, expected = example_inputs(ex["doc_id"])
    result = predictor(ex["model"]).predict(image, ocr["words"], boxes_for(ocr, image))
    check = f"Reproducibility check: this run {check_against_expected(result, expected)}."
    return (overlay(image, result, boxes_for(ocr, image)), summary_html(result, "cord"), table_rows(result),
            explanation(result, "cord", predictor(ex["model"]).fields) + "\n\n" + check, ground_truth_md(result, expected), result)


def run_upload(model_name: str, image, ocr_file):
    if image is None or ocr_file is None:
        raise gr.Error("Provide both a document image and its OCR JSON.")
    try:
        ocr = json.load(open(ocr_file if isinstance(ocr_file, str) else ocr_file.name))
    except (OSError, ValueError) as e:
        raise gr.Error(f"Could not read the OCR JSON: {e}")
    if not isinstance(ocr, dict) or "words" not in ocr or "boxes" not in ocr:
        raise gr.Error('The OCR JSON needs "words" and "boxes" (see the schema help).')
    if len(ocr["words"]) != len(ocr["boxes"]):
        raise gr.Error(f"{len(ocr['words'])} words but {len(ocr['boxes'])} boxes.")
    image = image.convert("RGB")
    dataset = MODELS[model_name].rsplit("-", 1)[-1]
    boxes = boxes_for(ocr, image)
    result = predictor(MODELS[model_name]).predict(image, [str(w) for w in ocr["words"]], boxes)
    return (overlay(image, result, boxes), summary_html(result, dataset), table_rows(result),
            explanation(result, dataset, predictor(MODELS[model_name]).fields), result)


# --------------------------------------------------------------------------- layout

def results_block():
    with gr.Row():
        with gr.Column(scale=5):
            vis = gr.Image(label="Predicted fields (cropped to the OCR region)", type="pil", interactive=False)
            gr.HTML(LEGEND_HTML)
        with gr.Column(scale=6):
            summary = gr.HTML()
            table = gr.Dataframe(headers=TABLE_HEADERS, interactive=False, wrap=True,
                                 column_widths=["4%", "22%", "18%", "11%", "17%", "14%", "14%"],
                                 max_height=760,
                                 label="Fields: calibrated-set state and recommended action")
            what = gr.Markdown()
    return vis, summary, table, what


def build_demo() -> gr.Blocks:
    with gr.Blocks(title="CaliDoc demo") as demo:
        gr.Markdown(
            "# CaliDoc: Complete-Field Reliability in Action\n"
            f"WACV 2027 · [GitHub]({GITHUB}) · [Supplement]({SUPPLEMENT}) · "
            f"[Models]({COLLECTION}) · Weights CC BY-NC-SA 4.0 (non-commercial)")
        gr.HTML(SCOPE_HTML)
        with gr.Tabs():
            with gr.Tab("Try a Real Example"):
                gr.Markdown(
                    "Choose a real CORD test document and run the released WACV 2027 model. "
                    "CaliDoc shows not only what the extractor predicts, but whether each complete "
                    "field should pass through, be reviewed, or abstain.")
                with gr.Row():
                    with gr.Column(scale=4):
                        choice = gr.Radio(list(EXAMPLES), value=DEFAULT_EXAMPLE, label="Example")
                        info = gr.Markdown()
                        with gr.Row():
                            load_btn = gr.Button("Load Example")
                            run_btn = gr.Button("Run CaliDoc", variant="primary")
                        gr.Markdown("Model: fixed to the CORD checkpoint for these CORD documents.")
                        with gr.Accordion("Technical details", open=False):
                            tech = gr.JSON(label="Example provenance and inputs")
                    with gr.Column(scale=5):
                        preview = gr.Image(label="Document", type="pil", interactive=False, height=520)
                vis, summary, table, what = results_block()
                gt = gr.Markdown()
                with gr.Accordion("Prediction JSON", open=False):
                    raw = gr.JSON()
                outs = [preview, info, tech, vis, summary, table, what, gt, raw]
                choice.change(load_example, choice, outs)
                load_btn.click(load_example, choice, outs)
                run_btn.click(run_example, choice, [vis, summary, table, what, gt, raw])
                demo.load(load_example, choice, outs)

            with gr.Tab("Upload Your Own"):
                gr.Markdown(
                    "**CaliDoc does not perform OCR.** The WACV pipeline takes OCR words and "
                    "bounding boxes as input. Models are dataset-specific: pick the checkpoint "
                    "whose benchmark matches your document type, and treat out-of-domain results "
                    "as outside the calibration guarantee.")
                with gr.Row():
                    with gr.Column(scale=4):
                        model = gr.Dropdown(list(MODELS), value="CORD (receipts)",
                                            label="Released model")
                        image = gr.Image(type="pil", label="Document image")
                        ocr = gr.File(label="OCR JSON", file_types=[".json"])
                        with gr.Accordion("Accepted OCR JSON schema", open=False):
                            gr.Markdown(OCR_SCHEMA_MD)
                        up_btn = gr.Button("Run CaliDoc", variant="primary")
                    with gr.Column(scale=5):
                        gr.Markdown(" | ".join(f"[{k}](https://huggingface.co/{v})"
                                               for k, v in MODELS.items()))
                u_vis, u_summary, u_table, u_what = results_block()
                with gr.Accordion("Prediction JSON", open=False):
                    u_raw = gr.JSON()
                up_btn.click(run_upload, [model, image, ocr],
                             [u_vis, u_summary, u_table, u_what, u_raw])
    return demo


if __name__ == "__main__":
    build_demo().launch()
