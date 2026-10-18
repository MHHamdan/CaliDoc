"""Command-line field extraction with a released CaliDoc model.

    python -m calidoc.predict \
        --model MHamdan/calidoc-layoutlmv3-large-cord \
        --image receipt.png --ocr receipt_ocr.json --out prediction.json

OCR is a prerequisite and is not run here. ``--ocr`` is a JSON object
``{"words": [...], "boxes": [[x0, y0, x1, y1], ...], "box_format": "pixel" | "normalized_1000"}``
(``box_format`` defaults to ``pixel``).
"""

from __future__ import annotations

import argparse
import json
import sys

from PIL import Image

from .inference import CaliDocPredictor, normalize_boxes


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--model", required=True, help="HF model id or local directory")
    ap.add_argument("--image", required=True)
    ap.add_argument("--ocr", required=True, help="JSON with words and boxes")
    ap.add_argument("--out", help="write prediction JSON here (default: stdout)")
    ap.add_argument("--device", default=None)
    ap.add_argument("--revision", default=None)
    args = ap.parse_args(argv)

    ocr = json.load(open(args.ocr))
    image = Image.open(args.image).convert("RGB")
    boxes = ocr["boxes"]
    if ocr.get("box_format", "pixel") == "pixel":
        boxes = normalize_boxes(boxes, *image.size)
    elif ocr["box_format"] != "normalized_1000":
        raise SystemExit(f"unknown box_format {ocr['box_format']!r}")

    predictor = CaliDocPredictor.from_pretrained(args.model, device=args.device,
                                                 revision=args.revision)
    result = predictor.predict(image, ocr["words"], boxes)
    text = json.dumps(result, indent=2, ensure_ascii=False)
    if args.out:
        with open(args.out, "w") as f:
            f.write(text + "\n")
    else:
        print(text)
    return 0


if __name__ == "__main__":
    sys.exit(main())
