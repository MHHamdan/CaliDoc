"""
CaliDoc: Calibrated Field-Level Risk and Annotation Auditing for Document
Information Extraction (WACV 2027).

Inference with a released checkpoint:

    from calidoc.inference import CaliDocPredictor

    predictor = CaliDocPredictor.from_pretrained("MHamdan/calidoc-layoutlmv3-large-cord")
    result = predictor.predict(image, words, boxes)   # boxes on the 0-1000 grid

Research code (training, probability caching, SpanCRC, boundary-escape audit)
lives in `calidoc.data`, `calidoc.models`, `calidoc.training` and `scripts/`.
"""

__version__ = "0.1.0"
__author__ = "Mohammed Hamdan"
