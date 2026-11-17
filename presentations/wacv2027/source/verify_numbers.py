"""Cross-check every numeric token on every slide against the camera-ready text.

Extracts text from the PPTX (shapes, tables, chart caches) and searches each number in
the main paper and supplement text (PyMuPDF extraction, hyphenation/whitespace removed).
Prints numbers not found verbatim; those are reviewed by hand in the Claim Audit.
Usage: python3 verify_numbers.py deck.pptx main.pdf supp.pdf [extra.json ...]
"""
import re, sys, json, zipfile
import fitz
from pptx import Presentation

deck, main_pdf, supp_pdf, *extra = sys.argv[1:]
def pdf_text(p):
    t = "".join(pg.get_text() for pg in fitz.open(p))
    t = t.replace("−", "-").replace(" ", "").replace(" ", "")
    return re.sub(r"\s+", "", t)
corpus = pdf_text(main_pdf) + "|" + pdf_text(supp_pdf)
for e in extra:
    corpus += "|" + re.sub(r"\s+", "", open(e).read())

prs = Presentation(deck)
z = zipfile.ZipFile(deck)
num = re.compile(r"(?<![\w.])\d[\d,]*(?:\.\d+)?%?")
SKIP = {"0", "1", "2", "3", "4", "5", "10", "100", "2027", "1597", "0.0", "1.0", "2.0", "3.0", "4.0", "0.5", "0.25", "0.75", "0.50", "0.00", "1.00"}
missing = {}
def check(slide_no, text):
    for m in num.findall(text):
        k = m.rstrip("%")
        if k in SKIP or len(k) <= 1:
            continue
        if k.replace(",", "") not in corpus and k not in corpus:
            missing.setdefault(slide_no, set()).add(m)
for i, s in enumerate(prs.slides, 1):
    for sh in s.shapes:
        if sh.has_text_frame:
            check(i, sh.text_frame.text)
        if sh.has_table:
            for r in sh.table.rows:
                for c in r.cells:
                    check(i, c.text)
        if sh.has_chart:
            for pl in sh.chart.plots:
                for ser in pl.series:
                    for v in ser.values:
                        if v:
                            check(i, f"{v:g}")
for k in sorted(missing):
    print(f"slide {k}: not found verbatim -> {sorted(missing[k])}")
print("checked", len(prs.slides), "slides")
