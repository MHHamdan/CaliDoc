"""Finalize the bundle: checksums (raw + processed), PROVENANCE.md,
environment freeze, and MANIFEST.json with counts, seeds, versions, and the
protocol constraints downstream training scripts must inherit."""

import datetime
import json
import platform
import shutil
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from common import BUNDLE_DIR, EXPECTED, RAW_DIR, sha256_file, status, write_json

FUNSD_URL = "https://guillaumejaume.github.io/FUNSD/dataset.zip"
CORD_REPO = "naver-clova-ix/cord-v2"


def checksum_tree(root: Path, out_file: Path, patterns):
    files = sorted(p for pat in patterns for p in root.rglob(pat) if p.is_file())
    with open(out_file, "w") as f:
        for p in files:
            f.write(f"{sha256_file(p)}  {p.relative_to(BUNDLE_DIR)}\n")
    return len(files)


def main():
    ck = BUNDLE_DIR / "checksums"
    ck.mkdir(parents=True, exist_ok=True)
    today = datetime.date.today().isoformat()

    # Raw checksums (copied from download stage).
    shutil.copy2(RAW_DIR / "funsd" / "dataset.zip.sha256", ck / "funsd_raw.sha256")
    shutil.copy2(RAW_DIR / "cord" / "cord_raw.sha256", ck / "cord_raw.sha256")
    funsd_zip_sha = (ck / "funsd_raw.sha256").read_text().split()[0]
    cord_rev = (RAW_DIR / "cord" / "REVISION").read_text().strip()

    # Processed checksums (parquet + label lists + every image).
    n_f = checksum_tree(BUNDLE_DIR / "funsd", ck / "funsd_processed.sha256",
                        ["*.parquet", "*.json", "*.png", "*.jpg"])
    n_c = checksum_tree(BUNDLE_DIR / "cord", ck / "cord_processed.sha256",
                        ["*.parquet", "*.json", "*.png", "*.jpg"])
    status("funsd", "checksum processed", ">=153 files", n_f, n_f >= 153)
    status("cord", "checksum processed", ">=1004 files", n_c, n_c >= 1004)

    # Environment freeze.
    freeze = subprocess.run([sys.executable, "-m", "pip", "freeze"],
                            capture_output=True, text=True, check=True).stdout
    (BUNDLE_DIR / "environment_freeze.txt").write_text(freeze)

    stats = {ds: json.loads((BUNDLE_DIR / "stats" / f"{ds}_stats.json").read_text())
             for ds in EXPECTED}
    labels = {ds: json.loads((BUNDLE_DIR / ds / "processed" / "label_list.json").read_text())
              for ds in EXPECTED}

    import datasets as hf_datasets
    import pyarrow
    import PIL

    manifest = {
        "created": today,
        "pipeline": "scripts/data_prep/make_datasets.sh",
        "normalized_format": {
            "storage": "parquet",
            "columns": ["doc_id", "tokens", "bboxes (0-1000 ints)",
                        "ner_tags (BIO strings)", "image_path (bundle-relative)",
                        "split", "ocr_confidence (null if absent in raw data)"],
            "note": "images stored at original resolution; resizing belongs in the dataloader",
        },
        "datasets": {
            "funsd": {
                "source": FUNSD_URL,
                "retrieved": today,
                "raw_sha256": funsd_zip_sha,
                "counts": {s: stats["funsd"][s]["n_docs"] for s in EXPECTED["funsd"]},
                "labels": labels["funsd"],
            },
            "cord": {
                "source": f"https://huggingface.co/datasets/{CORD_REPO}",
                "retrieved": today,
                "hf_revision": cord_rev,
                "counts": {s: stats["cord"][s]["n_docs"] for s in EXPECTED["cord"]},
                "labels": labels["cord"],
                "o_token_fraction_test": stats["cord"]["test"]["o_token_fraction"],
                "finding_o_class": "O never occurs in true CORD v2 annotations (every "
                    "valid_line has a category). The paper's 'O ~5% of test tokens' "
                    "claim was an artifact of the old fixed 61-label list spelling "
                    "sub-menu fields MENU.SUB_NM instead of the actual MENU.SUB.NM, "
                    "mapping those tokens to O. Fix the paper text and the Mondrian "
                    "O-class analysis for CORD accordingly.",
            },
            "docvqa": {
                "status": "NOT INCLUDED",
                "reason": "registration-gated at rrc.cvc.uab.es; cannot be scripted. "
                          "Optional -- paper refocused on NER. See PROVENANCE.md.",
            },
        },
        "splits": {
            "cord_val_es": {"file": "splits/cord_val_es.json", "seed": 42, "n": 50,
                            "purpose": "early stopping ONLY; temperature scaling fit here"},
            "cord_cal": {"file": "splits/cord_cal.json", "seed": 42, "n": 50,
                         "purpose": "conformal calibration ONLY"},
            "funsd_calib": {"files": [f"splits/funsd_calib_seed{s}.json" for s in (42, 123, 456)],
                            "scheme": "document-level 70/30 (35 calib / 15 eval) of test"},
            "funsd_es": {"file": "splits/funsd_es.json", "seed": 42, "n": 15,
                         "purpose": "early stopping ONLY; removed from training"},
        },
        "protocol_constraints": [
            "Training code must never see cord cal/ documents.",
            "Temperature scaling must be fit on cord val_es, never on cal.",
            "FUNSD early-stopping docs (funsd_es.json) must be removed from training.",
            "Conformal results on FUNSD must be reported across all three calib seeds.",
        ],
        "versions": {
            "python": platform.python_version(),
            "datasets": hf_datasets.__version__,
            "pyarrow": pyarrow.__version__,
            "pillow": PIL.__version__,
            "full_freeze": "environment_freeze.txt",
        },
    }
    write_json(BUNDLE_DIR / "MANIFEST.json", manifest)

    prov = f"""# CaliDoc Data Provenance

Generated {today} on lytos by scripts/data_prep/make_datasets.sh.

## FUNSD
- **Source (authoritative):** {FUNSD_URL}
- **Retrieved:** {today}
- **Archive sha256:** `{funsd_zip_sha}` (see checksums/funsd_raw.sha256)
- **Counts:** 149 train / 50 test documents (verified against the FUNSD paper, Jaume et al., ICDAR-OST 2019).
- **License:** FUNSD is distributed for research purposes only; see https://guillaumejaume.github.io/FUNSD/work/ (updated form understanding license terms on the official page).

## CORD v2
- **Source (authoritative):** https://huggingface.co/datasets/{CORD_REPO} (official Naver Clova release)
- **Retrieved:** {today}
- **Pinned HF revision:** `{cord_rev}` (raw parquet shard checksums in checksums/cord_raw.sha256)
- **Counts:** 800 train / 100 validation / 100 test documents (verified against Park et al., "CORD: A Consolidated Receipt Dataset for Post-OCR Parsing", 2019).
- **Label set (derived from data, not assumed):** {labels['cord']['num_bio_labels']} BIO labels over {labels['cord']['num_fields']} fields (cord/processed/label_list.json).
- **License:** CC BY 4.0 (per the dataset card).
- **Finding — O class:** O never occurs in true CORD v2 annotations; the paper's
  "O ~5% of test tokens" claim was an artifact of a label-spelling mismatch
  (`MENU.SUB_NM` vs actual `MENU.SUB.NM`) in the previous pipeline that mapped
  unknown categories to O. 4 fields occur only as single-word entities
  (B- without I-): MENU.ITEMSUBTOTAL, MENU.NUM, MENU.SUB.UNITPRICE, VOID_MENU.PRICE.
- **OCR confidence:** {"present" if any(s.get("has_ocr_confidence") for s in stats["cord"].values()) else "NOT present in raw annotations; ocr_confidence stored as null (relevant to the paper's λ_O=0 caveat)"}.

## DocVQA — NOT INCLUDED
- **Source:** https://rrc.cvc.uab.es/?ch=17 (Robust Reading Competition, Task 1)
- Registration-gated; cannot be downloaded by script. Optional for the refined
  NER-focused paper. If needed: register at rrc.cvc.uab.es, download Task 1
  train/val/test (39,463 / 5,349 / 5,188 questions), place under data/raw/docvqa/,
  and extend this pipeline before transfer.
"""
    (BUNDLE_DIR / "PROVENANCE.md").write_text(prov)
    status("bundle", "manifest+provenance", "written", "MANIFEST.json, PROVENANCE.md", True)


if __name__ == "__main__":
    main()
