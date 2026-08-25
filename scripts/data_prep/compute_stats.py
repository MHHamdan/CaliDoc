"""Compute and persist label-distribution statistics per dataset/split.

Feeds the Mondrian/per-class analysis. Also verifies the paper's claim that
the O class is ~5% of CORD test tokens (reported, not asserted — a mismatch
is flagged for the paper text, not a pipeline failure).
"""

import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from common import BUNDLE_DIR, EXPECTED, status, write_json

import pyarrow.parquet as pq


def split_stats(dataset, split):
    table = pq.read_table(BUNDLE_DIR / dataset / "processed" / f"{split}.parquet",
                          columns=["doc_id", "tokens", "ner_tags", "ocr_confidence"])
    has_conf = any(c is not None for c in table.column("ocr_confidence").to_pylist())
    tag_counts, entity_counts = Counter(), Counter()
    n_tokens, n_docs = 0, table.num_rows
    for tags in table.column("ner_tags").to_pylist():
        n_tokens += len(tags)
        for t in tags:
            tag_counts[t] += 1
            if t.startswith("B-"):
                entity_counts[t[2:]] += 1
    return {
        "n_docs": n_docs,
        "n_tokens": n_tokens,
        "avg_tokens_per_doc": round(n_tokens / n_docs, 2),
        "o_token_fraction": round(tag_counts.get("O", 0) / n_tokens, 6),
        "has_ocr_confidence": has_conf,
        "token_counts_per_tag": dict(sorted(tag_counts.items())),
        "entity_counts_per_field": dict(sorted(entity_counts.items())),
    }


def main():
    out = BUNDLE_DIR / "stats"
    out.mkdir(parents=True, exist_ok=True)

    for dataset, splits in EXPECTED.items():
        stats = {split: split_stats(dataset, split) for split in splits}
        write_json(out / f"{dataset}_stats.json", stats)
        for split, s in stats.items():
            status(dataset, f"stats {split}",
                   f"{splits[split]} docs",
                   f"{s['n_docs']} docs, {s['n_tokens']} tok, O={s['o_token_fraction']:.2%}",
                   s["n_docs"] == splits[split])

    o_test = split_stats("cord", "test")["o_token_fraction"]
    verdict = "consistent with ~5% claim" if 0.02 <= o_test <= 0.08 else "DIFFERS from ~5% claim -- fix paper text"
    print(f"cord     | verify O-fraction claim | paper says ~5% of test tokens | actual {o_test:.2%} | {verdict}")


if __name__ == "__main__":
    main()
