"""Download official CORD v2 (naver-clova-ix/cord-v2) via huggingface_hub snapshot.

Snapshot pins the exact dataset repo revision; raw parquet shards are
checksummed and split counts asserted against published statistics.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from common import EXPECTED, RAW_DIR, hard_stop, sha256_file, status

from huggingface_hub import snapshot_download


def main():
    raw = RAW_DIR / "cord"
    raw.mkdir(parents=True, exist_ok=True)

    from huggingface_hub import HfApi
    ref = HfApi().dataset_info("naver-clova-ix/cord-v2").sha

    local = Path(snapshot_download(
        repo_id="naver-clova-ix/cord-v2",
        repo_type="dataset",
        revision=ref,
        local_dir=str(raw / "snapshot"),
    ))
    (raw / "REVISION").write_text(f"{ref}\n")

    parquets = sorted(local.rglob("*.parquet"))
    if not parquets:
        hard_stop("no parquet shards found in cord-v2 snapshot")

    with open(raw / "cord_raw.sha256", "w") as f:
        for p in parquets:
            f.write(f"{sha256_file(p)}  {p.relative_to(local)}\n")

    # Count rows per split from the raw shards.
    import pyarrow.parquet as pq
    counts = {}
    for p in parquets:
        split = p.name.split("-")[0]
        counts[split] = counts.get(split, 0) + pq.read_metadata(p).num_rows

    ok = True
    for split, exp in EXPECTED["cord"].items():
        got = counts.get(split, 0)
        ok &= status("cord", f"download {split}", exp, got, got == exp)
    if not ok:
        hard_stop(f"CORD split counts {counts} do not match published 800/100/100")

    print(f"cord snapshot revision: {ref}")


if __name__ == "__main__":
    main()
