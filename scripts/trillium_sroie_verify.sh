#!/bin/bash
# WildReceipt addendum verification ON Trillium (spec #11: no training until
# checksum + bundle verification pass). Run via `ssh ... 'bash -s'`.
set -uo pipefail
# Checksum + extraction already verified OK on 2026-07-05; re-running both is
# harmless (idempotent) but kept for one-command reproducibility.
cd /scratch/hamdan/CaliDoc/data
echo "=== addendum checksum (must say OK) ==="
echo "387c7fb43f94d7e3863b1037d404f0a030627910e84a17e7f387149ea32ca265  sroie_pretagged_addendum_20260707.tar" | sha256sum -c - || { echo HASH_MISMATCH; exit 1; }
tar -xf sroie_pretagged_addendum_20260707.tar
echo "=== bundle addendum contents ==="
ls data_bundle/sroie/processed/ data_bundle/splits/ | head -20

# NOTE: never pipe `module load` — pipelines run in subshells and the
# environment changes are lost (cost one verify round-trip on 2026-07-05).
module --force purge
module load StdEnv/2023 gcc/12.3 cuda/12.2 python/3.11.5 arrow/17.0.0
source /scratch/hamdan/CaliDoc/venv/bin/activate
export HF_HOME=/scratch/hamdan/CaliDoc/hf_cache
cd /scratch/hamdan/CaliDoc/code
echo "=== verify_bundle + protocol guards on Trillium copy ==="
python - <<'EOF'
import sys, logging
logging.disable(logging.INFO)
sys.path.insert(0, '.')
from calidoc.data import verify_bundle, SROIEDataset, ProtocolViolation, conformal_split
B = '/scratch/hamdan/CaliDoc/data/data_bundle'
m = verify_bundle(B, 'sroie')
assert m['split_sizes'] == {'train': 566, 'val_es': 60, 'test': 347}, m['split_sizes']
assert m['num_labels'] == 9
print("splits:", m['split_sizes'], "| O:", {k: round(v,4) for k,v in m['o_fraction_per_source_split'].items()})
try:
    SROIEDataset(B, 'test')  # fit stage must refuse test
    raise SystemExit("GUARD MISSING")
except ProtocolViolation:
    print("stage guard OK")
cal42, ev42 = conformal_split(B, 'sroie', 42)
assert (len(cal42), len(ev42)) == (242, 105)
enc = SROIEDataset(B, 'val_es')[0]
print("val_es[0] shapes:", {k: tuple(v.shape) for k, v in enc.items()})
import os
os.environ['HF_HUB_OFFLINE'] = '1'; os.environ['TRANSFORMERS_OFFLINE'] = '1'
from transformers import LayoutLMv3ForTokenClassification
LayoutLMv3ForTokenClassification.from_pretrained("microsoft/layoutlmv3-large", num_labels=9)
print("offline -large load with 9 labels OK")
print("SROIE_TRILLIUM_VERIFY_OK")
EOF
echo "=== queue ==="
squeue -u hamdan
ssh -n trig-login01 squeue -u hamdan 2>/dev/null || true
echo "=== done ==="
