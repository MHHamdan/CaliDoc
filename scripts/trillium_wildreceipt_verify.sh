#!/bin/bash
# WildReceipt addendum verification ON Trillium (spec #11: no training until
# checksum + bundle verification pass). Run via `ssh ... 'bash -s'`.
set -uo pipefail
# Checksum + extraction already verified OK on 2026-07-05; re-running both is
# harmless (idempotent) but kept for one-command reproducibility.
cd /scratch/hamdan/CaliDoc/data
echo "=== addendum checksum (must say OK) ==="
echo "fef6f0e976e005a97422a8b25936db99682ca7159904b601988c7e4d9caaeb8f  wildreceipt_addendum_20260705.tar" | sha256sum -c - || { echo HASH_MISMATCH; exit 1; }
tar -xf wildreceipt_addendum_20260705.tar
echo "=== bundle addendum contents ==="
ls data_bundle/wildreceipt/processed/ data_bundle/splits/ | head -20

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
from calidoc.data import verify_bundle, WildReceiptDataset, ProtocolViolation, conformal_split
B = '/scratch/hamdan/CaliDoc/data/data_bundle'
m = verify_bundle(B, 'wildreceipt')
assert m['split_sizes'] == {'train': 1167, 'val_es': 100, 'test': 472}, m['split_sizes']
assert m['num_labels'] == 49
print("splits:", m['split_sizes'], "| O:", {k: round(v,4) for k,v in m['o_fraction_per_source_split'].items()})
try:
    WildReceiptDataset(B, 'test')  # fit stage must refuse test
    raise SystemExit("GUARD MISSING")
except ProtocolViolation:
    print("stage guard OK")
cal42, ev42 = conformal_split(B, 'wildreceipt', 42)
assert (len(cal42), len(ev42)) == (330, 142)
enc = WildReceiptDataset(B, 'val_es')[0]
print("val_es[0] shapes:", {k: tuple(v.shape) for k, v in enc.items()})
# 49-label head cache check for offline compute nodes
import os
os.environ['HF_HUB_OFFLINE'] = '1'; os.environ['TRANSFORMERS_OFFLINE'] = '1'
from transformers import LayoutLMv3ForTokenClassification
LayoutLMv3ForTokenClassification.from_pretrained("microsoft/layoutlmv3-large", num_labels=49)
print("offline -large load with 49 labels OK")
print("WILDRECEIPT_TRILLIUM_VERIFY_OK")
EOF
echo "=== queue ==="
squeue -u hamdan
ssh -n trig-login01 squeue -u hamdan 2>/dev/null || true
echo "=== done ==="
