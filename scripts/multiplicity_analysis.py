#!/usr/bin/env python3
"""Multiplicity control and dependence robustness for the boundary-escape audit.

Reads the committed boundary-escape results and reports, for every eligible
field type:

  * raw Monte-Carlo p-value,
  * Holm-adjusted p (FWER, arbitrary dependence)  -- PRIMARY,
  * BH-adjusted q  (FDR, requires independence/PRDS),
  * BY-adjusted q  (FDR, arbitrary dependence),
  * the document-blocked cluster-bootstrap 90% CI for edge enrichment.

Two family definitions are evaluated, because "the family" is a modelling
choice a reviewer can reasonably contest:

  per-dataset  each dataset audited on its own (the deployment scenario: a
               practitioner auditing one corpus). This is what the paper
               reports.
  pooled       one family over all eligible fields of the four benchmark
               datasets (the most conservative reading of "how many tests did
               these authors run"). SROIE-repaired is a separate lineage and
               is excluded from the pooled family, and reported on its own.

Writes results/<tag>/multiplicity_analysis.json and prints the table.
"""

from __future__ import annotations

import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from span_crc import (DEFAULT_RESULTS, adjusted_pvalues, bh_reject,
                      by_reject, holm_reject)

GAMMA = 0.10
BENCH = ['sroie', 'cord', 'funsd', 'wildreceipt']      # pooled family
ALL = BENCH + ['sroie_repaired']


def load_fields(ds):
    j = json.load(open(os.path.join(DEFAULT_RESULTS, f'span_crc_{ds}.json')))
    return j['per_unit'][0]['boundary_escape']['fields']


def main():
    out = {'gamma': GAMMA, 'primary_procedure': 'holm',
           'per_dataset': {}, 'pooled': {}, 'stability_across_splits': {}}

    # ---------------------------------------------------- per-dataset family
    print(f'{"dataset":<16}{"field":<26}{"p":>9}{"holm":>9}{"BH":>9}{"BY":>9}'
          f'{"enrich":>8}{"boot CI90":>16}  decision')
    print('-' * 112)
    for ds in ALL:
        fields = load_fields(ds)
        pv = {f: d['p_value'] for f, d in fields.items()
              if d['p_value'] is not None}
        adj = adjusted_pvalues(pv)
        dec = {'holm': holm_reject(pv, GAMMA), 'bh': bh_reject(pv, GAMMA),
               'by': by_reject(pv, GAMMA)}
        rows = {}
        for f in sorted(pv, key=lambda k: pv[k]):
            d, a = fields[f], adj[f]
            cb = d.get('cluster_bootstrap')
            ci = (f"[{cb['ci90_lo']:.2f},{cb['ci90_hi']:.2f}]" if cb else '---')
            verdict = ('REJECT' if dec['holm'][f] else
                       ('bh-only' if dec['bh'][f] else '-'))
            print(f'{ds:<16}{f:<26}{a["p"]:>9.4f}{a["holm"]:>9.4f}'
                  f'{a["bh"]:>9.4f}{a["by"]:>9.4f}'
                  f'{d["edge_enrichment"]:>8.2f}{ci:>16}  {verdict}')
            # NB: keys 'holm'/'bh'/'by' hold ADJUSTED P-VALUES (from
            # adjusted_pvalues); the decisions live under 'reject_*' so a
            # boolean can never shadow a p-value.
            rows[f] = {**a, 'enrichment': d['edge_enrichment'],
                       'n_informative_spans': d['n_informative_spans'],
                       'n_documents_informative': d.get('n_documents_informative'),
                       'cluster_bootstrap': cb,
                       'reject_holm': bool(dec['holm'][f]),
                       'reject_bh': bool(dec['bh'][f]),
                       'reject_by': bool(dec['by'][f])}
        out['per_dataset'][ds] = {
            'n_eligible': len(pv),
            'n_present': len(fields),
            'rejected_holm': sorted([f for f in pv if dec['holm'][f]]),
            'rejected_bh': sorted([f for f in pv if dec['bh'][f]]),
            'rejected_by': sorted([f for f in pv if dec['by'][f]]),
            'fields': rows,
        }
        print('-' * 112)

    # --------------------------------------------------------- pooled family
    pooled = {}
    for ds in BENCH:
        for f, d in load_fields(ds).items():
            if d['p_value'] is not None:
                pooled[f'{ds}:{f}'] = d['p_value']
    adj = adjusted_pvalues(pooled)
    dec = {'holm': holm_reject(pooled, GAMMA), 'bh': bh_reject(pooled, GAMMA),
           'by': by_reject(pooled, GAMMA)}
    out['pooled'] = {
        'n_eligible': len(pooled),
        'rejected_holm': sorted([k for k in pooled if dec['holm'][k]]),
        'rejected_bh': sorted([k for k in pooled if dec['bh'][k]]),
        'rejected_by': sorted([k for k in pooled if dec['by'][k]]),
        'fields': {k: {**adj[k], 'reject_holm': bool(dec['holm'][k]),
                       'reject_bh': bool(dec['bh'][k]),
                       'reject_by': bool(dec['by'][k])}
                   for k in pooled},
    }
    print(f'\nPOOLED family over the four benchmark datasets '
          f'({len(pooled)} eligible fields):')
    for k in sorted(pooled, key=lambda k: pooled[k])[:5]:
        a = adj[k]
        print(f'  {k:<40} p={a["p"]:.4f} holm={a["holm"]:.4f} '
              f'bh={a["bh"]:.4f} by={a["by"]:.4f} '
              f'{"REJECT" if dec["holm"][k] else "-"}')
    print(f'  Holm rejects {len(out["pooled"]["rejected_holm"])}, '
          f'BH {len(out["pooled"]["rejected_bh"])}, '
          f'BY {len(out["pooled"]["rejected_by"])}')

    # ------------------------------------------- stability across cal splits
    print('\nStability of the decision across calibration splits (Holm):')
    for ds in ALL:
        j = json.load(open(os.path.join(DEFAULT_RESULTS, f'span_crc_{ds}.json')))
        per = []
        for u in j['per_unit']:
            be = u['boundary_escape']
            pv = {f: d['p_value'] for f, d in be['fields'].items()
                  if d['p_value'] is not None}
            per.append(sorted([f for f, r in holm_reject(pv, GAMMA).items() if r]))
        out['stability_across_splits'][ds] = per
        same = all(p == per[0] for p in per)
        print(f'  {ds:<16} {per}  -> {"identical" if same else "VARIES"}')

    dst = os.path.join(DEFAULT_RESULTS, 'multiplicity_analysis.json')
    json.dump(out, open(dst, 'w'), indent=2)
    print(f'\nwrote {dst}')


if __name__ == '__main__':
    main()
