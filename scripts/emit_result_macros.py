#!/usr/bin/env python3
"""Emit every headline number in the paper as a LaTeX macro, from the JSONs.

The manuscript must not contain a hand-typed value that already exists in a
machine-readable artifact. This script writes
`docs/paper-wacv2027/generated_results_macros.tex`, a list of \\newcommand
definitions consumed by both the main paper and the supplementary, plus
`docs/paper-wacv2027/spancrc_table.tex` (the main SpanCRC results table) and
`docs/paper-wacv2027/backbone_table.tex` (the compact backbone F1 table).

check_gates.py re-runs this script into a temporary file and fails the build
if the regenerated macros differ from the committed ones, so a stale number
cannot survive a rebuild.

    python3 scripts/emit_result_macros.py
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent
RD = REPO / 'results' / 'local_20260803'
OUT = REPO / 'docs' / 'paper-wacv2027'
OUT.mkdir(parents=True, exist_ok=True)
NICE = {'sroie': 'SROIE', 'sroie_repaired': 'SROIE-rep.', 'cord': 'CORD',
        'funsd': 'FUNSD', 'wildreceipt': 'WildReceipt'}
ORDER = ['sroie', 'sroie_repaired', 'wildreceipt', 'funsd', 'cord']


def load(name):
    return json.loads((RD / name).read_text())


M: dict[str, str] = {}


def mac(name: str, value: str) -> None:
    """Register a macro; names must be alphabetic (TeX csname restriction)."""
    assert re.fullmatch(r'[A-Za-z]+', name), name
    M[name] = value


# ------------------------------------------------------------------ SpanCRC
span = {ds: load(f'span_crc_{ds}.json') for ds in ORDER}
for ds in ORDER:
    a = span[ds]['aggregate']
    tag = ds.replace('_', '')
    mac(f'{tag}SpanCovTok', f"{a['span_cov_token_marginal']['mean']:.3f}")
    mac(f'{tag}SpanCovFld', f"{a['span_cov_field_crc']['mean']:.3f}")
    mac(f'{tag}SpanCovFldSd', f"{a['span_cov_field_crc']['std']:.3f}")
    mac(f'{tag}CovGapTok', f"{a['span_covgap_pp_token_marginal']['mean']:.1f}")
    mac(f'{tag}CovGapFld', f"{a['span_covgap_pp_field_crc']['mean']:.1f}")
    mac(f'{tag}SizeTok', f"{a['avg_set_size_token_marginal']['mean']:.2f}")
    mac(f'{tag}SizeFld', f"{a['avg_set_size_field_crc']['mean']:.2f}")
    mac(f'{tag}RiskTok', f"{a['ev_doc_risk_token_marginal']['mean']:.3f}")
    mac(f'{tag}RiskFld', f"{a['ev_doc_risk_field_crc']['mean']:.3f}")
    st, sf = (a['avg_set_size_token_marginal']['mean'],
              a['avg_set_size_field_crc']['mean'])
    mac(f'{tag}SizePct', f"{(sf - st) / st * 100:+.0f}")
    mac(f'{tag}SizeMult', f"{sf / st:.1f}")
    mac(f'{tag}NEvSpans', f"{span[ds]['per_unit'][0]['n_ev_spans']:,}")

# ---------------------------------------------------------- boundary escape
tot_elig = 0
for ds in ORDER:
    be = span[ds]['per_unit'][0]['boundary_escape']
    tag = ds.replace('_', '')
    mac(f'{tag}NEligible', str(be['n_fields_tested']))
    mac(f'{tag}NPresent', str(len(be['fields'])))
    if ds in ('cord', 'funsd', 'wildreceipt'):
        tot_elig += be['n_fields_tested']
mac('cleanNEligible', str(tot_elig))
mac('mcFloor', f"{1 / (span['sroie']['per_unit'][0]['boundary_escape']['n_boot'] + 1):.5f}")
mac('nBoot', f"{span['sroie']['per_unit'][0]['boundary_escape']['n_boot']:,}")

mult = load('multiplicity_analysis.json')
for ds, fld, tag in (('sroie', 'ADDRESS', 'sroieAddr'),
                     ('sroie', 'COMPANY', 'sroieComp'),
                     ('sroie_repaired', 'ADDRESS', 'sroierepAddr'),
                     ('funsd', 'QUESTION', 'funsdQuest')):
    r = mult['per_dataset'][ds]['fields'][fld]
    mac(f'{tag}Enr', f"{r['enrichment']:.2f}")
    mac(f'{tag}P', f"{r['p']:.4f}")
    mac(f'{tag}Holm', f"{r['holm']:.4f}")
    mac(f'{tag}BH', f"{r['bh']:.4f}")
    mac(f'{tag}BY', f"{r['by']:.4f}")
    cb = r['cluster_bootstrap']
    if cb:
        mac(f'{tag}CIlo', f"{cb['ci90_lo']:.2f}")
        mac(f'{tag}CIhi', f"{cb['ci90_hi']:.2f}")
        mac(f'{tag}NDoc', str(cb['n_documents']))
mac('pooledNEligible', str(mult['pooled']['n_eligible']))
mac('pooledNRejectHolm', str(len(mult['pooled']['rejected_holm'])))
mac('pooledNRejectBH', str(len(mult['pooled']['rejected_bh'])))
mac('pooledNRejectBY', str(len(mult['pooled']['rejected_by'])))
mac('pooledAddrHolm',
    f"{mult['pooled']['fields']['sroie:ADDRESS']['holm']:.4f}")
mac('pooledAddrBY', f"{mult['pooled']['fields']['sroie:ADDRESS']['by']:.4f}")

# ----------------------------------------------------------- size and power
pw = load('boundary_escape_power.json')
for ds in ('funsd', 'wildreceipt', 'cord'):
    s = pw['size'][ds]
    tag = ds.replace('_', '')
    for proc in ('holm', 'bh', 'by'):
        mac(f'{tag}SizeRate{proc.upper()}',
            f"{s['familywise_rejection_rate_by_procedure'][proc]:.3f}")
    mac(f'{tag}SizeRateUncorr',
        f"{s['familywise_rejection_rate_uncorrected']:.3f}")
mac('nullReps', str(pw['null_replicates']))
mac('nCorruptSeeds', str(len(pw['corrupt_seeds'])))


def power_at(variant, ds, field, eps):
    cells = [c for c in pw['power'][variant][ds][field] if c['eps'] == eps]
    return (float(np.mean([c['detected'] for c in cells])),
            cells[0]['n_corrupted'],
            float(np.mean([len(c['false_positive_fields']) for c in cells])))


for eps, tag in ((0.02, 'Two'), (0.05, 'Five'), (0.10, 'Ten'), (0.50, 'Fifty')):
    p, n, fp = power_at('end_extend', 'funsd', 'QUESTION', eps)
    mac(f'funsdQuestPower{tag}', f"{p:.2f}")
    mac(f'funsdQuestNCorr{tag}', str(n))
mac('wildrecPowerMax', f"{max(power_at('end_extend', 'wildreceipt', f, e)[0] for f in pw['power']['end_extend']['wildreceipt'] for e in pw['doses']):.2f}")
mac('startShiftPowerMax', f"{max(power_at('start_shift', 'funsd', f, e)[0] for f in pw['power']['start_shift']['funsd'] for e in pw['doses']):.2f}")

# ------------------------------------------------------- first-token vs span
ft = load('first_token_entity_cp.json')
for r in ft['runs']:
    a, tag = r['aggregate'], r['dataset'].replace('_', '')
    suf = 'Two' if abs(r['alpha'] - 0.02) < 1e-9 else 'Ten'
    mac(f'{tag}FTCov{suf}', f"{a['first_token_coverage']['mean']:.3f}")
    mac(f'{tag}FTCovSd{suf}', f"{a['first_token_coverage']['std']:.3f}")
    mac(f'{tag}FTSize{suf}', f"{a['first_token_avg_set_size']['mean']:.2f}")
    mac(f'{tag}FTSingle{suf}',
        f"{a['first_token_singleton_rate']['mean'] * 100:.1f}")
    mac(f'{tag}FTFullSpan{suf}',
        f"{a['full_span_coverage_same_threshold']['mean']:.3f}")

# --------------------------------------------------------------- backbone F1
ms = load('multiseed_summary.json')['datasets']
for ds in ('cord', 'funsd', 'wildreceipt', 'sroie'):
    ps = ms[ds]['per_seed']
    tag = ds.replace('_', '')
    for key, name in (('f1', 'F'), ('ece', 'ECE'), ('mce', 'MCE')):
        v = [ps[s][key] for s in ps]
        mac(f'{tag}{name}', f"{np.mean(v):.2f}")
        mac(f'{tag}{name}Sd', f"{np.std(v):.2f}")
    mac(f'{tag}NSeeds', str(len(ps)))
cf1 = sorted(ms['cord']['per_seed'][s]['f1'] for s in ms['cord']['per_seed'])
mac('cordFSpread', f"{cf1[-1] - cf1[0]:.2f}")
mac('cordFSeedList', '/'.join(f'{v:.2f}' for v in cf1))
lac = [ms['cord']['per_seed'][s]['lac_cov_alpha_0.1'] for s in ms['cord']['per_seed']]
mac('cordLacSpread', f"{max(lac) - min(lac):.2f}")
for ds in ('cord', 'funsd', 'wildreceipt'):
    ps = ms[ds]['per_seed']
    tag = ds.replace('_', '')
    for a, suf in (('0.1', 'Ninety'), ('0.05', 'NinetyFive'), ('0.01', 'NinetyNine')):
        mac(f'{tag}LacCov{suf}',
            f"{np.mean([ps[s][f'lac_cov_alpha_{a}'] for s in ps]):.1f}")
        mac(f'{tag}LacSize{suf}',
            f"{np.mean([ps[s][f'lac_size_alpha_{a}'] for s in ps]):.2f}")

# ------------------------------------------------------- SROIE per-class / q
cc = load('conditional_cp_sroie.json')['per_unit']
marg = [u['alpha_0.1']['strategies']['marginal'] for u in cc]
mond = [u['alpha_0.1']['strategies']['classwise_mondrian'] for u in cc]
mac('sroieTokenCov', f"{np.mean([m['marginal_coverage'] for m in marg]) * 100:.1f}")
# the single marginal threshold: every class shares it under `marginal`
mac('sroieQhatMarg',
    f"{np.mean([list(m['per_class'].values())[0]['qhat'] for m in marg]):.4f}")
mac('sroieCovGapMarg', f"{np.mean([m['conditional_summary']['covgap_pp'] for m in marg]):.1f}")
mac('sroieCovGapMond', f"{np.mean([m['conditional_summary']['covgap_pp'] for m in mond]):.1f}")
mac('sroieSizeMond', f"{np.mean([m['avg_set_size'] for m in mond]):.2f}")
qs = {c: np.mean([m['per_class'][c]['qhat'] for m in mond])
      for c in mond[0]['per_class']}
mac('sroieQhatMin', f"{min(qs.values()):.4f}")
mac('sroieQhatMax', f"{max(qs.values()):.4f}")
mac('sroieQhatRatio', f"{max(qs.values()) / min(qs.values()):.0f}")
covs = {c: np.mean([m['per_class'][c]['coverage'] for m in marg])
        for c in marg[0]['per_class']}
mac('sroieNZeroClasses', str(sum(1 for v in covs.values() if v == 0.0)))
mac('sroieNClasses', str(len(covs)))
ns = {c: np.mean([m['per_class'][c]['n_eval'] for m in marg])
      for c in marg[0]['per_class']}
mac('sroieOShare', f"{ns['O'] / sum(ns.values()) * 100:.0f}")
mondcov = {c: np.mean([m['per_class'][c]['coverage'] for m in mond])
           for c in mond[0]['per_class']}
mac('sroieMondCovMin', f"{min(mondcov.values()) * 100:.1f}")
mac('sroieMondCovMax', f"{max(mondcov.values()) * 100:.1f}")

ccr = load('conditional_cp_sroie_repaired.json')['per_unit']
margr = [u['alpha_0.1']['strategies']['marginal'] for u in ccr]
mac('sroierepCovGapMarg',
    f"{np.mean([m['conditional_summary']['covgap_pp'] for m in margr]):.1f}")
for c, tag in (('B-ADDRESS', 'BAddr'), ('I-ADDRESS', 'IAddr'), ('B-TOTAL', 'BTotal')):
    mac(f'sroie{tag}CovStd',
        f"{np.mean([m['per_class'][c]['coverage'] for m in marg]):.3f}")
    mac(f'sroie{tag}CovRep',
        f"{np.mean([m['per_class'][c]['coverage'] for m in margr]):.3f}")

# ------------------------------------------------------------- CORD schema
ccc = load('conditional_cp_cord.json')['per_unit'][0]['alpha_0.1']
# three CORD classes have no evaluation tokens and carry only `n_eval`;
# "evaluated classes" therefore means those with a coverage estimate.
_cpc = ccc['strategies']['marginal']['per_class']
_ev = {k: v for k, v in _cpc.items() if 'coverage' in v}
mac('cordNEvalTokens', f"{sum(v['n_eval'] for v in _ev.values()):,}")
mac('cordNClassesPresent', str(len(_cpc)))
mac('cordNClasses', str(len(_ev)))
mac('cordNZeroClasses',
    str(sum(1 for v in _ev.values() if v['coverage'] == 0.0)))
mac('cordCovGapMarg',
    f"{ccc['strategies']['marginal']['conditional_summary']['covgap_pp']:.1f}")
mac('cordCovGapClustered',
    f"{ccc['strategies']['clustered_k3']['conditional_summary']['covgap_pp']:.1f}")
mac('cordCovGapInterp',
    f"{ccc['strategies']['interpolated_lambda0.75']['conditional_summary']['covgap_pp']:.1f}")
mac('cordCovGapMond',
    f"{ccc['strategies']['classwise_mondrian']['conditional_summary']['covgap_pp']:.1f}")
mac('cordSizeMond', f"{ccc['strategies']['classwise_mondrian']['avg_set_size']:.2f}")
mac('cordSizeMarg', f"{ccc['strategies']['marginal']['avg_set_size']:.2f}")
mac('cordFeasBound', str(ccc['feasibility_bound_n']))
mac('cordNBelowFeas', str(ccc['n_classes_below_feasibility']))
mn = ccc['matched_n_control']
mac('matchedNCovs', '/'.join(
    f"{mn[f'fixed_cohort_n_{n}']['mean_class_conditional_coverage']:.3f}"
    for n in (5, 10, 20, 50, 100)))

# ----------------------------------------------------- SROIE alignment gate
txt = (REPO / 'results' / 'sroie_alignment_diagnosis_20260810.txt').read_text()


def block(label):
    m = re.search(rf'===== {label}[^=]*?=====\n(.*?)(?:\n  by field:|\n=====|\Z)',
                  txt, re.S)
    d = {}
    for line in m.group(1).strip().splitlines():
        g = re.match(r'\s*(\w+)\s+(\d+)\s+([\d.]+)%\s+cumulative\s+([\d.]+)%', line)
        if g:
            d[g.group(1)] = (int(g.group(2)), float(g.group(3)), float(g.group(4)))
    return d


comb = block('combined')
n_all = int(re.search(r'===== combined: (\d+) field instances', txt).group(1))
mac('sroieNInstances', f'{n_all:,}')
for cause, tag in (('strict', 'Strict'), ('punctuation', 'Punct'),
                   ('numeric', 'Numeric'), ('noncontiguous', 'Noncontig'),
                   ('unmatched', 'Unmatched')):
    n, pct, cum = comb[cause]
    mac(f'sroieAlign{tag}N', f'{n:,}')
    mac(f'sroieAlign{tag}Pct', f'{pct:.2f}')
    mac(f'sroieAlign{tag}Cum', f'{cum:.2f}')
for split, tag in (('train', 'Train'), ('test', 'Test')):
    m = re.search(rf'===== {split}: (\d+) field instances over (\d+) docs =====\n'
                  r'\s*strict\s+(\d+)\s+([\d.]+)%', txt)
    mac(f'sroieGate{tag}Pct', f'{float(m.group(4)):.2f}')
    mac(f'sroieGate{tag}N', f'{int(m.group(3)):,}')
    mac(f'sroieGate{tag}Tot', f'{int(m.group(1)):,}')
m = re.search(r'===== test.*?by field:\n(.*?)\n  examples', txt, re.S)
for line in m.group(1).strip().splitlines():
    g = re.match(r'\s*(\w+)\s+n=\s*(\d+)\s+strict\s+([\d.]+)%', line)
    if g:
        mac(f'sroieField{g.group(1).capitalize()}Strict', f'{float(g.group(3)):.1f}')

# ---------------------------------------------------------------- m0 sweep
sens = load('spancrc_sensitivity.json')
d = sens['datasets']['cord']
mac('cordSizeMzeroTen', f"{d['10']['avg_set_size_candidate_rule']['mean']:.2f}")
mac('cordSizeMzeroFifty', f"{d['50']['avg_set_size_candidate_rule']['mean']:.2f}")
covs_m0 = [d[k]['span_coverage']['mean'] for k in ('10', '15', '25', '50')]
mac('cordMzeroCovRange', f"{(max(covs_m0) - min(covs_m0)) * 100:.1f}")
other = [sens['datasets'][x][k]['span_coverage']['mean']
         for x in ('funsd', 'wildreceipt', 'sroie', 'sroie_repaired')
         for k in ('10', '15', '25', '50')]
rng_other = max(
    max(sens['datasets'][x][k]['span_coverage']['mean'] for k in ('10', '15', '25', '50')) -
    min(sens['datasets'][x][k]['span_coverage']['mean'] for k in ('10', '15', '25', '50'))
    for x in ('funsd', 'wildreceipt', 'sroie', 'sroie_repaired'))
mac('otherMzeroCovRange', f"{rng_other * 100:.1f}")
mac('mzero', str(span['sroie']['per_unit'][0]['field_crc']['min_cal_spans']))

# ------------------------------------------------------------------ ablations
abl_c = load('ablations_cord.json')
mac('cordEmptyRate',
    f"{abl_c['per_unit'][0]['empty_sets']['alpha_0.1']['empty_set_rate'] * 100:.1f}")
mac('cordEmptyCorrect',
    f"{abl_c['per_unit'][0]['empty_sets']['alpha_0.1']['accuracy_on_empty'] * 100:.1f}")
mac('cordApsSize',
    f"{abl_c['per_unit'][0]['scoring_ablation']['aps']['alpha_0.1']['avg_set_size']:.1f}")
mac('cordMcEce', f"{abl_c['mc_dropout']['ece']:.2f}")
mac('cordMcCov',
    f"{abl_c['mc_dropout']['conformal_on_mc_probs']['alpha_0.1']['coverage'] * 100:.1f}")
mac('cordMcPasses', str(abl_c['mc_dropout']['n_passes']))
abl_f = load('ablations_funsd.json')['per_class_f1']
for c in ('HEADER', 'ANSWER', 'QUESTION'):
    mac(f'funsd{c.capitalize()}F', f"{abl_f[c]['f1-score']:.3f}")
    mac(f'funsd{c.capitalize()}Sup', str(int(abl_f[c]['support'])))

# ------------------------------------------------------------- noise dose
nd = load('noise_dose_summary.json')['by_model']['class_targeted']
k0, k2 = '0.0', '0.02'
mac('noiseFlipped', f"{nd[k2]['n_flipped_train_tokens']['mean']:.0f}")
mac('noiseNSeeds', str(nd[k2]['n_seeds']))
mac('noiseTotalTokens', '19,367')
mac('noiseCovClean',
    f"{nd[k0]['target_class_coverage_marginal']['mean'] * 100:.1f}")
mac('noiseCovDirty',
    f"{nd[k2]['target_class_coverage_marginal']['mean'] * 100:.1f}")
mac('noiseFOne', f"{nd[k0]['test_f1']['mean'] * 100:.2f}")
mac('noiseFTwo', f"{nd[k2]['test_f1']['mean'] * 100:.2f}")

# ---------------------------------------------------------- perturbations
pf = load('perturbation_coverage_funsd.json')['summary']
mac('robustFunsdFDrop', f"{pf['mean_f1_drop_pp']:.2f}")
mac('robustFunsdFDropMax', f"{pf['max_f1_drop_pp']:.2f}")
worst = max(load(f'perturbation_coverage_{d}.json')['summary']['max_coverage_drop_pp']
            for d in ('cord', 'funsd', 'wildreceipt', 'sroie'))
mac('robustWorstCovDrop', f"{worst:.2f}")
mac('robustNCells', str(load('perturbation_coverage_cord.json')['summary']['n_cells']))

# ------------------------------------------------------------ cross-conformal
# cross-conformal runs on the seed-42 checkpoint, so it must be compared
# against the seed-42 fixed-split coverage, not the 3-seed mean.
cc5 = load('phase2_cord.json')['results']['cross_conformal_k5']['lac']
_s42 = ms['cord']['per_seed']['42']
mx = max(abs(cc5[f'alpha_{a}']['mean'] * 100 - _s42[f'lac_cov_alpha_{a}'])
         for a in ('0.1', '0.05', '0.01'))
mac('crossConfMaxDelta', f"{mx:.2f}")

# --------------------------------------------------------------- temperature
for ds in ('cord', 'funsd', 'wildreceipt', 'sroie'):
    t = json.loads((REPO / 'results' / 'cache' /
                    f'{ds}_local_20260803.json').read_text())['temperature']
    mac(f'{ds}Temp', f'{t:.2f}')

# ================================================== review-response additions
# Span-length stratification, aggregation ablation, field-support economics,
# operational set sizes (results/local_20260803/span_length_analysis.json).
sla = load('span_length_analysis.json')['datasets']
for ds in ORDER:
    u0 = sla[ds]['per_unit'][0]
    tag = ds.replace('_', '')
    ls = u0['length_stats_ev']
    mac(f'{tag}LenMedian', f"{ls['median']:.0f}")
    mac(f'{tag}LenPninety', f"{ls['p90']:.0f}")
    mac(f'{tag}LenMax', str(ls['max']))
    mac(f'{tag}SpearmanLenScore',
        f"{u0['correlations']['spearman_length_vs_max_score']:.2f}")
    agg = u0['aggregation_ablation']
    mac(f'{tag}AggMeanOwn', f"{agg['mean']['own_event_rate']:.3f}")
    mac(f'{tag}AggMeanTrueCov', f"{agg['mean']['true_full_span_coverage']:.3f}")
    mac(f'{tag}AggMeanTrueRisk', f"{agg['mean']['true_doc_field_miss_risk']:.3f}")
    mac(f'{tag}AggMaxTrueRisk', f"{agg['max']['true_doc_field_miss_risk']:.3f}")
    op = u0['operational_set_size']['field_crc']
    mac(f'{tag}OpSingleton', f"{op['singleton_rate'] * 100:.1f}")
    mac(f'{tag}OpMulti', f"{op['multi_rate'] * 100:.1f}")
    mac(f'{tag}OpEmpty', f"{op['empty_rate'] * 100:.1f}")
    mac(f'{tag}OpPninety', f"{op['p90']:.0f}")
    mac(f'{tag}OpPninetyfive', f"{op['p95']:.0f}")
    mac(f'{tag}OpSpanAllSingle', f"{op['span_all_singleton_rate'] * 100:.1f}")
    mac(f'{tag}OpSpanAnyMulti', f"{op['span_any_multi_rate'] * 100:.1f}")
# CORD support economics
u0c = sla['cord']['per_unit'][0]
fs = u0c['field_support']
mac('cordPctSpansFallback',
    f"{fs['frac_ev_spans_in_fallback_fields'] * 100:.1f}")
mac('cordNFieldsOwn', str(fs['n_fields_own_threshold']))
mac('cordNFieldsFallback', str(fs['n_fields_fallback']))
mac('cordNFieldsTotal', str(fs['n_field_types']))
# length-bucket extremes quoted in the text (CORD long-span cost,
# FUNSD long-span eligibility)
mac('cordLongBucketCov',
    f"{u0c['stratified_reliability']['field_crc']['5-8']['full_span_coverage']:.3f}")
mac('cordLongBucketN',
    str(u0c['stratified_reliability']['field_crc']['5-8']['n_spans']))
u0f = sla['funsd']['per_unit'][0]
mac('funsdLongBucketCov',
    f"{u0f['stratified_reliability']['field_crc']['>=9']['full_span_coverage']:.3f}")
mac('funsdLongElig',
    f"{u0f['boundary_audit_eligibility_by_length']['by_bucket']['>=9']['informative_rate'] * 100:.0f}")
# SROIE alpha worked example: mean gold fields per evaluation document
sroie_spans = load('span_crc_sroie.json')['per_unit'][0]['n_ev_spans']
mac('sroieSpansPerDoc', '4')  # 420 spans / 105 docs = 4.0 exactly
assert sroie_spans == 420

# SpanCRC-level interpolation (results/local_20260803/spancrc_interpolation.json)
interp = load('spancrc_interpolation.json')['datasets']['cord']['per_unit'][0]
_rules = {r['rule']: r for r in interp['rules']}
_mid = _rules['interp_tau_0.50']
mac('cordInterpCovGapMid', f"{_mid['overall']['span_covgap_pp']:.1f}")
mac('cordInterpCovMid', f"{_mid['overall']['full_span_coverage']:.3f}")
mac('cordInterpRiskMid', f"{_mid['overall']['doc_field_miss_risk']:.3f}")
mac('cordInterpSizeMid', f"{_mid['avg_set_size']:.2f}")
mac('cordInterpEmptyMid', f"{_mid['empty_set_rate'] * 100:.1f}")
_nofb = _rules['field_crc_no_fallback']
mac('cordNoFallbackCovGap', f"{_nofb['overall']['span_covgap_pp']:.1f}")
mac('cordNoFallbackSize', f"{_nofb['avg_set_size']:.2f}")
_ci = _rules['crc_interp_tau_0.50']
mac('cordCrcInterpCovGapMid', f"{_ci['overall']['span_covgap_pp']:.1f}")
mac('cordCrcInterpSizeMid', f"{_ci['avg_set_size']:.2f}")
mac('cordCrcInterpRiskMid', f"{_ci['overall']['doc_field_miss_risk']:.3f}")
mac('cordCrcInterpCovMid', f"{_ci['overall']['full_span_coverage']:.3f}")
_fi = {r['rule']: r for r in json.loads(
    (RD / 'spancrc_interpolation.json').read_text())['datasets']['funsd']
    ['per_unit'][0]['rules']}['crc_interp_tau_0.50']
mac('funsdCrcInterpSizeMid', f"{_fi['avg_set_size']:.2f}")
mac('funsdCrcInterpCovMid', f"{_fi['overall']['full_span_coverage']:.3f}")

# Generalized position masks (results/local_20260803/boundary_escape_masks.json)
masks = load('boundary_escape_masks.json')['datasets']
mac('maskFamilyTotal',
    str(sum(v['n_tests_in_family'] for v in masks.values())))
_rej = sorted(sum((v['rejected_pairs_holm'] for v in masks.values()), []))
assert _rej == ['ADDRESS::edges', 'COMPANY::edges'], _rej
mac('maskNRejected', str(len(_rej)))


# ------------------------------- GPU review-response additions (conditional)
# Emitted only when the artifacts exist so the regeneration gate is
# deterministic before and after the runs land.

def _entity_f1_from_cache(cache_stem: str) -> float:
    """Entity-level micro-F1 on the full test unit, computed from the cached
    probabilities (argmax) and gold labels -- artifact-derived, no re-eval."""
    import numpy as _np
    from seqeval.metrics import f1_score as _f1
    z = _np.load(REPO / 'results' / 'cache' / f'{cache_stem}.npz')
    meta = json.loads((REPO / 'results' / 'cache' / f'{cache_stem}.json').read_text())
    id2label = {int(k): v for k, v in meta['id2label'].items()}
    y, p, ds = z['test_labels'], z['test_probs'].argmax(1), z['test_docstart']
    gold, pred = [], []
    for d in range(len(ds) - 1):
        s, e = int(ds[d]), int(ds[d + 1])
        gold.append([id2label[int(t)] for t in y[s:e]])
        pred.append([id2label[int(t)] for t in p[s:e]])
    return float(_f1(gold, pred)) * 100


BB = REPO / 'results' / 'backbone_generality'
for ds in ('funsd', 'wildreceipt', 'sroie'):
    p = BB / f'span_crc_{ds}.json'
    if not p.exists():
        continue
    d = json.loads(p.read_text())
    a = d['aggregate']
    tag = f'base{ds.capitalize()}'
    mac(f'{tag}SpanCovTok', f"{a['span_cov_token_marginal']['mean']:.3f}")
    mac(f'{tag}SpanCovFld', f"{a['span_cov_field_crc']['mean']:.3f}")
    mac(f'{tag}RiskTok', f"{a['ev_doc_risk_token_marginal']['mean']:.3f}")
    mac(f'{tag}RiskFld', f"{a['ev_doc_risk_field_crc']['mean']:.3f}")
    mac(f'{tag}SizeTok', f"{a['avg_set_size_token_marginal']['mean']:.2f}")
    mac(f'{tag}SizeFld', f"{a['avg_set_size_field_crc']['mean']:.2f}")
    mac(f'{tag}CovGapTok', f"{a['span_covgap_pp_token_marginal']['mean']:.1f}")
    mac(f'{tag}CovGapFld', f"{a['span_covgap_pp_field_crc']['mean']:.1f}")
    be = d['per_unit'][0]['boundary_escape']
    mac(f'{tag}NRejected', str(be['n_fields_rejected']))
    mac(f'{tag}F', f"{_entity_f1_from_cache(ds + '_base_local_20260815'):.2f}")

# --- multi-seed Base (seeds 42/123/456) and multi-seed clean protocol -------
# Emitted only when all three seeds are present; mean +- std over TRAINING
# seeds (calibration-split mean within each seed, as everywhere else).
def _seed_paths(ds):
    return [BB / f'span_crc_{ds}.json',
            BB / 's123' / f'span_crc_{ds}.json',
            BB / 's456' / f'span_crc_{ds}.json']


for ds in ('sroie', 'funsd', 'wildreceipt'):
    paths = _seed_paths(ds)
    if not all(p.exists() for p in paths):
        continue
    aggs = [json.loads(p.read_text())['aggregate'] for p in paths]
    tag = f'base{ds.capitalize()}Ms'
    for key, name in (('span_cov_token_marginal', 'SpanCovTok'),
                      ('span_cov_field_crc', 'SpanCovFld'),
                      ('ev_doc_risk_token_marginal', 'RiskTok'),
                      ('ev_doc_risk_field_crc', 'RiskFld'),
                      ('span_covgap_pp_field_crc', 'CovGapFld'),
                      ('avg_set_size_field_crc', 'SizeFld')):
        v = [a[key]['mean'] for a in aggs]
        fmt = '.1f' if 'CovGap' in name else ('.2f' if 'Size' in name else '.3f')
        mac(f'{tag}{name}', f"{np.mean(v):{fmt}}")
        mac(f'{tag}{name}Sd', f"{np.std(v):{fmt}}")
    nrej = [json.loads(p.read_text())['per_unit'][0]['boundary_escape']
            ['n_fields_rejected'] for p in paths]
    mac(f'{tag}NRejectedAll', '/'.join(str(x) for x in nrej))
    if ds == 'sroie':
        rej_sets = [tuple(json.loads(p.read_text())['per_unit'][0]
                          ['boundary_escape']['rejected_fields']) for p in paths]
        mac('baseSroieMsRejectedIdentical',
            'yes' if len(set(rej_sets)) == 1 and rej_sets[0] == ('ADDRESS', 'COMPANY')
            else 'no')
mac('baseNSeeds', str(sum(1 for p in _seed_paths('sroie') if p.exists())))

TCD = REPO / 'results' / 'traincal'
_tc_paths = [TCD / f'b8_s{s}' / 'span_crc_wildreceipt_traincal.json'
             for s in (42, 123, 456)]
if all(p.exists() for p in _tc_paths):
    aggs = [json.loads(p.read_text())['aggregate'] for p in _tc_paths]
    for key, name in (('span_cov_token_marginal', 'SpanCovTok'),
                      ('span_cov_field_crc', 'SpanCovFld'),
                      ('ev_doc_risk_field_crc', 'RiskFld'),
                      ('span_covgap_pp_token_marginal', 'CovGapTok'),
                      ('span_covgap_pp_field_crc', 'CovGapFld'),
                      ('avg_set_size_field_crc', 'SizeFld')):
        v = [a[key]['mean'] for a in aggs]
        fmt = '.1f' if 'CovGap' in name else ('.2f' if 'Size' in name else '.3f')
        mac(f'traincalMs{name}', f"{np.mean(v):{fmt}}")
        mac(f'traincalMs{name}Sd', f"{np.std(v):{fmt}}")
    f1s = [_entity_f1_from_cache(f'wildreceipt_traincal_traincal_b8_s{s}_local_20260815')
           for s in (42, 123, 456)]
    mac('traincalMsF', f"{np.mean(f1s):.2f}")
    mac('traincalMsFSd', f"{np.std(f1s):.2f}")
    nrej = [json.loads(p.read_text())['per_unit'][0]['boundary_escape']
            ['n_fields_rejected'] for p in _tc_paths]
    mac('traincalMsNRejectedAll', '/'.join(str(x) for x in nrej))
mac('traincalNSeeds', str(sum(1 for p in _tc_paths if p.exists())))

TC = REPO / 'results' / 'traincal' / 'span_crc_wildreceipt_traincal.json'
if TC.exists():
    d = json.loads(TC.read_text())
    a = d['aggregate']
    mac('traincalSpanCovTok', f"{a['span_cov_token_marginal']['mean']:.3f}")
    mac('traincalSpanCovFld', f"{a['span_cov_field_crc']['mean']:.3f}")
    mac('traincalRiskTok', f"{a['ev_doc_risk_token_marginal']['mean']:.3f}")
    mac('traincalRiskFld', f"{a['ev_doc_risk_field_crc']['mean']:.3f}")
    mac('traincalSizeFld', f"{a['avg_set_size_field_crc']['mean']:.2f}")
    mac('traincalCovGapTok', f"{a['span_covgap_pp_token_marginal']['mean']:.1f}")
    mac('traincalCovGapFld', f"{a['span_covgap_pp_field_crc']['mean']:.1f}")
    be = d['per_unit'][0]['boundary_escape']
    mac('traincalNRejected', str(be['n_fields_rejected']))
    mac('traincalF',
        f"{_entity_f1_from_cache('wildreceipt_traincal_traincal_local_20260815'):.2f}")

# ---------------------------------- label-error detector ranges (Table S7)
# The main paper quotes AUPRC ranges and class-level ranks for the E2
# comparison; both are derived here so they cannot drift from the table.
_le = {}
for _p in sorted(RD.glob('label_error_baselines_*.json')):
    if 'LINEARPROBE' in _p.name:
        continue
    _d = json.loads(_p.read_text())
    _le[f"{_d['noise_model']}_{_d['eps']}"] = _d['detectors']
_ct = [k for k in _le if k.startswith('class_targeted')]
_base = ['cleanlab', 'icp_pvalue', 'aum', 'datamap_conf', 'datamap_var',
         'max_softmax']


def _auprc(det, runs):
    return [_le[k][det]['auprc'] for k in runs]


def _rank(det, runs):
    return [_le[k][det]['class_level']['rank_of_corrupted_class']
            for k in runs]


for _det, _tag in (('mondrian_gap', 'Ours'), ('cleanlab', 'Cleanlab'),
                   ('aum', 'Aum')):
    _v = _auprc(_det, _le)
    mac(f'le{_tag}AuprcLo', f'{min(_v):.3f}')
    mac(f'le{_tag}AuprcHi', f'{max(_v):.3f}')
_r = _rank('mondrian_gap', _ct)
mac('leOursRankLo', str(min(_r)))
mac('leOursRankHi', str(max(_r)))
_rb = [x for d in _base for x in _rank(d, _ct)]
mac('leBaseRankLo', str(min(_rb)))
mac('leBaseRankHi', str(max(_rb)))
mac('leNClasses',
    str(_le[_ct[0]]['mondrian_gap']['class_level']['n_classes']))
mac('leOursRankUniform',
    str(_rank('mondrian_gap', [k for k in _le if k.startswith('uniform')])[0]))
mac('leOursRankBioShift',
    str(_rank('mondrian_gap', [k for k in _le if k.startswith('bio_shift')])[0]))
mac('leOursBestRank', str(min(_r)))

# ------------------------------- seed-42 baselines for seed-42 ablations
# The MC-dropout ablation runs on the seed-42 checkpoint, so it must be
# compared against seed-42 ECE/coverage, not the three-seed means.
mac('cordEceSeed', f"{_s42['ece']:.2f}")
mac('cordLacCovSeed', f"{_s42['lac_cov_alpha_0.1']:.1f}")

# --------------------------------- SROIE marginal-threshold per-class detail
_scc = load('conditional_cp_sroie.json')['per_unit']
for _cls, _nm in (('O', 'sroieOCovMarg'), ('I-ADDRESS', 'sroieIAddrCovMarg')):
    _v = [u['alpha_0.1']['strategies']['marginal']['per_class'][_cls]['coverage']
          for u in _scc]
    mac(_nm, f'{100 * float(np.mean(_v)):.1f}')
_tokcov = [u['alpha_0.1']['strategies']['marginal']['marginal_coverage']
           for u in _scc]
mac('sroieTokenCovFrac', f'{float(np.mean(_tokcov)):.3f}')

# ==================================================================== write
hdr = ('% GENERATED by scripts/emit_result_macros.py -- do not edit by hand.\n'
       '% Every headline number in the paper is defined here, straight from\n'
       '% the committed result artifacts. check_gates.py fails if a rebuild\n'
       '% of this file differs from the committed copy.\n')
body = '\n'.join(f'\\newcommand{{\\{k}}}{{{v}}}' for k, v in sorted(M.items()))
(OUT / 'generated_results_macros.tex').write_text(hdr + body + '\n')

# ------------------------------------------------- main SpanCRC table
rows = []
for ds in ORDER:
    a = span[ds]['aggregate']
    tag = ds.replace('_', '')
    sd = (f"$\\pm${a['span_cov_field_crc']['std']:.3f}"
          if a['span_cov_field_crc']['std'] > 0 else '')
    rows.append(
        f"{NICE[ds]} & \\{tag}RiskTok & \\textbf{{\\{tag}RiskFld}} & "
        f"\\{tag}SpanCovTok & \\textbf{{\\{tag}SpanCovFld}}{sd} & "
        f"\\{tag}CovGapTok & \\textbf{{\\{tag}CovGapFld}} \\\\")
(OUT / 'spancrc_table.tex').write_text(
    "% GENERATED by scripts/emit_result_macros.py -- do not edit by hand.\n"
    r"""\begin{table}[t]
\centering
\caption{Token-marginal threshold versus field-conditional SpanCRC at the
$90\%$ target, mean over calibration splits. \textbf{Field-miss risk} is the
realized per-document fraction of gold fields missed---the quantity the
theorem bounds in expectation (at $\alpha=0.10$); the remaining columns are
empirical. A single-split realization may exceed $\alpha$, since the bound is
on its expectation over calibration draws.}
\label{tab:spancrc}
\footnotesize
\setlength{\tabcolsep}{2.6pt}
\begin{tabular}{lcccccc}
\toprule
& \multicolumn{2}{c}{Field-miss risk} & \multicolumn{2}{c}{Full-span cov.}
& \multicolumn{2}{c}{CovGap (pp)} \\
\cmidrule(lr){2-3}\cmidrule(lr){4-5}\cmidrule(lr){6-7}
Dataset & tok. & CRC & tok. & CRC & tok. & CRC \\
\midrule
""" + '\n'.join(rows) + r"""
\bottomrule
\end{tabular}
\end{table}
""")

# --------------------------------------- SROIE per-class table (main Tab. 2)
_order = sorted(covs, key=lambda c: -ns[c])
_rows = []
for c in _order:
    q = qs[c]
    bold = '\\textbf{' + f'{q:.4f}' + '}' if q in (max(qs.values()),
                                                  sorted(qs.values())[-3]) else f'{q:.4f}'
    _rows.append(f"\\texttt{{{c}}} & {ns[c]:,.0f} & {covs[c]*100:.1f}\\% & "
                 f"{mondcov[c]*100:.1f}\\% & {bold} \\\\")
_marg_m = np.mean([m['marginal_coverage'] for m in marg]) * 100
_mond_m = np.mean([m['marginal_coverage'] for m in mond]) * 100
(OUT / 'sroie_perclass_table.tex').write_text(
    "% GENERATED by scripts/emit_result_macros.py -- do not edit by hand.\n"
    r"""\begin{table}[t]
\centering
\caption{SROIE, LAC at the 90\% target, mean over 3 calibration splits.
\textbf{Marg.\ cov}: per-class coverage under the \emph{single marginal}
threshold $\hat{q}=\sroieQhatMarg$. \textbf{Mond.}: class-conditional
calibration, repairing per-class coverage at set size \sroieSizeMond{} by
spending quantiles spanning a factor of \sroieQhatRatio.}
\label{tab:mondrian_sroie}
\small
\setlength{\tabcolsep}{4pt}
\begin{tabular}{lrrrr}
\toprule
Class & $n$ & Marg.\ cov & Mond.\ cov & Mond.\ $\hat{q}$ \\
\midrule
""" + '\n'.join(_rows) + r"""
\midrule
Marginal (all) & --- & """ + f'{_marg_m:.1f}' + r"""\% & """ + f'{_mond_m:.1f}' + r"""\% & --- \\
\bottomrule
\end{tabular}
\end{table}
""")

# ------------------------------------------- compact backbone table (main)
(OUT / 'backbone_table.tex').write_text(
    "% GENERATED by scripts/emit_result_macros.py -- do not edit by hand.\n"
    r"""\begin{table}[t]
\centering
\caption{Backbone entity F1, reported only to establish that the extractor is
competent; F1 is not this paper's contribution. Published values are
\emph{reference points} under protocols that differ from ours in layout
granularity, annotation source and schema, so the table is not a leaderboard
and no row is a target. Ours uses word-level positions and the full 54-label
CORD schema. The full historical comparison is in the supplementary.}
\label{tab:ner_results}
\footnotesize
\setlength{\tabcolsep}{3pt}
\begin{tabular}{lccc}
\toprule
Model & FUNSD & CORD & WildRec. \\
\midrule
\multicolumn{4}{l}{\emph{Word-level, reference}~\cite{zhang2023readingorder}} \\
LayoutLMv3 seq.\ lab. & 78.77 & 82.72 & --- \\
Token Path Pred. & 80.40 & 91.85 & --- \\
\midrule
\multicolumn{4}{l}{\emph{Segment-level, reference}~\cite{huang2022layoutlmv3}} \\
LayoutLMv3-large & 92.08 & 97.46 & --- \\
\midrule
\methodname{} (ours) & \funsdF\,$\pm$\,\funsdFSd
  & \cordF\,$\pm$\,\cordFSd
  & \wildreceiptF\,$\pm$\,\wildreceiptFSd \\
\bottomrule
\end{tabular}
\end{table}
""")

print(f'wrote {len(M)} macros -> generated_results_macros.tex')
print(f'wrote spancrc_table.tex, backbone_table.tex')
