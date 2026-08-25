#!/usr/bin/env python3
"""Emit the WACV 2027 supplementary tables straight from the result JSONs.

Companion to emit_supplementary_tables.py (which emits the ablation block);
this script emits everything the WACV supplementary needs that did not exist
at the time of that script: SpanCRC per-field detail, m0 sensitivity,
boundary-escape eligibility with BH-adjusted q-values, the size/power study,
the first-token comparison at alpha=0.02, cross-conformal, conditional-CP
baselines, ECE/MCE, robustness grids, FUNSD per-class F1, the SROIE alignment
diagnosis, and the repaired-SROIE coverage shift.

Writes docs/paper-wacv2027/supp_wacv_tables.tex, one \\subsection-free stream
of table environments with stable \\label{}s that supplementary.tex places.

Usage: python3 scripts/emit_wacv_supplementary.py
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np

RD = Path('results/local_20260803')
OUT = Path('docs/paper-wacv2027/supp_wacv_tables.tex')
NICE = {'sroie': 'SROIE', 'sroie_repaired': 'SROIE (repaired)', 'cord': 'CORD',
        'funsd': 'FUNSD', 'wildreceipt': 'WildReceipt'}
DS_ORDER = ['sroie', 'sroie_repaired', 'wildreceipt', 'funsd', 'cord']


def load(name):
    return json.loads((RD / name).read_text())


def esc(s):
    return s.replace('_', '\\_')


def bh_q(pvals):
    """BH-adjusted q-values for a dict name->p."""
    items = sorted(pvals.items(), key=lambda kv: kv[1])
    m = len(items)
    q = {}
    prev = 1.0
    for i in range(m - 1, -1, -1):
        name, p = items[i]
        prev = min(prev, p * m / (i + 1))
        q[name] = prev
    return q


# ------------------------------------------------------------- SpanCRC

def tbl_spancrc_perfield():
    """Two floats: the large 57-field listing does not fit one page."""
    GROUPS = [('a', ['sroie', 'sroie_repaired', 'wildreceipt', 'funsd'],
               'SROIE, repaired SROIE, WildReceipt and FUNSD'),
              ('b', ['cord'], 'CORD (54-label schema)')]
    out = []
    for tag, dss, what in GROUPS:
        rows = []
        for ds in dss:
            j = load(f'span_crc_{ds}.json')
            u = j['per_unit'][0]
            for f, d in sorted(u['per_field'].items()):
                fb = '\\checkmark' if d['fallback'] else ''
                rows.append(
                    f"{NICE[ds]} & \\texttt{{{esc(f)}}} & {d['n_cal_spans']} & "
                    f"{d['n_ev_spans']} & {d['lambda']:.4f} & {fb} & "
                    f"{d['span_cov_token_threshold']:.3f} & "
                    f"{d['span_cov_field_crc']:.3f} \\\\")
            rows.append('\\midrule')
        out.append(r"""\begin{table*}[t]
\centering
\caption{SpanCRC per-field detail on the first calibration split ---
""" + what + r""". Calibration/evaluation span counts, the field threshold
$\hat\lambda_t$, whether the field falls back to the pooled threshold
($<m_0{=}25$ calibration spans), and full-span coverage under the
token-marginal threshold versus field-conditional CRC.}
\label{tab:supp_spancrc_perfield_""" + tag + r"""}
\small
\setlength{\tabcolsep}{4pt}
\begin{tabular}{llrrrcrr}
\toprule
Dataset & Field & $n_{\text{cal}}$ & $n_{\text{ev}}$ & $\hat\lambda_t$ &
fallback & Cov (tok.) & Cov (CRC) \\
\midrule
""" + '\n'.join(rows[:-1]) + r"""
\bottomrule
\end{tabular}
\end{table*}
""")
    return '\n'.join(out)


def tbl_m0():
    j = load('spancrc_sensitivity.json')
    rows = []
    for ds in ['cord', 'wildreceipt', 'funsd', 'sroie', 'sroie_repaired']:
        a = j['datasets'][ds]
        for m0 in ['10', '15', '25', '50']:
            d = a[m0]
            rows.append(
                f"{NICE[ds] if m0=='10' else ''} & {m0} & "
                f"{d['span_coverage']['mean']:.4f} & "
                f"{d['ev_doc_risk']['mean']:.4f} & "
                f"{d['span_covgap_pp']['mean']:.2f} & "
                f"{d['avg_set_size_candidate_rule']['mean']:.3f} & "
                f"{d['n_fields_fallback']['mean']:.0f}/{d['n_fields_cal']['mean']:.0f} \\\\")
        rows.append('\\midrule')
    return r"""\begin{table*}[t]
\centering
\caption{$m_0$ sensitivity (descriptive; $m_0{=}25$ is the pre-specified
operating point everywhere in the paper). Mean over calibration splits at the
90\% target; set size under the deployable candidate-label rule; realized risk
is the per-document field-miss fraction whose expectation CRC bounds.}
\label{tab:supp_m0}
\footnotesize
\setlength{\tabcolsep}{2.5pt}
\begin{tabular}{lrrrrrr}
\toprule
Dataset & $m_0$ & Span cov. & Risk & CovGap & Size & Fallback \\
\midrule
""" + '\n'.join(rows[:-1]) + r"""
\bottomrule
\end{tabular}
\end{table*}
"""


# ----------------------------------------------------- boundary escape

def tbl_be_eligibility():
    """Two floats: 57 field rows do not fit one page."""
    mult = load('multiplicity_analysis.json')['per_dataset']
    GROUPS = [('a', ['sroie', 'sroie_repaired', 'funsd'],
               'SROIE, repaired SROIE and FUNSD'),
              ('b', ['cord', 'wildreceipt'], 'CORD and WildReceipt')]
    out = []
    for tag, dss, what in GROUPS:
        rows = []
        for ds in dss:
            j = load(f'span_crc_{ds}.json')
            be = j['per_unit'][0]['boundary_escape']
            pv = {f: d['p_value'] for f, d in be['fields'].items()
                  if d['p_value'] is not None}
            qv = bh_q(pv) if pv else {}
            adj = mult[ds]['fields']
            for f, d in sorted(be['fields'].items(),
                               key=lambda kv: (kv[1]['p_value'] is None,
                                               kv[1]['p_value'] or 1.0)):
                p = d['p_value']
                enr = d['edge_enrichment']
                rej = d.get('holm_rejected', False)
                enr_s = '---' if enr is None else f'{enr:.2f}'
                p_s = '---' if p is None else f'{p:.4f}'
                holm_s = '---' if f not in adj else f"{adj[f]['holm']:.4f}"
                q_s = '---' if f not in qv else f'{qv[f]:.4f}'
                by_s = '---' if f not in adj else f"{adj[f]['by']:.4f}"
                cb = d.get('cluster_bootstrap')
                ci_s = (f"[{cb['ci90_lo']:.2f},{cb['ci90_hi']:.2f}]"
                        if cb else '---')
                nd_s = str(d.get('n_documents_informative', 0))
                rej_s = ('\\textbf{yes}' if rej else
                         ('no' if p is not None else 'ineligible'))
                rows.append(
                    f"{NICE[ds]} & \\texttt{{{esc(f)}}} & {d['n_spans']} & "
                    f"{d['n_informative_spans']} & {nd_s} & {enr_s} & {p_s} & "
                    f"{holm_s} & {q_s} & {by_s} & {ci_s} & {rej_s} \\\\")
            rows.append('\\midrule')
        out.append(r"""\begin{table*}[t]
\centering
\caption{Boundary-escape eligibility and results for \emph{every} field type
--- """ + what + r""". First calibration split, $\tau$ = pooled CRC threshold,
$B{=}5000$ Monte-Carlo draws, seed $0$, level $0.10$. A field is eligible only
if it contributes at least one informative span ($0 < k_s < L_s$); ineligible
fields are untested, never certified clean. \textbf{Holm} (FWER across the
field-level tests) is the primary decision; \textbf{BH} (FDR under positive
dependence) and \textbf{BY} (FDR, arbitrary dependence) are shown for
robustness, and all three agree everywhere. \textbf{CI} is the
document-blocked cluster-bootstrap $90\%$ interval for enrichment over
\textbf{D} documents, which allows arbitrary within-document dependence. The
Monte-Carlo resolution floor is $p = 1/(B{+}1) \approx 2\times10^{-4}$.}
\label{tab:supp_be_eligibility_""" + tag + r"""}
\scriptsize
\setlength{\tabcolsep}{2.5pt}
\begin{tabular}{llrrrrrrrrll}
\toprule
Dataset & Field & Spans & Inf. & D & Enrich. & $p$ & Holm & BH & BY & CI$_{90}$ & Rej. \\
\midrule
""" + '\n'.join(rows[:-1]) + r"""
\bottomrule
\end{tabular}
\end{table*}
""")
    return '\n'.join(out)


def tbl_pooled_multiplicity():
    m = load('multiplicity_analysis.json')
    rows = []
    for k in sorted(m['pooled']['fields'], key=lambda k: m['pooled']['fields'][k]['p'])[:8]:
        a = m['pooled']['fields'][k]
        ds, f = k.split(':', 1)
        rj = '\\textbf{yes}' if a['reject_holm'] else 'no'
        rows.append(f"{NICE[ds]} & \\texttt{{{esc(f)}}} & {a['p']:.4f} & "
                    f"{a['holm']:.4f} & {a['bh']:.4f} & {a['by']:.4f} & "
                    f"{rj} \\\\")
    stab = m['stability_across_splits']
    stab_rows = []
    for ds, per in stab.items():
        parts = []
        for pl in per:
            if pl:
                inner = ', '.join('\\texttt{' + esc(x) + '}' for x in pl)
                parts.append('\\{' + inner + '\\}')
            else:
                parts.append('$\\emptyset$')
        cells = '; '.join(parts)
        stab_rows.append(f"{NICE[ds]} & {cells} \\\\")
    return r"""\begin{table*}[t]
\centering
\caption{Pooled-family multiplicity control: one family over all
""" + str(m['pooled']['n_eligible']) + r""" eligible fields of the four
benchmark datasets---the most conservative reading of how many tests were run.
The eight smallest $p$-values are shown. Holm, BH and BY all reject exactly the
two SROIE fields, as they do under the per-dataset families reported in the
main paper.}
\label{tab:supp_pooled_mult}
\footnotesize
\setlength{\tabcolsep}{4pt}
\begin{tabular}{llrrrrl}
\toprule
Dataset & Field & $p$ & Holm & BH & BY & Rejected \\
\midrule
""" + '\n'.join(rows) + r"""
\bottomrule
\end{tabular}
\end{table*}

\begin{table}[t]
\centering
\caption{Stability of the Holm decision across the calibration splits.}
\label{tab:supp_stability}
\footnotesize
\setlength{\tabcolsep}{3pt}
\begin{tabular}{lp{0.60\linewidth}}
\toprule
Dataset & Rejected set per split \\
\midrule
""" + '\n'.join(stab_rows) + r"""
\bottomrule
\end{tabular}
\end{table}
"""


def tbl_power():
    j = load('boundary_escape_power.json')
    out = []
    # size arm
    srows = []
    for ds, d in j['size'].items():
        inf = d['n_informative_by_field']
        bp = d['familywise_rejection_rate_by_procedure']
        srows.append(f"{NICE[ds]} & {d['n_replicates']} & "
                     f"{sum(inf.values())} & {len(inf)} & "
                     f"{d['familywise_rejection_rate_uncorrected']:.3f} & "
                     f"{bp['holm']:.3f} & {bp['bh']:.3f} & {bp['by']:.3f} \\\\")
    out.append(r"""\begin{table}[t]
\centering
\caption{Size of the boundary-escape test under the synthetic
exchangeable-position null: escape positions are redrawn uniformly within each
informative span, holding the $(L_s, k_s)$ structure fixed, and the full
pipeline is re-run per replicate. ``None'' is the familywise rejection rate
with no multiplicity correction at all, which is what the correction is
buying.}
\label{tab:supp_be_size}
\footnotesize
\setlength{\tabcolsep}{3pt}
\begin{tabular}{lrrrrrrr}
\toprule
& & & & \multicolumn{4}{c}{Familywise rejection rate} \\
\cmidrule(lr){5-8}
Dataset & Reps & Inf. & Fields & None & Holm & BH & BY \\
\midrule
""" + '\n'.join(srows) + r"""
\bottomrule
\end{tabular}
\end{table}
""")
    # power arms
    for variant, vname in (('end_extend', 'end-extension (one corrupted edge token)'),
                           ('start_shift', 'start-shift (edge + interior corrupted)')):
        rows = []
        for ds, fields in j['power'][variant].items():
            for f, cells in fields.items():
                for eps in j['doses']:
                    sub = [c for c in cells if c['eps'] == eps]
                    det = np.mean([c['detected'] for c in sub])
                    fp = np.mean([len(c['false_positive_fields']) for c in sub])
                    inf = np.mean([c['n_informative'] for c in sub])
                    nc = sub[0]['n_corrupted']
                    rows.append(f"{NICE[ds]} & \\texttt{{{esc(f)}}} & {eps:.2f} & "
                                f"{nc} & {inf:.0f} & {det:.2f} & {fp:.2f} \\\\")
                rows.append('\\midrule')
        out.append(r"""\begin{table*}[t]
\centering
\caption{Boundary-escape power under """ + vname + r""" corruption of the
calibration labels; $10$ corruption seeds per cell, thresholds and test fully
recomputed from the corrupted labels. ``FP'' counts BH rejections of untouched
fields per run; on FUNSD these are exclusively the borderline clean
\texttt{QUESTION} signal ($p{=}0.036$ uncorrupted) being tipped over the BH
threshold, not new false discoveries (see text).}
\label{tab:supp_be_power_""" + variant + r"""}
\footnotesize
\setlength{\tabcolsep}{2.5pt}
\begin{tabular}{llrrrrr}
\toprule
Dataset & Field & $\epsilon$ & Corrupted & Inf. & Power & FP/run \\
\midrule
""" + '\n'.join(rows[:-1]) + r"""
\bottomrule
\end{tabular}
\end{table*}
""")
    return '\n'.join(out)


# ------------------------------------------------- first-token comparison

def tbl_first_token():
    j = load('first_token_entity_cp.json')
    rows = []
    for r in j['runs']:
        a = r['aggregate']
        rows.append(
            f"{NICE[r['dataset']]} & {r['alpha']:.2f} & "
            f"{a['first_token_coverage']['mean']:.3f}$\\pm${a['first_token_coverage']['std']:.3f} & "
            f"{a['first_token_avg_set_size']['mean']:.2f} & "
            f"{a['first_token_singleton_rate']['mean']:.3f} & "
            f"{a['full_span_coverage_same_threshold']['mean']:.3f} & "
            f"{a['spancrc_span_coverage']['mean']:.3f} \\\\")
    return r"""\begin{table*}[t]
\centering
\caption{First-token entity-level CP (the construction of the closest prior
work) replicated on our probabilities, mean over three calibration splits.
``Span cov.''\ is the fraction of complete gold spans covered at the same
threshold---the quantity the first-token unit does not certify; the last
column is pooled SpanCRC at the same $\alpha$ for contrast. Protocol
differences from the published numbers (backbone, score, schema, calibration
split) are listed in the main paper.}
\label{tab:supp_first_token}
\footnotesize
\setlength{\tabcolsep}{2pt}
\begin{tabular}{lrrrrrr}
\toprule
Dataset & $\alpha$ & FT cov. & FT size & Singleton & Span cov. & SpanCRC \\
\midrule
""" + '\n'.join(rows) + r"""
\bottomrule
\end{tabular}
\end{table*}
"""


# --------------------------------------------------- conditional CP / CORD

def tbl_conditional():
    j = load('conditional_cp_cord.json')
    u = j['per_unit'][0]['alpha_0.1']
    s = u['strategies']
    def row(name, key):
        d = s[key]
        return (f"{name} & {d['conditional_summary']['covgap_pp']:.1f} & "
                f"{d['avg_set_size']:.2f} & "
                f"{d['marginal_coverage']*100:.1f} \\\\")
    rows = [row('Marginal', 'marginal')]
    for k in ('clustered_k2', 'clustered_k3', 'clustered_k4', 'clustered_k5',
              'clustered_k8'):
        rows.append(row(f"Clustered CP ($k{{=}}{k.split('_k')[1]}$)", k))
    for lam in ('0.25', '0.5', '0.75'):
        rows.append(row(f"Interpolated ($\\lambda{{=}}{lam}$)",
                        f'interpolated_lambda{lam}'))
    rows.append(row('Class-conditional (Mondrian)', 'classwise_mondrian'))
    return r"""\begin{table}[t]
\centering
\caption{Conditional-CP baselines on corrected CORD at the 90\% target (first
calibration split): CovGap over the 44 evaluated classes, average set size,
and marginal coverage. $24$ classes fall below the feasibility bound
$n \ge \lceil(1-\alpha)/\alpha\rceil - 1 = 8$.}
\label{tab:supp_conditional}
\footnotesize
\setlength{\tabcolsep}{3pt}
\begin{tabular}{lrrr}
\toprule
Strategy & CovGap (pp) & Size & Marg.\ cov.\ (\%) \\
\midrule
""" + '\n'.join(rows) + r"""
\bottomrule
\end{tabular}
\end{table}
"""


def tbl_matched_n():
    j = load('conditional_cp_cord.json')
    mc = j['per_unit'][0]['alpha_0.1']['matched_n_control']
    rows = []
    for n in ['5', '10', '20', '50', '100']:
        d = mc[f'fixed_cohort_n_{n}']
        rows.append(f"{n} & {d['mean_class_conditional_coverage']:.3f} \\\\")
    return r"""\begin{table}[t]
\centering
\caption{Matched-$n$ control on clean CORD: a fixed cohort of well-populated
classes subsampled to $n$ calibration tokens per class, so only $n$ varies.
Nominal coverage $0.90$.}
\label{tab:supp_matched_n}
\small
\begin{tabular}{rr}
\toprule
$n$ per class & Mean class-conditional coverage \\
\midrule
""" + '\n'.join(rows) + r"""
\bottomrule
\end{tabular}
\end{table}
"""


def tbl_sroie_repaired_shift():
    a = load('conditional_cp_sroie.json')['per_unit']
    b = load('conditional_cp_sroie_repaired.json')['per_unit']
    def mean_cov(units, cls):
        vals = [u['alpha_0.1']['strategies']['marginal']['per_class'][cls]['coverage']
                for u in units if cls in u['alpha_0.1']['strategies']['marginal']['per_class']]
        return float(np.mean(vals)) if vals else float('nan')
    classes = sorted(a[0]['alpha_0.1']['strategies']['marginal']['per_class'])
    rows = []
    for c in classes:
        rows.append(f"\\texttt{{{esc(c)}}} & {mean_cov(a, c):.3f} & "
                    f"{mean_cov(b, c):.3f} \\\\")
    return r"""\begin{table}[t]
\centering
\caption{Per-class marginal-threshold coverage on SROIE before and after the
relaxed-matcher repair (mean over three calibration splits, 90\% target). The
repair \emph{redistributes} token-level errors rather than uniformly improving
them; the span-level and positional statistics in the main paper are what
register it as a fix.}
\label{tab:supp_sroie_shift}
\small
\begin{tabular}{lrr}
\toprule
Class & Standard & Repaired \\
\midrule
""" + '\n'.join(rows) + r"""
\bottomrule
\end{tabular}
\end{table}
"""


# --------------------------------------------------------- calibration etc.

def tbl_ece():
    m = load('multiseed_summary.json')['datasets']
    rows = []
    for ds in ['cord', 'wildreceipt', 'funsd', 'sroie']:
        ps = m[ds]['per_seed']
        ece = [ps[s]['ece'] for s in ps]
        mce = [ps[s]['mce'] for s in ps]
        f1 = [ps[s]['f1'] for s in ps]
        rows.append(f"{NICE[ds]} & {np.mean(f1):.2f}$\\pm${np.std(f1):.2f} & "
                    f"{np.mean(ece):.2f}$\\pm${np.std(ece):.2f} & "
                    f"{np.mean(mce):.2f}$\\pm${np.std(mce):.2f} \\\\")
    return r"""\begin{table}[t]
\centering
\caption{Per-dataset F1, ECE and MCE (temperature-scaled, \%), mean $\pm$ std
over training seeds \{42, 123, 456\}. SROIE is the case-study lineage.}
\label{tab:supp_ece}
\small
\begin{tabular}{lrrr}
\toprule
Dataset & F1 & ECE & MCE \\
\midrule
""" + '\n'.join(rows) + r"""
\bottomrule
\end{tabular}
\end{table}
"""


def tbl_cross_conformal():
    j = load('phase2_cord.json')['results']['cross_conformal_k5']['lac']
    # seed-42 checkpoint on both sides: cross-conformal is a seed-42 run
    m = load('multiseed_summary.json')['datasets']['cord']['per_seed']['42']
    rows = []
    for a, fx in (('alpha_0.1', m['lac_cov_alpha_0.1']),
                  ('alpha_0.05', m['lac_cov_alpha_0.05']),
                  ('alpha_0.01', m['lac_cov_alpha_0.01'])):
        cc = j[a]['mean'] * 100
        tgt = 100 - float(a.split('_')[1]) * 100
        rows.append(f"{tgt:.0f}\\% & {fx:.2f} & {cc:.2f} & {abs(fx-cc):.2f} \\\\")
    return r"""\begin{table}[t]
\centering
\caption{$K{=}5$ cross-conformal validation on CORD (LAC, seed-42 checkpoint)
against the fixed independent calibration split. Agreement within $0.2$
points at every target rules out a data-reuse artifact.}
\label{tab:supp_crossconf}
\small
\begin{tabular}{lrrr}
\toprule
Target & Fixed split & Cross-conformal & $|\Delta|$ \\
\midrule
""" + '\n'.join(rows) + r"""
\bottomrule
\end{tabular}
\end{table}
"""


def tbl_funsd_perclass():
    j = load('ablations_funsd.json')
    pc = j['per_class_f1']
    rows = []
    for cls, d in sorted(pc.items()):
        if isinstance(d, dict) and 'f1-score' in d:
            rows.append(f"\\texttt{{{esc(cls)}}} & {d['f1-score']:.3f} & "
                        f"{d['precision']:.3f} & {d['recall']:.3f} & "
                        f"{int(d['support'])} \\\\")
    return r"""\begin{table}[t]
\centering
\caption{FUNSD per-class entity metrics (seed-42 checkpoint).}
\label{tab:supp_funsd_perclass}
\small
\begin{tabular}{lrrrr}
\toprule
Class & F1 & Prec. & Rec. & Support \\
\midrule
""" + '\n'.join(rows) + r"""
\bottomrule
\end{tabular}
\end{table}
"""


def tbl_robustness():
    parts = []
    for ds in ['cord', 'funsd', 'wildreceipt', 'sroie']:
        j = load(f'perturbation_coverage_{ds}.json')
        rows = []
        for pert, sevs in sorted(j['perturbations'].items()):
            cells_f = [sevs[f'severity_{s}']['f1_drop_pp'] for s in (1, 2, 3)]
            cells_c = [sevs[f'severity_{s}']['coverage_drop_pp'] for s in (1, 2, 3)]
            rows.append(f"\\texttt{{{esc(pert)}}} & " +
                        ' & '.join(f'{v:.2f}' for v in cells_f) + ' & ' +
                        ' & '.join(f'{v:.2f}' for v in cells_c) + ' \\\\')
        parts.append(r"""\begin{table}[t]
\centering
\caption{Robustness grid, """ + NICE[ds] + r""" (seed-42): F1 drop and coverage
drop (pp, positive = worse) under each perturbation at severities 1--3.
Temperature and quantile are fitted on clean data and held fixed; only images
are perturbed (gold OCR preserved).}
\label{tab:supp_robust_""" + ds + r"""}
\footnotesize
\setlength{\tabcolsep}{2.5pt}
\begin{tabular}{lrrrrrr}
\toprule
& \multicolumn{3}{c}{F1 drop (pp)} & \multicolumn{3}{c}{Cov.\ drop (pp)} \\
\cmidrule(lr){2-4}\cmidrule(lr){5-7}
Perturbation & s1 & s2 & s3 & s1 & s2 & s3 \\
\midrule
""" + '\n'.join(rows) + r"""
\bottomrule
\end{tabular}
\end{table}
""")
    return '\n'.join(parts)


def tbl_sroie_alignment():
    txt = (Path('results') / 'sroie_alignment_diagnosis_20260810.txt').read_text()
    # parse the combined block and the test per-field block
    rows = []
    m = re.search(r'===== test.*?by field:\n(.*?)\n  examples', txt, re.S)
    for line in m.group(1).strip().splitlines():
        f = re.match(r'\s*(\w+)\s+n=\s*(\d+)\s+strict\s+([\d.]+)%\s+recoverable\s+([\d.]+)%\s+hard-fail\s+(\d+)', line)
        if f:
            rows.append(f"\\texttt{{{f.group(1)}}} & {f.group(2)} & "
                        f"{f.group(3)} & {f.group(4)} & {f.group(5)} \\\\")
    m2 = re.search(r'===== combined: (\d+) field instances =====\n(.*)', txt, re.S)
    comb = []
    for line in m2.group(2).strip().splitlines():
        f = re.match(r'\s*(\w+)\s+(\d+)\s+([\d.]+)%\s+cumulative\s+([\d.]+)%', line)
        if f:
            comb.append(f"\\texttt{{{f.group(1)}}} & {f.group(2)} & "
                        f"{f.group(3)} & {f.group(4)} \\\\")
    return r"""\begin{table}[t]
\centering
\caption{SROIE Task-3 alignment diagnosis (persisted artifact
\texttt{sroie\_alignment\_diagnosis}). Left: failure-cause cascade over all
$3{,}890$ field instances (both splits); each instance is credited to the
first matcher that recovers it. The pre-registered gate admits only the
\texttt{strict} level.}
\label{tab:supp_sroie_align}
\footnotesize
\begin{tabular}{lrrr}
\toprule
Cause & $n$ & \% & Cumulative \% \\
\midrule
""" + '\n'.join(comb) + r"""
\bottomrule
\end{tabular}

\vspace{0.6em}

\footnotesize
\begin{tabular}{lrrrr}
\toprule
Field (test) & $n$ & Strict \% & Recoverable \% & Hard fail \\
\midrule
""" + '\n'.join(rows) + r"""
\bottomrule
\end{tabular}
\end{table}
"""


def main():
    parts = [
        '% GENERATED by scripts/emit_wacv_supplementary.py -- do not edit by hand.',
        tbl_spancrc_perfield(),
        tbl_m0(),
        tbl_be_eligibility(),
        tbl_pooled_multiplicity(),
        tbl_power(),
        tbl_first_token(),
        tbl_conditional(),
        tbl_matched_n(),
        tbl_sroie_repaired_shift(),
        tbl_ece(),
        tbl_cross_conformal(),
        tbl_funsd_perclass(),
        tbl_sroie_alignment(),
        tbl_robustness(),
    ]
    OUT.write_text('\n'.join(parts) + '\n')
    print(f'wrote {OUT}')


if __name__ == '__main__':
    main()
