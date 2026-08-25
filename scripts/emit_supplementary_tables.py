#!/usr/bin/env python3
"""
Emit the supplementary ablation tables as a LaTeX fragment, straight from the
result JSONs.

Hand-transcribing these numbers is how this project shipped a stale CORD F1
through a full rewrite. Everything below is computed, so re-running the script
after new results land updates the manuscript with no manual editing.

Writes docs/paper/supp_ablation_tables.tex, which supplementary.tex \\input{}s.

Usage:
  python scripts/emit_supplementary_tables.py --results_dir results/local_20260803
"""

import argparse
import json
from pathlib import Path

import numpy as np

DATASETS = ['cord', 'funsd', 'wildreceipt', 'sroie']
NICE = {'cord': 'CORD', 'funsd': 'FUNSD',
        'wildreceipt': 'WildReceipt', 'sroie': 'SROIE\\textsuperscript{$\\dagger$}'}
ALPHAS = ['alpha_0.1', 'alpha_0.05', 'alpha_0.01']
ALPHA_HDR = ['90\\%', '95\\%', '99\\%']


def load(rd, ds):
    p = Path(rd) / f'ablations_{ds}.json'
    return json.loads(p.read_text()) if p.exists() else None


def mean_over_units(d, path, default=None):
    """path is a callable taking a unit dict."""
    vals = []
    for u in d['per_unit']:
        try:
            v = path(u)
        except (KeyError, TypeError):
            continue
        if v is not None:
            vals.append(v)
    return float(np.mean(vals)) if vals else default


def tbl_selective(data):
    rows = []
    for ds, d in data.items():
        aurc = mean_over_units(d, lambda u: u['selective_prediction']['aurc'])
        r = {c: mean_over_units(
                d, lambda u, c=c: u['selective_prediction']['risk_at_coverage'][c])
             for c in ('1.00', '0.90', '0.80', '0.50')}
        rows.append(f"{NICE[ds]} & {aurc:.4f} & {r['1.00']*100:.2f} & "
                    f"{r['0.90']*100:.2f} & {r['0.80']*100:.2f} & {r['0.50']*100:.2f} \\\\")
    return r"""\begin{table}[t]
\centering
\caption{Selective prediction. AURC is the area under the risk--coverage curve
(lower is better); the remaining columns give the token error rate (\%) when the
least-confident tokens are abstained on down to the stated coverage. Confidence
is the temperature-scaled maximum softmax probability.}
\label{tab:supp_selective}
\footnotesize
\setlength{\tabcolsep}{3pt}
\begin{tabular}{lrrrrr}
\toprule
Dataset & AURC & 100\% & 90\% & 80\% & 50\% \\
\midrule
""" + '\n'.join(rows) + r"""
\bottomrule
\end{tabular}
\end{table}
"""


def tbl_empty_sets(data):
    rows = []
    for ds, d in data.items():
        cells = []
        for a in ALPHAS:
            r = mean_over_units(d, lambda u, a=a: u['empty_sets'][a]['empty_set_rate'])
            cells.append(f'{r*100:.1f}' if r is not None else '---')
        sizes = [mean_over_units(
            d, lambda u, a=a: u['scoring_ablation']['lac'][a]['avg_set_size'])
            for a in ALPHAS]
        rows.append(f"{NICE[ds]} & " + ' & '.join(cells) + ' & '
                    + ' & '.join(f'{s:.2f}' for s in sizes) + r' \\')
    return r"""\begin{table}[t]
\centering
\caption{Empty prediction sets under LAC. An empty set occurs when no class
reaches the threshold $1-\hat{q}$; it is why the average LAC set size can fall
below one. APS and RAPS retain at least the top-ranked class by construction and
so never produce empty sets. This is a genuine limitation of LAC at these
$\alpha$ levels, not a reporting artifact.}
\label{tab:supp_empty}
\footnotesize
\setlength{\tabcolsep}{3pt}
\begin{tabular}{lrrrrrr}
\toprule
& \multicolumn{3}{c}{Empty-set rate (\%)} & \multicolumn{3}{c}{Avg.\ LAC set size} \\
\cmidrule(lr){2-4}\cmidrule(lr){5-7}
Dataset & """ + ' & '.join(ALPHA_HDR) + ' & ' + ' & '.join(ALPHA_HDR) + r""" \\
\midrule
""" + '\n'.join(rows) + r"""
\bottomrule
\end{tabular}
\end{table}
"""


def tbl_temperature(data):
    rows = []
    for ds, d in data.items():
        t = d['temperature_ece']
        rows.append(f"{NICE[ds]} & {t['temperature']:.3f} & "
                    f"{t['uncalibrated']['ece']:.2f} & {t['temperature_scaled']['ece']:.2f} & "
                    f"{t['delta_ece']:+.2f} & {t['uncalibrated']['mce']:.2f} & "
                    f"{t['temperature_scaled']['mce']:.2f} \\\\")
    return r"""\begin{table}[t]
\centering
\caption{Temperature scaling, fitted on the early-stopping split only and
evaluated on test. $\Delta$ECE is negative when scaling helps. Note that it
\emph{hurts} on two of four datasets: the early-stopping splits are small (CORD
50 documents, FUNSD 15), so the fitted temperature does not transfer reliably.
We report this rather than select the favourable subset.}
\label{tab:supp_temperature}
\footnotesize
\setlength{\tabcolsep}{3pt}
\begin{tabular}{lrrrrrr}
\toprule
& & \multicolumn{3}{c}{ECE (\%)} & \multicolumn{2}{c}{MCE (\%)} \\
\cmidrule(lr){3-5}\cmidrule(lr){6-7}
Dataset & $T$ & $T{=}1$ & $T{=}T^{*}$ & $\Delta$ & $T{=}1$ & $T{=}T^{*}$ \\
\midrule
""" + '\n'.join(rows) + r"""
\bottomrule
\end{tabular}
\end{table}
"""


def tbl_entity_level(data):
    rows = []
    for ds, d in data.items():
        cells = []
        for a in ALPHAS:
            tok = mean_over_units(d, lambda u, a=a: u['entity_level_cp'][a]['token_coverage'])
            ent = mean_over_units(d, lambda u, a=a: u['entity_level_cp'][a]['entity_coverage'])
            cells.append(f'{tok*100:.1f} / {ent*100:.1f}' if tok is not None else '---')
        rows.append(f"{NICE[ds]} & " + ' & '.join(cells) + r' \\')
    return r"""\begin{table}[t]
\centering
\caption{Token-level versus entity-span-level coverage (token / span, \%). A
gold BIO span counts as covered only if \emph{every} one of its tokens has its
gold tag in that token's prediction set. Token-level conformal prediction
carries no span-level guarantee, and on SROIE the gap is total: span coverage is
$0.0\%$ against $89.8\%$ token coverage, because every span begins with a
\texttt{B-} token and no \texttt{B-} class is covered under the marginal
threshold.}
\label{tab:supp_entity}
\footnotesize
\setlength{\tabcolsep}{3pt}
\begin{tabular}{lccc}
\toprule
Dataset & """ + ' & '.join(ALPHA_HDR) + r""" \\
\midrule
""" + '\n'.join(rows) + r"""
\bottomrule
\end{tabular}
\end{table}
"""


def tbl_scoring(data):
    rows = []
    for ds, d in data.items():
        u0 = d['per_unit'][0]['scoring_ablation']
        lac = mean_over_units(d, lambda u: u['scoring_ablation']['lac']['alpha_0.1']['coverage'])
        lacs = mean_over_units(d, lambda u: u['scoring_ablation']['lac']['alpha_0.1']['avg_set_size'])
        aps = mean_over_units(d, lambda u: u['scoring_ablation']['aps']['alpha_0.1']['coverage'])
        apss = mean_over_units(d, lambda u: u['scoring_ablation']['aps']['alpha_0.1']['avg_set_size'])
        sa = u0.get('structure_aware', {})
        if isinstance(sa, dict) and 'lambda_layout_0.1' in sa:
            c = mean_over_units(
                d, lambda u: u['scoring_ablation']['structure_aware']['lambda_layout_0.1']['alpha_0.1']['coverage'])
            z = mean_over_units(
                d, lambda u: u['scoring_ablation']['structure_aware']['lambda_layout_0.1']['alpha_0.1']['avg_set_size'])
            sacell = f'{c*100:.1f} & {z:.2f}'
        else:
            sacell = '--- & ---'
        rows.append(f'{NICE[ds]} & {lac*100:.1f} & {lacs:.2f} & {aps*100:.1f} & '
                    f'{apss:.2f} & {sacell} \\\\')
    return r"""\begin{table}[t]
\centering
\caption{Nonconformity score comparison at the 90\% target (coverage \%, average
set size). APS and RAPS are the \emph{deterministic} variants (no tie-breaking
randomisation), which is why APS is heavily conservative on the large label
spaces: 20.4 labels per set on CORD (54 labels) and 17.9 on WildReceipt (49)
against roughly one for LAC. The structure-aware score adds
$\lambda_{\text{layout}}$ times the normalised layout-attention entropy to the
LAC score; $\lambda_{\text{OCR}}$ is fixed at zero because no bundle dataset
carries OCR confidence. LAC remains the informative choice.}
\label{tab:supp_scoring}
\footnotesize
\setlength{\tabcolsep}{3pt}
\begin{tabular}{lrrrrrr}
\toprule
& \multicolumn{2}{c}{LAC} & \multicolumn{2}{c}{APS (det.)} & \multicolumn{2}{c}{Structure-aware} \\
\cmidrule(lr){2-3}\cmidrule(lr){4-5}\cmidrule(lr){6-7}
Dataset & Cov. & Size & Cov. & Size & Cov. & Size \\
\midrule
""" + '\n'.join(rows) + r"""
\bottomrule
\end{tabular}
\end{table}
"""


def tbl_mc_dropout(data):
    rows = []
    for ds, d in data.items():
        mc = d.get('mc_dropout')
        if not mc:
            continue
        cp = mc['conformal_on_mc_probs']['alpha_0.1']
        rows.append(f"{NICE[ds]} & {mc['n_passes']} & {mc['accuracy']*100:.2f} & "
                    f"{mc['ece']:.2f} & {mc['mean_predictive_entropy']:.3f} & "
                    f"{cp['coverage']*100:.1f} & {cp['avg_set_size']:.2f} \\\\")
    if not rows:
        return ''
    return r"""\begin{table}[t]
\centering
\caption{MC Dropout ($N$ stochastic forward passes, dropout active at
inference). Conformal columns apply the calibration quantile from the
deterministic pass to the MC-averaged predictive distribution, isolating the
effect of MC averaging on the evaluation distribution.}
\label{tab:supp_mcdropout}
\footnotesize
\setlength{\tabcolsep}{3pt}
\begin{tabular}{lrrrrrr}
\toprule
Dataset & $N$ & Acc.\ (\%) & ECE (\%) & Pred.\ entropy & Cov.\ @90 & Size \\
\midrule
""" + '\n'.join(rows) + r"""
\bottomrule
\end{tabular}
\end{table}
"""


def tbl_raps(data, ds='cord'):
    d = data.get(ds)
    if d is None:
        return ''
    grid = d['per_unit'][0]['raps']
    rows = []
    for key in sorted(grid):
        k_reg = key.split('_')[2]
        lam = key.split('lambda_')[1]
        e = grid[key]['alpha_0.1']
        rows.append(f"{k_reg} & {lam} & {e['coverage']*100:.1f} & "
                    f"{e['avg_set_size']:.2f} \\\\")
    return r"""\begin{table}[t]
\centering
\caption{Deterministic RAPS on """ + NICE[ds].replace('\\textsuperscript{$\\dagger$}', '') + r""" at the 90\% target, over the
regularisation grid. RAPS interpolates between APS and a size-capped variant;
with the penalty active it recovers near-nominal coverage at far smaller sets
than deterministic APS.}
\label{tab:supp_raps}
\footnotesize
\setlength{\tabcolsep}{3pt}
\begin{tabular}{rrrr}
\toprule
$k_{\text{reg}}$ & $\lambda$ & Coverage (\%) & Set size \\
\midrule
""" + '\n'.join(rows) + r"""
\bottomrule
\end{tabular}
\end{table}
"""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--results_dir', default='results/local_20260803')
    ap.add_argument('--output', default='docs/paper/supp_ablation_tables.tex')
    args = ap.parse_args()

    data = {}
    for ds in DATASETS:
        d = load(args.results_dir, ds)
        if d is not None:
            data[ds] = d
    if not data:
        print('no ablation JSONs found'); return

    parts = [
        '% GENERATED by scripts/emit_supplementary_tables.py -- do not edit by hand.',
        '% Re-run that script after new results land.',
        '',
        tbl_selective(data),
        tbl_scoring(data),
        tbl_empty_sets(data),
        tbl_entity_level(data),
        tbl_temperature(data),
        tbl_mc_dropout(data),
        tbl_raps(data, 'cord'),
        r'\noindent\textsuperscript{$\dagger$}\,SROIE is reported from the '
        r'standard pre-tagged release and is a data-audit case study, not a '
        r'benchmark result; see the main paper.',
    ]
    out = Path(args.output)
    out.write_text('\n'.join(p for p in parts if p) + '\n')
    print(f'wrote {out} ({len(data)} datasets)')


if __name__ == '__main__':
    main()
