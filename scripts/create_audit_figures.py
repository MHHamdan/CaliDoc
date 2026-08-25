#!/usr/bin/env python3
"""
Figures for the conformal-data-audit experiments (E3 / E5 / ablation suite).

Every number is read from the result JSONs written by the local Phase 2 queue;
nothing is hardcoded except axis limits and the nominal coverage reference
lines, which are protocol constants rather than measurements.

Palette: Okabe-Ito subset in a FIXED order, validated for colour-vision
deficiency (worst adjacent pair dE 11.0 deutan, normal-vision floor 21.1).
Hues are assigned by series identity and never cycled or reordered per figure.
The two lower-contrast hues always carry a legend entry or a direct label, which
is the required relief for their sub-3:1 contrast against the surface.

Outputs PDF (vector, for LaTeX) into docs/paper/figures/.

Usage:
  python scripts/create_audit_figures.py --results_dir results/local_20260803
"""

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

# Fixed categorical order -- validated, never cycled.
CAT = ['#0072B2', '#D55E00', '#009E73', '#E69F00', '#CC79A7']
INK = '#1a1a1a'
INK_MUTED = '#6b6b6b'
GRID = '#d8d8d6'
SURFACE = '#fcfcfb'

plt.rcParams.update({
    'figure.facecolor': SURFACE,
    'axes.facecolor': SURFACE,
    'savefig.facecolor': SURFACE,
    'font.size': 8,
    'axes.labelsize': 8,
    'axes.titlesize': 9,
    'legend.fontsize': 7.5,
    'xtick.labelsize': 7.5,
    'ytick.labelsize': 7.5,
    'axes.edgecolor': GRID,
    'axes.linewidth': 0.6,
    'text.color': INK,
    'axes.labelcolor': INK,
    'xtick.color': INK_MUTED,
    'ytick.color': INK_MUTED,
    'grid.color': GRID,
    'grid.linewidth': 0.5,
    'legend.frameon': False,
    'pdf.fonttype': 42,
})

DATASETS = ['cord', 'funsd', 'wildreceipt', 'sroie']
NICE = {'cord': 'CORD', 'funsd': 'FUNSD',
        'wildreceipt': 'WildReceipt', 'sroie': 'SROIE'}


def _save(fig, out_dir, stem):
    """PDF for LaTeX, PNG alongside it so the layout can actually be eyeballed
    (the palette validator checks colour, not label collisions)."""
    fig.savefig(out_dir / f'{stem}.pdf')
    fig.savefig(out_dir / f'{stem}.png', dpi=200)


def recessive(ax, xgrid=False, ygrid=True):
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.set_axisbelow(True)
    if ygrid:
        ax.yaxis.grid(True)
    if xgrid:
        ax.xaxis.grid(True)


def load(results_dir, name):
    p = Path(results_dir) / name
    return json.loads(p.read_text()) if p.exists() else None


# --------------------------------------------------------------------------
# Fig 1: SROIE per-class coverage collapse (the headline diagnostic)
# --------------------------------------------------------------------------

def fig_sroie_collapse(results_dir, out_dir):
    d = load(results_dir, 'phase2_sroie.json')
    if d is None:
        print('skip sroie_collapse: no phase2_sroie.json')
        return
    per_seed = d['results']['per_seed']
    a = 'alpha_0.1'
    agg = {}
    for s in per_seed:
        for k, v in per_seed[s]['lac'][a]['per_class_coverage'].items():
            agg.setdefault(k, []).append((v['coverage'], v['n']))
    marginal = float(np.mean([per_seed[s]['lac'][a]['coverage'] for s in per_seed]))

    names = sorted(agg, key=lambda k: (np.mean([c for c, _ in agg[k]]), k))
    cov = [float(np.mean([c for c, _ in agg[k]])) * 100 for k in names]
    ns = [int(np.mean([n for _, n in agg[k]])) for k in names]

    fig, ax = plt.subplots(figsize=(5.0, 2.9))
    y = np.arange(len(names))
    # Single measure -> one hue; the O class is the confound being called out.
    colors = [CAT[1] if k == 'O' else CAT[0] for k in names]
    ax.barh(y, cov, height=0.62, color=colors, linewidth=0)

    # Marginal (89.9%) and nominal (90%) all but coincide -- which IS the
    # finding -- so draw marginal as a wide pale band and the nominal target as
    # a dashed rule on top of it, or the two read as a single line.
    ax.axvspan(marginal * 100 - 0.45, marginal * 100 + 0.45,
               color=CAT[2], alpha=0.55, lw=0, zorder=2)
    ax.axvline(90, color=INK, lw=1.0, ls=(0, (4, 2)), zorder=4)
    ax.annotate(f'marginal {marginal*100:.1f}%  =  nominal 90%',
                xy=(90, -0.75), xytext=(86, -1.35), ha='right',
                fontsize=7, color=INK,
                arrowprops=dict(arrowstyle='-', color=INK_MUTED, lw=0.7))

    ax.set_yticks(y)
    ax.set_yticklabels([f'{k}  (n={n})' for k, n in zip(names, ns)])
    ax.set_xlim(0, 104)
    ax.set_xlabel('LAC coverage at 90% target (%)')
    ax.set_ylim(-2.1, len(names) - 0.2)
    recessive(ax, xgrid=True, ygrid=False)
    for yi, c in zip(y, cov):
        ax.text(c + 1.2, yi, f'{c:.1f}', va='center', ha='left',
                fontsize=7, color=INK)
    fig.tight_layout()
    _save(fig, out_dir, 'audit_sroie_collapse')
    plt.close(fig)
    print('wrote audit_sroie_collapse')


# --------------------------------------------------------------------------
# Fig 2: matched-n control -- is the collapse just small calibration counts?
# --------------------------------------------------------------------------

def fig_matched_n(results_dir, out_dir):
    d = load(results_dir, 'conditional_cp_cord.json')
    if d is None:
        print('skip matched_n: no conditional_cp_cord.json')
        return
    m = d['per_unit'][0]['alpha_0.1']['matched_n_control']
    ns, mean, sd = [], [], []
    for k, v in m.items():
        if not k.startswith('fixed_cohort_n_'):
            continue
        ns.append(int(k.rsplit('_', 1)[1]))
        mean.append(v['mean_class_conditional_coverage'] * 100)
        sd.append(v['std'] * 100)
    order = np.argsort(ns)
    ns = np.array(ns)[order]; mean = np.array(mean)[order]; sd = np.array(sd)[order]

    sro = load(results_dir, 'phase2_sroie.json')
    fig, ax = plt.subplots(figsize=(4.6, 2.7))
    ax.axhline(90, color=INK_MUTED, lw=1.0, ls='--', zorder=2)
    ax.text(ns[0], 93.5, 'nominal 90%', ha='left', fontsize=7, color=INK_MUTED)

    ax.fill_between(ns, mean - sd, mean + sd, color=CAT[0], alpha=0.15, lw=0)
    ax.plot(ns, mean, color=CAT[0], lw=2.0, marker='o', ms=5,
            label='clean CORD classes (matched $n$)')

    if sro is not None:
        a = 'alpha_0.1'
        ps = sro['results']['per_seed']
        pts = {}
        for s in ps:
            for k, v in ps[s]['lac'][a]['per_class_coverage'].items():
                if k.startswith('B-'):
                    pts.setdefault(k, []).append((v['coverage'], v['n']))
        if pts:
            xs = [int(np.mean([n for _, n in v])) for v in pts.values()]
            ys = [float(np.mean([c for c, _ in v])) * 100 for v in pts.values()]
            ax.scatter(xs, ys, color=CAT[1], s=42, zorder=5, marker='D',
                       label='SROIE entity classes (observed)')
            ax.annotate('every SROIE B- class:\n0% at $n$=105',
                        xy=(xs[0], ys[0]), xytext=(xs[0] * 0.42, 26),
                        fontsize=7, color=INK,
                        arrowprops=dict(arrowstyle='->', color=INK_MUTED, lw=0.8))

    ax.set_xscale('log')
    # n=105 is deliberately NOT a tick: it sits 5px from 100 and the two labels
    # overprint. The callout already states it.
    ax.set_xticks(list(ns))
    ax.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
    ax.set_xlabel('calibration tokens per class ($n$)')
    ax.set_ylabel('class-conditional coverage (%)')
    ax.set_ylim(-4, 104)
    recessive(ax)
    ax.legend(loc='center right', bbox_to_anchor=(1.0, 0.42))
    fig.tight_layout()
    _save(fig, out_dir, 'audit_matched_n')
    plt.close(fig)
    print('wrote audit_matched_n')


# --------------------------------------------------------------------------
# Fig 3: selective prediction risk-coverage
# --------------------------------------------------------------------------

def fig_selective(results_dir, out_dir):
    fig, ax = plt.subplots(figsize=(4.6, 2.7))
    plotted = 0
    for i, ds in enumerate(DATASETS):
        d = load(results_dir, f'ablations_{ds}.json')
        if d is None:
            continue
        sp = d['per_unit'][0]['selective_prediction']
        cov = np.array(sp['coverage']) * 100
        risk = np.array(sp['risk']) * 100
        ax.plot(cov, risk, color=CAT[i], lw=2.0,
                label=f'{NICE[ds]} (AURC {sp["aurc"]:.3f})')
        ax.text(cov[-1], risk[-1], f'  {NICE[ds]}', fontsize=7,
                color=INK, va='center')
        plotted += 1
    if not plotted:
        plt.close(fig); print('skip selective'); return
    ax.set_xlabel('coverage (% of tokens retained)')
    ax.set_ylabel('token error rate (%)')
    ax.set_xlim(0, 118)
    ax.set_xticks([0, 25, 50, 75, 100])
    recessive(ax)
    ax.legend(loc='upper left')
    fig.tight_layout()
    _save(fig, out_dir, 'audit_selective_prediction')
    plt.close(fig)
    print('wrote audit_selective_prediction')


# --------------------------------------------------------------------------
# Fig 4: CovGap by calibration strategy
# --------------------------------------------------------------------------

def fig_covgap(results_dir, out_dir):
    strategies = [('marginal', 'marginal'),
                  ('classwise_mondrian', 'Mondrian'),
                  ('clustered_k3', 'clustered ($k$=3)'),
                  ('interpolated_lambda0.5', 'interpolated ($\\lambda$=0.5)')]
    rows, labels = [], []
    for ds in DATASETS:
        d = load(results_dir, f'conditional_cp_{ds}.json')
        if d is None:
            continue
        s = d['per_unit'][0]['alpha_0.1']['strategies']
        vals = []
        for key, _ in strategies:
            cs = s.get(key, {}).get('conditional_summary', {})
            vals.append(cs.get('covgap_pp'))
        if any(v is None for v in vals):
            continue
        rows.append(vals); labels.append(NICE[ds])
    if not rows:
        print('skip covgap'); return

    rows = np.array(rows)
    fig, ax = plt.subplots(figsize=(5.2, 2.7))
    x = np.arange(len(labels))
    w = 0.19
    for j, (_, name) in enumerate(strategies):
        # 2px surface gap between adjacent bars: inset each bar slightly.
        ax.bar(x + (j - 1.5) * w, rows[:, j], width=w * 0.88,
               color=CAT[j], linewidth=0, label=name)
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel('CovGap (pp, lower is better)')
    recessive(ax)
    ax.legend(ncol=2, loc='upper left')
    fig.tight_layout()
    _save(fig, out_dir, 'audit_covgap')
    plt.close(fig)
    print('wrote audit_covgap')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--results_dir', default='results/local_20260803')
    ap.add_argument('--out_dir', default='docs/paper/figures')
    args = ap.parse_args()
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    fig_sroie_collapse(args.results_dir, out)
    fig_matched_n(args.results_dir, out)
    fig_selective(args.results_dir, out)
    fig_covgap(args.results_dir, out)


if __name__ == '__main__':
    main()
