#!/usr/bin/env python3
"""Emit the review-response supplementary tables as standalone table PDFs.

Reads only committed result artifacts (span_length_analysis.json,
spancrc_interpolation.json, boundary_escape_masks.json) and compiles each
table with pdflatex (standalone class) into
docs/CaliDoc_WACV2027_clean_source/tables/. No number is hand-entered.

    python3 scripts/emit_review_tables.py
"""

from __future__ import annotations

import json
import subprocess
import tempfile
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
RD = REPO / 'results' / 'local_20260803'
TABLES = REPO / 'docs' / 'CaliDoc_WACV2027_clean_source' / 'tables'
NICE = {'sroie': 'SROIE', 'sroie_repaired': 'SROIE (rep.)', 'cord': 'CORD',
        'funsd': 'FUNSD', 'wildreceipt': 'WildReceipt'}
ORDER = ['sroie', 'sroie_repaired', 'wildreceipt', 'funsd', 'cord']
BUCKETS = ['1', '2', '3-4', '5-8', '>=9']
BUCKET_TEX = {'1': '$L{=}1$', '2': '$L{=}2$', '3-4': '$L{=}3$--$4$',
              '5-8': '$L{=}5$--$8$', '>=9': '$L{\\ge}9$'}


def load(name):
    return json.loads((RD / name).read_text())


def compile_pdf(body: str, name: str) -> None:
    doc = (
        "\\documentclass[preview,border=3pt]{standalone}\n"
        "\\usepackage{booktabs,amsmath,multirow}\n"
        "\\usepackage[T1]{fontenc}\n"
        "\\begin{document}\n"
        "\\footnotesize\\setlength{\\tabcolsep}{4pt}\n"
        f"{body}\n"
        "\\end{document}\n")
    with tempfile.TemporaryDirectory() as td:
        tex = Path(td) / f'{name}.tex'
        tex.write_text(doc)
        r = subprocess.run(['pdflatex', '-interaction=nonstopmode',
                            '-halt-on-error', tex.name],
                           cwd=td, capture_output=True, text=True)
        if r.returncode != 0:
            tail = '\n'.join(r.stdout.splitlines()[-25:])
            raise RuntimeError(f'{name}: pdflatex failed\n{tail}')
        (TABLES / f'{name}.pdf').write_bytes((Path(td) / f'{name}.pdf').read_bytes())
    print(f'wrote tables/{name}.pdf')


sla = load('span_length_analysis.json')['datasets']


# ------------------------------------------------- span-length stratification
def tbl_span_length():
    rows = []
    for ds in ORDER:
        u = sla[ds]['per_unit'][0]
        strat = u['stratified_reliability']['field_crc']
        elig = u['boundary_audit_eligibility_by_length']['by_bucket']
        cells = []
        for b in BUCKETS:
            s = strat.get(b)
            e = elig.get(b)
            if s is None:
                cells.append('--- & --- & ---')
            else:
                er = f"{e['informative_rate'] * 100:.0f}" if e else '---'
                cells.append(f"{s['n_spans']} & {s['full_span_coverage']:.3f} & {er}")
        sp = u['correlations']['spearman_length_vs_max_score']
        rows.append(f"{NICE[ds]} & " + ' & '.join(cells) + f" & {sp:.2f} \\\\")
    header = ' & '.join(
        f"\\multicolumn{{3}}{{c}}{{{BUCKET_TEX[b]}}}" for b in BUCKETS)
    sub = ' & '.join(['$n$ & cov & el.\\%'] * len(BUCKETS))
    return (
        "\\begin{tabular}{l" + "rrr" * len(BUCKETS) + "r}\n\\toprule\n"
        f" & {header} & \\multirow{{2}}{{*}}{{$\\rho_{{L,S}}$}} \\\\\n"
        f"Dataset & {sub} & \\\\\n\\midrule\n"
        + '\n'.join(rows) +
        "\n\\bottomrule\n\\end{tabular}")


# ------------------------------------------------------ aggregation ablation
def tbl_aggregation():
    label = {'max': '$\\max$ (ours)', 'mean': 'mean',
             'trimmed_max': '2nd-largest', 'q90': 'within-span $Q_{0.9}$'}
    rows = []
    for ds in ORDER:
        agg = sla[ds]['per_unit'][0]['aggregation_ablation']
        for i, kind in enumerate(['max', 'mean', 'trimmed_max', 'q90']):
            a = agg[kind]
            pre = NICE[ds] if i == 0 else ''
            rows.append(
                f"{pre} & {label[kind]} & {a['own_event_rate']:.3f} & "
                f"{a['true_full_span_coverage']:.3f} & "
                f"{a['true_doc_field_miss_risk']:.3f} & "
                f"{a['avg_token_set_size']:.2f} \\\\")
        rows.append('\\addlinespace')
    return (
        "\\begin{tabular}{llrrrr}\n\\toprule\n"
        "Dataset & Span score & Own-event & Full-span & Doc field-miss & Avg.\\ set \\\\\n"
        " & (pooled CRC) & rate & coverage & risk & size \\\\\n\\midrule\n"
        + '\n'.join(rows[:-1]) +
        "\n\\bottomrule\n\\end{tabular}")


# ------------------------------------------------------- CORD interpolation
def tbl_interp():
    interp = load('spancrc_interpolation.json')['datasets']['cord']['per_unit'][0]
    label = {'pooled_crc': 'pooled CRC ($\\tau{=}0$)',
             'field_crc_m0_fallback': 'field CRC + $m_0$ fallback (deployed)',
             'interp_tau_0.25': 'interpolated $\\tau{=}0.25$',
             'interp_tau_0.50': 'interpolated $\\tau{=}0.50$',
             'interp_tau_0.75': 'interpolated $\\tau{=}0.75$',
             'field_crc_no_fallback': 'field CRC, no fallback ($\\tau{=}1$)',
             'crc_interp_tau_0.25': 'level-calibrated $\\tau{=}0.25$ (guaranteed)',
             'crc_interp_tau_0.50': 'level-calibrated $\\tau{=}0.50$ (guaranteed)',
             'crc_interp_tau_0.75': 'level-calibrated $\\tau{=}0.75$ (guaranteed)'}
    rows = []
    for r in interp['rules']:
        o = r['overall']
        rows.append(
            f"{label[r['rule']]} & {o['full_span_coverage']:.3f} & "
            f"{o['doc_field_miss_risk']:.3f} & {o['span_covgap_pp']:.1f} & "
            f"{r['avg_set_size']:.2f} & {r['p90_set_size']:.0f} & "
            f"{r['empty_set_rate'] * 100:.1f} \\\\")
    return (
        "\\begin{tabular}{lrrrrrr}\n\\toprule\n"
        "Rule & Full-span & Doc field- & Span Cov- & Avg.\\ set & p90 & Empty \\\\\n"
        " & coverage & miss risk & Gap (pp) & size & size & (\\%) \\\\\n\\midrule\n"
        + '\n'.join(rows) +
        "\n\\bottomrule\n\\end{tabular}")


# ------------------------------------------------------- operational sets
def tbl_operational():
    rows = []
    for ds in ORDER:
        op = sla[ds]['per_unit'][0]['operational_set_size']['field_crc']
        rows.append(
            f"{NICE[ds]} & {op['mean']:.2f} & {op['p90']:.0f} & {op['p95']:.0f} & "
            f"{op['empty_rate'] * 100:.1f} & {op['singleton_rate'] * 100:.1f} & "
            f"{op['multi_rate'] * 100:.1f} & "
            f"{op['span_all_singleton_rate'] * 100:.1f} & "
            f"{op['span_any_multi_rate'] * 100:.1f} \\\\")
    return (
        "\\begin{tabular}{lrrrrrrrr}\n\\toprule\n"
        " & \\multicolumn{3}{c}{Set size} & \\multicolumn{3}{c}{Token rate (\\%)} "
        "& \\multicolumn{2}{c}{Span rate (\\%)} \\\\\n"
        "\\cmidrule(lr){2-4}\\cmidrule(lr){5-7}\\cmidrule(lr){8-9}\n"
        "Dataset & mean & p90 & p95 & empty & single & multi & all-single & any-multi \\\\\n"
        "\\midrule\n" + '\n'.join(rows) +
        "\n\\bottomrule\n\\end{tabular}")


# ---------------------------------------------------------- position masks
def tbl_masks():
    masks = load('boundary_escape_masks.json')['datasets']
    rows = []
    for ds in ORDER:
        m = masks[ds]
        rej = ', '.join(r.replace('::', ' (') + ')'
                        for r in m['rejected_pairs_holm']) or 'none'
        rej = rej.replace('_', '\\_')
        ne = m['per_mask']['edges']['n_fields_tested']
        ni = m['per_mask']['interior']['n_fields_tested']
        rows.append(f"{NICE[ds]} & {ne} & {ni} & {m['n_tests_in_family']} & "
                    f"{rej} \\\\")
    return (
        "\\begin{tabular}{lrrrl}\n\\toprule\n"
        " & \\multicolumn{2}{c}{Fields tested} & & \\\\\n"
        "\\cmidrule(lr){2-3}\n"
        "Dataset & edges & interior & Family size & Holm rejections \\\\\n"
        "\\midrule\n" + '\n'.join(rows) +
        "\n\\bottomrule\n\\end{tabular}")


# --------------------------------------- backbone generality + clean protocol
def tbl_generality():
    """Large (committed lineage) vs Base (new runs) and the clean-calibration
    WildReceipt protocol; emitted only when the GPU artifacts exist."""
    import subprocess as _sp  # noqa: F401  (keep imports local & obvious)
    bb = REPO / 'results' / 'backbone_generality'
    tc = REPO / 'results' / 'traincal' / 'span_crc_wildreceipt_traincal.json'
    if not ((bb / 'span_crc_funsd.json').exists()
            and (bb / 'span_crc_wildreceipt.json').exists() and tc.exists()):
        return None

    def agg(path):
        d = json.loads(Path(path).read_text())['aggregate']
        return (d['span_cov_token_marginal']['mean'],
                d['span_cov_field_crc']['mean'],
                d['ev_doc_risk_token_marginal']['mean'],
                d['ev_doc_risk_field_crc']['mean'],
                d['avg_set_size_token_marginal']['mean'],
                d['avg_set_size_field_crc']['mean'])

    import numpy as _np

    def multi(paths):
        """mean +- sd across seeds if all present, else None."""
        if not all(Path(p).exists() for p in paths):
            return None
        vals = _np.array([agg(p) for p in paths])
        return vals.mean(0), vals.std(0)

    def fmt_row(label, path=None, ms=None):
        if ms is None:
            ct, cf, rt, rf, st, sf = agg(path)
            return (f"{label} & {ct:.3f} & {cf:.3f} & {rt:.3f} & {rf:.3f} & "
                    f"{st:.2f} & {sf:.2f} \\\\")
        m, s = ms
        return (f"{label} & {m[0]:.3f}$\\pm${s[0]:.3f} & {m[1]:.3f}$\\pm${s[1]:.3f} & "
                f"{m[2]:.3f}$\\pm${s[2]:.3f} & {m[3]:.3f}$\\pm${s[3]:.3f} & "
                f"{m[4]:.2f}$\\pm${s[4]:.2f} & {m[5]:.2f}$\\pm${s[5]:.2f} \\\\")

    rows = []
    for ds, nice in (('sroie', 'SROIE'), ('funsd', 'FUNSD'),
                     ('wildreceipt', 'WildReceipt')):
        if not (bb / f'span_crc_{ds}.json').exists():
            continue
        rows.append(fmt_row(f'{nice} / Large', RD / f'span_crc_{ds}.json'))
        ms = multi([bb / f'span_crc_{ds}.json',
                    bb / 's123' / f'span_crc_{ds}.json',
                    bb / 's456' / f'span_crc_{ds}.json'])
        if ms is not None:
            rows.append(fmt_row(f'{nice} / Base (3 seeds)', ms=ms))
        else:
            rows.append(fmt_row(f'{nice} / Base', bb / f'span_crc_{ds}.json'))
    rows.append('\\addlinespace')
    tcd = REPO / 'results' / 'traincal'
    ms = multi([tcd / f'b8_s{s}' / 'span_crc_wildreceipt_traincal.json'
                for s in (42, 123, 456)])
    if ms is not None:
        rows.append(fmt_row('WildReceipt / Large, clean-cal (3 seeds)', ms=ms))
    else:
        rows.append(fmt_row('WildReceipt / Large, clean-cal', tc))
    return (
        "\\begin{tabular}{lrrrrrr}\n\\toprule\n"
        " & \\multicolumn{2}{c}{Full-span cov.} & \\multicolumn{2}{c}{Doc field-miss risk}"
        " & \\multicolumn{2}{c}{Avg.\\ set size} \\\\\n"
        "\\cmidrule(lr){2-3}\\cmidrule(lr){4-5}\\cmidrule(lr){6-7}\n"
        "Setting & token & field & token & field & token & field \\\\\n"
        "\\midrule\n" + '\n'.join(rows) +
        "\n\\bottomrule\n\\end{tabular}")


if __name__ == '__main__':
    compile_pdf(tbl_span_length(), 'supp_span_length')
    compile_pdf(tbl_aggregation(), 'supp_aggregation')
    compile_pdf(tbl_interp(), 'supp_interp')
    compile_pdf(tbl_operational(), 'supp_operational')
    compile_pdf(tbl_masks(), 'supp_masks')
    g = tbl_generality()
    if g is not None:
        compile_pdf(g, 'supp_generality')
    else:
        print('generality artifacts not present yet; supp_generality skipped')
