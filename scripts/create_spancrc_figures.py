#!/usr/bin/env python3
"""Figures for the SpanCRC / boundary-escape sections.

Every value is read from `results/local_20260803/span_crc_<ds>.json`, which is
itself written by `scripts/span_crc.py` only after that script reproduces the
committed token-level span coverage to 1e-4. Nothing here is hand-entered.

    python3 scripts/create_spancrc_figures.py \
        --results_dir results/local_20260803 --out_dir docs/paper-wacv2027/figures

Produces:
  spancrc_main.pdf      (a) span coverage token-marginal vs field CRC, per dataset
                        (b) boundary-escape edge enrichment per field, BH-marked
  spancrc_setsize.pdf   set-size cost of the field-conditional threshold
"""

from __future__ import annotations

import argparse
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# Categorical slots 1 and 2 of the validated reference palette; the pair clears
# the lightness band, chroma floor, CVD separation (dE 24.7 protan) and the
# normal-vision floor (dE 33.6) against a light surface.
C_TOKEN = "#2a78d6"   # slot 1, blue   -> token-marginal threshold
C_FIELD = "#eb6834"   # slot 2, orange -> field-conditional SpanCRC
INK = "#0b0b0b"
INK_MUTED = "#52514e"
GRID = "#d8d7d2"
SURFACE = "#ffffff"

plt.rcParams.update({
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "font.family": "serif",
    "font.serif": ["Times New Roman", "Nimbus Roman", "DejaVu Serif"],
    "font.size": 8,
    "axes.edgecolor": INK_MUTED,
    "axes.linewidth": 0.6,
    "xtick.color": INK_MUTED,
    "ytick.color": INK_MUTED,
    "text.color": INK,
    "axes.labelcolor": INK,
    "figure.facecolor": SURFACE,
    "axes.facecolor": SURFACE,
    "hatch.linewidth": 0.6,
})

ORDER = ["sroie", "sroie_repaired", "wildreceipt", "funsd", "cord"]
PRETTY = {
    "sroie": "SROIE",
    "sroie_repaired": "SROIE (repaired)",
    "wildreceipt": "WildReceipt",
    "funsd": "FUNSD",
    "cord": "CORD",
}
# axis-tick variants, kept short so the five categories do not collide
SHORT = {
    "sroie": "SROIE",
    "sroie_repaired": "SROIE-rep.",
    "wildreceipt": "WildRec.",
    "funsd": "FUNSD",
    "cord": "CORD",
}


def load(results_dir, ds):
    with open(os.path.join(results_dir, f"span_crc_{ds}.json")) as fh:
        return json.load(fh)


def fig_main(data, out_dir, alpha=0.10):
    target = 1.0 - alpha
    fig, (ax1, ax2) = plt.subplots(
        1, 2, figsize=(7.0, 2.15), gridspec_kw={"width_ratios": [1.15, 1.0]}
    )

    # ---- (a) span coverage: token-marginal vs field-conditional CRC ----
    names = [d for d in ORDER if d in data]
    tok = [data[d]["aggregate"]["span_cov_token_marginal"]["mean"] for d in names]
    tok_e = [data[d]["aggregate"]["span_cov_token_marginal"]["std"] for d in names]
    fld = [data[d]["aggregate"]["span_cov_field_crc"]["mean"] for d in names]
    fld_e = [data[d]["aggregate"]["span_cov_field_crc"]["std"] for d in names]

    x = np.arange(len(names))
    w = 0.36
    # 2px-equivalent surface gap between adjacent fills -> small inset on width
    ax1.bar(x - w / 2 - 0.012, tok, w, yerr=tok_e, capsize=1.8, color=C_TOKEN,
            label="Token-marginal threshold", zorder=3,
            error_kw={"elinewidth": 0.7, "ecolor": INK_MUTED})
    ax1.bar(x + w / 2 + 0.012, fld, w, yerr=fld_e, capsize=1.8, color=C_FIELD,
            label="Field-conditional SpanCRC", zorder=3, hatch="////",
            edgecolor=SURFACE, linewidth=0.0,
            error_kw={"elinewidth": 0.7, "ecolor": INK_MUTED})

    ax1.axhline(target, color=INK, lw=0.9, ls=(0, (3, 2)), zorder=4)
    ax1.annotate(f"target {target:.2f}", xy=(len(names) - 0.42, target),
                 xytext=(0, 3), textcoords="offset points",
                 ha="right", va="bottom", fontsize=6.6, color=INK)

    # direct-label only the two cells that carry the argument
    for xi, (t, f) in enumerate(zip(tok, fld)):
        if t < 0.30:
            ax1.annotate(f"{t:.3f}", xy=(xi - w / 2, t), xytext=(0, 2.5),
                         textcoords="offset points", ha="center", va="bottom",
                         fontsize=6.6, color=INK)
            ax1.annotate(f"{f:.3f}", xy=(xi + w / 2, f), xytext=(0, 2.5),
                         textcoords="offset points", ha="center", va="bottom",
                         fontsize=6.6, color=INK)

    ax1.set_xticks(x)
    ax1.set_xticklabels([SHORT[d] for d in names], fontsize=6.4,
                        rotation=16, ha="right", rotation_mode="anchor")
    ax1.set_ylabel("Full-span coverage", fontsize=8)
    ax1.set_ylim(0, 1.30)
    ax1.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
    ax1.grid(axis="y", color=GRID, lw=0.5, zorder=0)
    ax1.set_axisbelow(True)
    for s in ("top", "right"):
        ax1.spines[s].set_visible(False)
    ax1.legend(loc="upper left", fontsize=6.4, frameon=False,
               handlelength=1.1, borderaxespad=0.15, labelspacing=0.3)
    ax1.set_title("(a) Empirical full-span coverage",
                  fontsize=8, color=INK, pad=5, loc="left")

    # ---- (b) boundary-escape enrichment per field ----
    rows = []
    for ds in names:
        be = data[ds]["per_unit"][0]["boundary_escape"]
        for fname, st in be["fields"].items():
            if st.get("edge_enrichment") is None:
                continue
            # Holm is the primary decision (FWER under arbitrary dependence)
            rows.append((ds, fname, st["edge_enrichment"],
                         bool(st.get("holm_rejected"))))
    # keep the most-enriched field per dataset plus every rejected field
    keep, seen = [], set()
    for ds in names:
        sub = sorted([r for r in rows if r[0] == ds], key=lambda r: -r[2])
        for r in sub:
            if r[3] or (ds not in seen):
                keep.append(r)
                seen.add(ds)
    keep = sorted(keep, key=lambda r: r[2])

    ypos = np.arange(len(keep))
    for yi, (ds, fname, enr, rej) in zip(ypos, keep):
        col = C_FIELD if rej else INK_MUTED
        ax2.plot([1.0, enr], [yi, yi], color=col, lw=1.2, solid_capstyle="round",
                 zorder=2, alpha=0.9)
        ax2.plot([enr], [yi], marker="o" if rej else "s", ms=5.0 if rej else 4.0,
                 color=col, zorder=3,
                 markeredgecolor=SURFACE, markeredgewidth=0.8)

    ax2.axvline(1.0, color=INK, lw=0.9, ls=(0, (3, 2)), zorder=1)
    ax2.annotate("no positional\nstructure", xy=(1.0, len(keep) - 0.55),
                 xytext=(4, 0), textcoords="offset points",
                 ha="left", va="center", fontsize=6.2, color=INK_MUTED)

    ax2.set_yticks(ypos)
    ax2.set_yticklabels(
        [f"{PRETTY[ds]} · {fn}" for ds, fn, _, _ in keep],
        fontsize=6.4)
    ax2.set_xlabel("Edge enrichment  $W_t\\,/\\,\\mathbb{E}[W_t]$", fontsize=8)
    ax2.set_xlim(0, max(r[2] for r in keep) * 1.18)
    ax2.grid(axis="x", color=GRID, lw=0.5, zorder=0)
    ax2.set_axisbelow(True)
    for s in ("top", "right", "left"):
        ax2.spines[s].set_visible(False)
    ax2.tick_params(axis="y", length=0)

    # legend by state, with shape as the secondary encoding
    h_rej = plt.Line2D([], [], marker="o", ms=5, color=C_FIELD, lw=1.2,
                       markeredgecolor=SURFACE, markeredgewidth=0.8,
                       label="rejected (Holm, FWER 10%)")
    h_not = plt.Line2D([], [], marker="s", ms=4, color=INK_MUTED, lw=1.2,
                       markeredgecolor=SURFACE, markeredgewidth=0.8,
                       label="not rejected")
    ax2.legend(handles=[h_rej, h_not], loc="lower right", fontsize=6.4,
               frameon=False, handlelength=1.4, borderaxespad=0.2)
    ax2.set_title("(b) Boundary-escape test", fontsize=8, color=INK, pad=5, loc="left")

    fig.tight_layout(pad=0.4, w_pad=1.4)
    dst = os.path.join(out_dir, "spancrc_main.pdf")
    fig.savefig(dst, bbox_inches="tight", pad_inches=0.015)
    plt.close(fig)
    return dst


def fig_setsize(data, out_dir):
    names = [d for d in ORDER if d in data]
    fig, ax = plt.subplots(figsize=(3.35, 1.72))
    tok = [data[d]["aggregate"]["avg_set_size_token_marginal"]["mean"] for d in names]
    fld = [data[d]["aggregate"]["avg_set_size_field_crc"]["mean"] for d in names]
    x = np.arange(len(names))
    w = 0.36
    ax.bar(x - w / 2 - 0.012, tok, w, color=C_TOKEN, zorder=3,
           label="Token-marginal")
    ax.bar(x + w / 2 + 0.012, fld, w, color=C_FIELD, zorder=3,
           label="Field SpanCRC", hatch="////", edgecolor=SURFACE,
           linewidth=0.0)
    for xi, (t, f) in enumerate(zip(tok, fld)):
        ax.annotate(f"{(f - t) / t * 100:+.0f}%", xy=(xi, max(t, f)),
                    xytext=(0, 2.5), textcoords="offset points", ha="center",
                    va="bottom", fontsize=6.2, color=INK_MUTED)
    ax.set_xticks(x)
    ax.set_xticklabels([SHORT[d] for d in names],
                       fontsize=6.0, rotation=14, ha="right")
    ax.set_ylabel("Avg. set size", fontsize=8)
    ax.grid(axis="y", color=GRID, lw=0.5, zorder=0)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.legend(fontsize=6.2, frameon=False, handlelength=1.1, loc="upper left")
    fig.tight_layout(pad=0.3)
    dst = os.path.join(out_dir, "spancrc_setsize.pdf")
    fig.savefig(dst, bbox_inches="tight", pad_inches=0.015)
    plt.close(fig)
    return dst


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results_dir", default="results/local_20260803")
    ap.add_argument("--out_dir", default="docs/paper-wacv2027/figures")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    data = {}
    for ds in ORDER:
        p = os.path.join(args.results_dir, f"span_crc_{ds}.json")
        if os.path.exists(p):
            data[ds] = load(args.results_dir, ds)
    if not data:
        raise SystemExit(f"no span_crc_*.json under {args.results_dir}; "
                         "run scripts/span_crc.py --all first")

    for dst in (fig_main(data, args.out_dir), fig_setsize(data, args.out_dir)):
        print("wrote", dst)


if __name__ == "__main__":
    main()
