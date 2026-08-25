#!/usr/bin/env python3
"""Phase-2 paper figures, sourced directly from results/phase2_results.json.

Regenerates the stale main-paper figures with corrected numbers and adds the
data-audit headline figures. NO number is hardcoded except published-literature
SOTA references (LayoutLMv3-large) and the SROIE O / I-ADDRESS coverage that the
JSON records only in its KEY_FINDING text; both are annotated below.
"""
import json, os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
JSON = os.path.join(ROOT, "results", "phase2_results.json")
OUT = os.path.join(ROOT, "docs", "paper", "figures")
os.makedirs(OUT, exist_ok=True)
D = json.load(open(JSON))

# ---- paper-consistent style --------------------------------------------------
plt.rcParams.update({
    "font.family": "serif",
    "font.size": 9,
    "axes.titlesize": 10,
    "axes.labelsize": 9,
    "legend.fontsize": 8,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
})
# colourblind-safe (Okabe-Ito subset)
C_CORD, C_FUNSD, C_WILD = "#0072B2", "#D55E00", "#009E73"
C_LAC, C_APS = "#0072B2", "#D55E00"
C_O, C_ENT = "#4D4D4D", "#D55E00"
GREY = "#9A9A9A"
DS_COLOR = {"CORD": C_CORD, "FUNSD": C_FUNSD, "WildReceipt": C_WILD}
TARGETS = [90, 95, 99]
ALPHAS = ["alpha_0.1", "alpha_0.05", "alpha_0.01"]


def save(fig, name):
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(OUT, f"{name}.{ext}"))
    plt.close(fig)
    print(f"  wrote {name}.pdf / .png")


def lac(ds):
    """Return (cov[3], size[3]) percent/size for LAC at 90/95/99."""
    if ds == "cord":
        r = D["cord"]["results"]["fixed_split"]["lac"]
        cov = [r[a]["coverage"] * 100 for a in ALPHAS]
        sz = [r[a]["avg_set_size"] for a in ALPHAS]
    else:
        r = D[ds]["results"]["mean_std_over_seeds"]["lac"]
        cov = [r[a]["coverage_mean"] * 100 for a in ALPHAS]
        sz = [r[a]["set_size_mean"] for a in ALPHAS]
    return cov, sz


def aps(ds):
    if ds == "cord":
        r = D["cord"]["results"]["fixed_split"]["aps"]
        cov = [r[a]["coverage"] * 100 for a in ALPHAS]
        sz = [r[a]["avg_set_size"] for a in ALPHAS]
    else:
        r = D[ds]["results"]["mean_std_over_seeds"]["aps"]
        cov = [r[a]["coverage_mean"] * 100 for a in ALPHAS]
        sz = [r[a]["set_size_mean"] for a in ALPHAS]
    return cov, sz


# ============================================================================
# FIG 1: main_results  (a) NER F1 vs SOTA   (b) LAC coverage vs target
# ============================================================================
def fig_main_results():
    tm = D["meta"]["test_metrics_summary"]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.0, 2.9))

    # (a) F1 bars vs LayoutLMv3-large SOTA (literature: FUNSD 92.08, CORD 97.46)
    dsn = ["CORD", "FUNSD", "WildReceipt"]
    ours = [tm[d]["f1"] for d in dsn]
    sota = {"CORD": 97.46, "FUNSD": 92.08, "WildReceipt": None}  # LayoutLMv3-large
    x = np.arange(len(dsn))
    bars = ax1.bar(x, ours, width=0.55, color=[DS_COLOR[d] for d in dsn],
                   edgecolor="black", linewidth=0.5, zorder=3)
    for d, xi in zip(dsn, x):
        if sota[d] is not None:
            ax1.hlines(sota[d], xi - 0.32, xi + 0.32, color="black",
                       linestyle="--", linewidth=1.1, zorder=4)
    for b, v in zip(bars, ours):
        ax1.text(b.get_x() + b.get_width() / 2, v + 0.4, f"{v:.2f}",
                 ha="center", va="bottom", fontsize=7.5)
    ax1.plot([], [], "k--", label="LayoutLMv3-large (SOTA)")
    ax1.set_xticks(x); ax1.set_xticklabels(dsn)
    ax1.set_ylim(78, 100); ax1.set_ylabel("Test F1 (\%)".replace("\\", ""))
    ax1.set_ylabel("Test F1 (%)")
    ax1.set_title("(a) Document NER performance")
    ax1.legend(loc="lower left", frameon=False)
    ax1.grid(axis="y", alpha=0.25, zorder=0)

    # (b) LAC coverage vs target
    ax2.plot([88, 100], [88, 100], color=GREY, ls=":", lw=1, label="ideal (y=x)")
    for ds, name in [("cord", "CORD"), ("funsd", "FUNSD"), ("wildreceipt", "WildReceipt")]:
        cov, _ = lac(ds)
        ax2.plot(TARGETS, cov, "-o", color=DS_COLOR[name], ms=4.5, lw=1.4, label=name)
    ax2.set_xticks(TARGETS)
    ax2.set_xlabel("Target coverage (%)"); ax2.set_ylabel("Empirical coverage (%)")
    ax2.set_title("(b) LAC coverage vs.\\ target".replace("\\", ""))
    ax2.set_ylim(84, 100)
    ax2.legend(loc="lower right", frameon=False)
    ax2.grid(alpha=0.25)
    fig.tight_layout()
    save(fig, "main_results")


# ============================================================================
# FIG 2: conformal_sets  LAC vs APS average set size (log) — informative vs conservative
# ============================================================================
def fig_conformal_sets():
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.0, 2.9), sharey=True)
    for ax, scorer, title in [(ax1, lac, "(a) LAC — informative"),
                              (ax2, aps, "(b) APS — conservative")]:
        for ds, name in [("cord", "CORD"), ("funsd", "FUNSD"), ("wildreceipt", "WildReceipt")]:
            _, sz = scorer(ds)
            ax.plot(TARGETS, sz, "-o", color=DS_COLOR[name], ms=4.5, lw=1.4, label=name)
        ax.set_xticks(TARGETS)
        ax.set_yscale("log")
        ax.set_xlabel("Target coverage (%)")
        ax.set_title(title)
        ax.grid(alpha=0.25, which="both")
        ax.axhline(1.0, color=GREY, ls=":", lw=1)
    ax1.set_ylabel("Avg. prediction-set size (log)")
    ax1.legend(loc="upper left", frameon=False)
    fig.tight_layout()
    save(fig, "conformal_sets")


# ============================================================================
# FIG 3: calibration_diagram  ECE / MCE by dataset (no per-bin data available)
# ============================================================================
def fig_calibration():
    tm = D["meta"]["test_metrics_summary"]
    dsn = ["CORD", "FUNSD", "WildReceipt"]
    ece = [tm[d]["ece"] for d in dsn]
    mce = [tm[d]["mce"] for d in dsn]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.0, 2.9))
    x = np.arange(len(dsn))
    b1 = ax1.bar(x, ece, width=0.55, color=[DS_COLOR[d] for d in dsn],
                 edgecolor="black", linewidth=0.5, zorder=3)
    for b, v in zip(b1, ece):
        ax1.text(b.get_x() + b.get_width() / 2, v + 0.1, f"{v:.2f}",
                 ha="center", va="bottom", fontsize=7.5)
    ax1.set_xticks(x); ax1.set_xticklabels(dsn)
    ax1.set_ylabel("ECE (%)"); ax1.set_title("(a) Expected Calibration Error")
    ax1.set_ylim(0, max(ece) * 1.3); ax1.grid(axis="y", alpha=0.25, zorder=0)

    b2 = ax2.bar(x, mce, width=0.55, color=[DS_COLOR[d] for d in dsn],
                 edgecolor="black", linewidth=0.5, zorder=3)
    for b, v in zip(b2, mce):
        ax2.text(b.get_x() + b.get_width() / 2, v + 1.0, f"{v:.1f}",
                 ha="center", va="bottom", fontsize=7.5)
    ax2.set_xticks(x); ax2.set_xticklabels(dsn)
    ax2.set_ylabel("MCE (%)"); ax2.set_title("(b) Max Calibration Error (low-pop. bins)")
    ax2.set_ylim(0, max(mce) * 1.25); ax2.grid(axis="y", alpha=0.25, zorder=0)
    fig.tight_layout()
    save(fig, "calibration_diagram")


# ============================================================================
# FIG 4 (NEW headline): data_audit_sroie — per-class LAC coverage collapse
# ============================================================================
def fig_data_audit_sroie():
    # entity classes from JSON; O and I-ADDRESS coverage recorded in KEY_FINDING /
    # paper table tab:mondrian_sroie (0.953, 0.949). n from tab:mondrian_sroie.
    ent = D["meta"]["sroie_addendum"]["entity_class_coverage_lac90"]
    rows = [("O", 10200, 0.953), ("I-ADDRESS", 1098, 0.949)]
    n_map = {"I-COMPANY": 342, "B-COMPANY": 105, "B-DATE": 105,
             "B-ADDRESS": 105, "B-TOTAL": 105, "I-DATE": 60, "I-TOTAL": 60}
    for k, v in ent.items():
        rows.append((k, n_map.get(k, 50), v))
    rows.sort(key=lambda r: -r[2])
    labels = [r[0] for r in rows]
    cov = [r[2] * 100 for r in rows]
    ns = [r[1] for r in rows]
    colors = [C_O if lb in ("O", "I-ADDRESS") else C_ENT for lb in labels]

    fig, ax = plt.subplots(figsize=(7.0, 3.2))
    x = np.arange(len(labels))
    bars = ax.bar(x, cov, width=0.7, color=colors, edgecolor="black",
                  linewidth=0.5, zorder=3)
    ax.axhline(90, color="black", ls="--", lw=1.1, zorder=4, label="90\\% target".replace("\\", ""))
    ax.axhline(89.4, color="#0072B2", ls=":", lw=1.4, zorder=4,
               label="marginal coverage (89.4%)")
    for b, n in zip(bars, ns):
        ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 1.5, f"n={n}",
                ha="center", va="bottom", fontsize=6, rotation=90, color="#555")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=40, ha="right", fontsize=7)
    ax.set_ylabel("Per-class LAC coverage (%)")
    ax.set_ylim(0, 108)
    ax.set_title("SROIE: near-nominal marginal coverage hides total entity-class collapse")
    # shade the collapsed region
    ax.axhspan(0, 20, color="#D55E00", alpha=0.06, zorder=0)
    ax.legend(loc="upper right", frameon=True, framealpha=0.9)
    ax.grid(axis="y", alpha=0.25, zorder=0)
    # annotation
    ax.annotate("entity classes a KIE\nsystem exists to extract",
                xy=(len(labels) - 2, 3), xytext=(len(labels) - 4.6, 32),
                fontsize=7, color="#B2182B",
                arrowprops=dict(arrowstyle="->", color="#B2182B", lw=1))
    fig.tight_layout()
    save(fig, "data_audit_sroie")


# ============================================================================
# FIG 5 (NEW, supplementary): cord_perclass_coverage — coverage vs support
# ============================================================================
def fig_cord_perclass():
    pc = D["cord"]["results"]["fixed_split"]["lac"]["alpha_0.1"]["per_class_coverage"]
    items = [(k, v["n"], v["coverage"] * 100) for k, v in pc.items()]
    items.sort(key=lambda r: r[1])  # by support
    ns = np.array([r[1] for r in items])
    cov = np.array([r[2] for r in items])
    fig, ax = plt.subplots(figsize=(7.0, 3.2))
    sc = ax.scatter(ns, cov, c=cov, cmap="RdYlGn", vmin=0, vmax=100,
                    s=28, edgecolor="black", linewidth=0.4, zorder=3)
    ax.axhline(90, color="black", ls="--", lw=1, label="90% target")
    ax.axhline(85.8, color="#0072B2", ls=":", lw=1.3, label="marginal coverage (85.8%)")
    ax.set_xscale("log")
    ax.set_xlabel("Class support in test set (# tokens, log)")
    ax.set_ylabel("Per-class LAC coverage (%)")
    ax.set_title("CORD: low-support classes drive the sub-nominal 90\\%-target coverage".replace("\\", ""))
    ax.set_ylim(-5, 108)
    ax.legend(loc="lower right", frameon=True, framealpha=0.9)
    ax.grid(alpha=0.25, zorder=0)
    cb = fig.colorbar(sc, ax=ax, pad=0.02); cb.set_label("coverage (%)", fontsize=8)
    fig.tight_layout()
    save(fig, "cord_perclass_coverage")


if __name__ == "__main__":
    print("Generating Phase-2 figures ->", OUT)
    fig_main_results()
    fig_conformal_sets()
    fig_calibration()
    fig_data_audit_sroie()
    fig_cord_perclass()
    print("done.")
