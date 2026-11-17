"""Render the deck's display equations as transparent high-DPI PNGs (matplotlib mathtext).

Every equation is transcribed from the camera-ready main paper (Eqs. 1-4, Sec. 3.3-3.4)
or supplement (Prop. 3, Sec. S4.3). Output: source/assets/eq_*.png
"""
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

matplotlib.rcParams["mathtext.fontset"] = "cm"
INK = "#1E293B"
OUT = os.path.join(os.path.dirname(__file__), "assets")

EQS = {
    # Eq. 1
    "eq_span_score": r"$S(s)\;=\;\max_{a\leq i<b}\;\left(1-p(g_i\mid x_i)\right)$",
    # Eq. 2
    "eq_doc_risk": r"$R_j(\lambda)\;=\;\dfrac{1}{|\mathcal{S}_j|}\sum_{s\in\mathcal{S}_j}\mathbf{1}\left[S(s)>\lambda\right]$",
    # Eq. 3
    "eq_lambda_hat": r"$\hat{\lambda}\;=\;\inf\left\{\lambda:\;\dfrac{n\hat{R}(\lambda)+1}{n+1}\leq\alpha\right\}$",
    # guarantee (Sec. 3.3)
    "eq_guarantee": r"$\mathbb{E}\left[R_{n+1}(\hat{\lambda})\right]\;\leq\;\alpha$",
    # candidate-label rule (Sec. 3.3)
    "eq_candidate": r"$C(x_i)\;=\;\{\,y:\;1-p(y\mid x_i)\leq\hat{\lambda}_{\mathrm{type}(y)}\,\}$",
    # escape set + statistic (Eq. 4)
    "eq_escape": r"$E(s)=\{\,i\in[a,b):\;1-p(g_i\mid x_i)>\tau\,\},\qquad W_t=\sum_{s\in\mathcal{I}_t}\left|E(s)\cap B_s\right|$",
    # Prop. 3 (Supp. S4.3)
    "eq_interp": r"$\mathrm{thr}_t(\mu)=\min\{1,\;\tau\lambda_t^{A}+(1-\tau)\,\mu\}$",
}

def render(name, tex, size=30):
    fig = plt.figure(figsize=(0.01, 0.01))
    fig.text(0, 0, tex, fontsize=size, color=INK)
    fig.savefig(os.path.join(OUT, name + ".png"), dpi=300, transparent=True,
                bbox_inches="tight", pad_inches=0.04)
    plt.close(fig)

if __name__ == "__main__":
    os.makedirs(OUT, exist_ok=True)
    for k, v in EQS.items():
        render(k, v)
    print("rendered", len(EQS))
