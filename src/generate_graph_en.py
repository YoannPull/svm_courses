# -*- coding: utf-8 -*-
"""
SVM – Compact teaching figures
- Colored backgrounds on hyperplanes
- Segment M perpendicular to the hyperplane (length M = 1/||w||)
- Soft margin + slack vectors ξ_i (readable selection, properly circled SVs)
- "Kernel trick" (circles): 2D view of points + 3D view (explicit r^2 lift) with linear hyperplane
- SVR: ε-insensitive tube + slacks ξ / ξ*

Figure catalog (files saved to FIG_DIR):
1) svm_1_separable_points.png
   - Separable scatter (points only, tight zoom)
2) svm_2_nonseparable_points.png
   - Non-separable scatter (moons), no boundary
3) svm_3_quasi_separable_points.png
   - Quasi-separable: 2 swapped points prevent a perfect hyperplane
4) svm_4_separable_hyperplan_fond.png
   - Linear SVM (hard-margin approx): hyperplane + colored decision regions
5) svm_5_perceptron_multi_hyperplans.png
   - Perceptron: examples of possible hyperplanes (non-uniqueness)
6) svm_6_marges_sv_M.png
   - Margins f(x)=±1, support vectors circled, segment M perpendicular to f(x)=0
7) svm_7_soft_margin_slack.png
   - Soft margin (moderate C): hyperplane + margins + slack arrows ξ_i
8) kernel_circles_points_2D.png
   - "Circles" dataset in 2D (points only)
9) kernel_circles_points_3D_with_plane.png
   - Polynomial lifting z=r² and linear hyperplane f=0 in 3D (no "bell")
10) svr_epsilon_tube_demo.png
    - SVR (RBF): prediction f(x), ε-insensitive tube ±ε, SVs, slacks ξ / ξ*
"""

# =========================
# Imports (deduplicated)
# =========================
import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import FancyArrowPatch, Rectangle
from sklearn.datasets import make_blobs, make_moons, make_circles
from sklearn.svm import SVC, SVR
from sklearn.linear_model import Perceptron
# (Matplotlib >= 3 handles projection="3d" without importing Axes3D explicitly)

# =========================
# Global options
# =========================
SAVE_FIGS     = True           # Save figures into FIG_DIR
FIG_DIR       = "figures_en"   # Output folder for images
FIG_DPI       = 300            # DPI of saved files
SHOW_TITLES   = False          # Show figure titles
SHOW_LEGENDS  = True           # Show legends
LEGEND_LOC    = "upper left"   # Legend location (everywhere)
PLOT_TRUE     = False          # False = "just save" (no on-screen display)

# Legend opacity / aesthetics
LEGEND_FRAME_ALPHA = 0.55      # legend box/frame alpha
LEGEND_ITEM_ALPHA  = 0.85      # markers/lines alpha in legend
LEGEND_TEXT_ALPHA  = 1.00      # legend text alpha

# "Slack" specifics (Figure 7)
SHOW_SLACK_LABELS  = True      # show ξ_i labels on arrows
MAX_SLACK_ARROWS   = 4         # max number of slack vectors (reduced)
MIN_SLACK_TO_DRAW  = 0.12      # threshold on ξ_i to draw (stricter)
SLACK_MIN_SEP      = 0.18      # min relative distance between selected arrows
SLACK_COLOR        = "#8b5cf6" # soft violet for slack arrows

# Zooms (axes)
PAD_SMALL   = 0.10   # for figs 1→4 (tight zoom)
PAD_DEFAULT = 0.22

# Output
if SAVE_FIGS:
    os.makedirs(FIG_DIR, exist_ok=True)

def maybe_save(fig, filename):
    """Save the figure if SAVE_FIGS=True."""
    if SAVE_FIGS:
        fig.savefig(os.path.join(FIG_DIR, filename), dpi=FIG_DPI, bbox_inches="tight")
    if not PLOT_TRUE:
        # Close to free memory (and avoid any window)
        plt.close(fig)
        plt.show = lambda *args, **kwargs: None

# =========================
# Global Matplotlib style
# =========================
plt.rcParams.update({
    "figure.dpi": 140,
    "figure.figsize": (7.2, 5.6),
    "axes.grid": True,
    "grid.alpha": 0.25,
    "axes.facecolor": "#f9fafb",
    "axes.edgecolor": "#e5e7eb",
    "axes.titleweight": "bold",
    "axes.titlesize": 14,
    "axes.labelsize": 12,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.frameon": True,
    "legend.framealpha": LEGEND_FRAME_ALPHA,
    "legend.facecolor": "white",
})

# Colors + RNG
rng = np.random.RandomState(42)
palette = {"neg": "#1f77b4", "pos": "#ff7f0e"}  # Blue / Orange

# =========================
# Helpers (reused everywhere)
# =========================
def tune_legend_alpha(leg,
                      alpha_items=LEGEND_ITEM_ALPHA,
                      alpha_text=LEGEND_TEXT_ALPHA,
                      alpha_frame=LEGEND_FRAME_ALPHA):
    """Homogeneous fine-tuning of legend element opacity."""
    if leg is None:
        return
    fr = leg.get_frame()
    if fr is not None:
        fr.set_alpha(alpha_frame)
    for h in getattr(leg, "legendHandles", []):
        try:
            h.set_alpha(alpha_items)
        except Exception:
            pass
    for txt in leg.get_texts():
        txt.set_alpha(alpha_text)

def set_limits(ax, X, pad=PAD_DEFAULT):
    """Axis limits with margin 'pad' around the data."""
    ax.set_xlim(X[:, 0].min()-pad, X[:, 0].max()+pad)
    ax.set_ylim(X[:, 1].min()-pad, X[:, 1].max()+pad)

def make_grid(X, pad=0.28, n=400):
    """Regular 2D mesh around the scatter X (±pad)."""
    x_min, x_max = X[:, 0].min()-pad, X[:, 0].max()+pad
    y_min, y_max = X[:, 1].min()-pad, X[:, 1].max()+pad
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, n),
                         np.linspace(y_min, y_max, n))
    return xx, yy, (x_min, x_max, y_min, y_max)

def plot_points(ax, X, y, title="", with_legend=True):
    """
    2-class scatter (0/1) with fixed colors.
    Graphic: point cloud; useful to compare before/after separation.
    """
    sc0 = ax.scatter(X[y == 0, 0], X[y == 0, 1], s=36, c=palette["neg"],
                     edgecolor="white", linewidth=0.7, label="Class 0")
    sc1 = ax.scatter(X[y == 1, 0], X[y == 1, 1], s=36, c=palette["pos"],
                     edgecolor="white", linewidth=0.7, label="Class 1")
    ax.set_xlabel("x₁"); ax.set_ylabel("x₂")
    if SHOW_TITLES and title:
        ax.set_title(title)
    if with_legend and SHOW_LEGENDS:
        leg = ax.legend(loc=LEGEND_LOC, framealpha=LEGEND_FRAME_ALPHA)
        tune_legend_alpha(leg)
    return sc0, sc1

def plot_decision_regions(ax, clf, X, pad=0.18, fill=False, boundary=True, margins=False):
    """
    Shows: (i) colored decision regions; (ii) boundary f(x)=0; (iii) margins f(x)=±1
    - pad: mesh zoom around the data
    - fill=True: fill decision regions
    - boundary=True: draws f(x)=0
    - margins=True: draws f(x)=±1
    """
    xx, yy, (x_min, x_max, y_min, y_max) = make_grid(X, pad=pad)
    Z = clf.decision_function(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)
    if fill:
        ax.contourf(xx, yy, (Z > 0).astype(int),
                    levels=[-0.5, 0.5, 1.5], alpha=0.10,
                    colors=[palette["neg"], palette["pos"]])
    if boundary:
        ax.contour(xx, yy, Z, levels=[0], linewidths=2.0, colors="#111111")
    if margins:
        ax.contour(xx, yy, Z, levels=[-1, 1], linestyles="--",
                   linewidths=1.6, colors="#111111")
    ax.set_xlim(x_min, x_max); ax.set_ylim(y_min, y_max)

def draw_margin_perpendicular(ax, w, b, X, color="blue"):
    """
    Draw a short segment perpendicular to the hyperplane f(x)=0, centered in the data region,
    and annotate it 'M' to illustrate the margin distance ~ 1/||w||.
    """
    w = np.asarray(w)
    norm_w = np.linalg.norm(w)
    if norm_w < 1e-12:
        return
    # point near center, lying on f(x)=0
    x_mid = 0.5 * (X[:, 0].min() + X[:, 0].max())
    y_mid = 0.5 * (X[:, 1].min() + X[:, 1].max())
    if abs(w[1]) > abs(w[0]):
        x0 = x_mid
        y0 = -(w[0]/w[1]) * x0 - b / w[1]
    else:
        y0 = y_mid
        x0 = -(w[1] * y0 + b) / w[0]
    p0 = np.array([x0, y0])            # on f(x)=0
    delta = w / (norm_w ** 2)          # length ~ 1/||w||
    p1 = p0 + delta
    ax.plot([p0[0], p1[0]], [p0[1], p1[1]], lw=2.0, color=color)
    # 'M' label, tangential offset
    t = np.array([-w[1], w[0]])
    if np.linalg.norm(t) > 0:
        t = t / np.linalg.norm(t)
    label_pos = (p0 + p1) / 2 + 0.12 * t
    ax.text(label_pos[0], label_pos[1], "M",
            fontsize=12, fontweight="bold",
            bbox=dict(boxstyle="round,pad=0.2", fc="none", ec="none"),
            ha="center", va="center")

# ---------- Slack utilities (Figure 7) ----------
def compute_slack(X, y01, w, b):
    """Compute ξ_i = max(0, 1 - y_i f(x_i)) with y_i ∈ {-1, +1} (from y ∈ {0,1})."""
    y_pm1 = 2 * y01 - 1  # 0/1 -> -1/+1
    f = X.dot(w) + b
    xi = np.maximum(0.0, 1.0 - y_pm1 * f)
    return xi, f, y_pm1

def draw_slack_vectors(ax, X, y01, w, b,
                       max_k=MAX_SLACK_ARROWS,
                       min_xi=MIN_SLACK_TO_DRAW,
                       min_sep_ratio=SLACK_MIN_SEP,
                       color=SLACK_COLOR,
                       annotate=SHOW_SLACK_LABELS):
    """
    For a few points violating the margin (ξ_i > 0), draw the arrow to the margin f(x)=y_i.
    Greedy selection for readability.
    """
    w = np.asarray(w)
    ww = np.dot(w, w)
    if ww <= 1e-14:
        return [], []

    xi, f, y_pm1 = compute_slack(X, y01, w, b)
    cand = np.where(xi > min_xi)[0]
    if cand.size == 0:
        return [], []

    # Min relative distance (as % of max span)
    scale = max(np.ptp(X[:, 0]), np.ptp(X[:, 1]))
    min_sep = max(1e-8, float(min_sep_ratio) * float(scale))

    # Sort by ξ descending, then spaced selection
    cand = cand[np.argsort(-xi[cand])]
    picked = []
    for i in cand:
        if len(picked) >= max_k:
            break
        if all(np.linalg.norm(X[i] - X[j]) >= min_sep for j in picked):
            picked.append(i)

    if len(picked) == 0:
        return [], []

    arrows = []
    labels = []
    t_vec = np.array([-w[1], w[0]])
    t_norm = np.linalg.norm(t_vec)
    t_vec = t_vec / t_norm if t_norm > 0 else np.array([0.0, 1.0])

    for j, i in enumerate(picked):
        delta_f = (y_pm1[i] - f[i])           # target - current in f-space
        delta_x = (delta_f / ww) * w          # 2D vector
        p0 = X[i]
        p1 = p0 + delta_x

        ax.annotate("", xy=p1, xytext=p0,
                    arrowprops=dict(arrowstyle="->", lw=2.0, color=color))

        if annotate:
            mid = (p0 + p1) / 2.0
            offset = (0.16 + 0.02*j) * ((-1)**j) * t_vec
            label = r"$\xi_{%d}$" % (j+1)
            ax.text(mid[0] + offset[0], mid[1] + offset[1], label,
                    fontsize=11, color=color,
                    ha="center", va="center",
                    bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="none", alpha=0.70))
        arrows.append((p0, p1))
        labels.append((i, xi[i]))
    return arrows, labels

# ---------- NEW: masks for SVs on / in the margin ----------
def sv_on_in_masks(clf, X, y01, C, tol_alpha=None, tol_margin=1e-3):
    """
    Return two boolean masks (len(X)):
      - sv_on : SVs on the margin (y f(x) ≈ 1)
      - sv_in : SVs inside the margin (y f(x) < 1)
    SVs are detected via alpha_i>0 with a tolerance robust to scale.
    """
    y = 2*y01 - 1
    f = clf.decision_function(X)  # f(x)
    alpha = np.zeros(len(X))
    if getattr(clf, "support_", None) is not None:
        alpha[clf.support_] = np.abs(clf.dual_coef_).ravel()

    # Robust tolerance: relative to observed alpha scale (not to C)
    if tol_alpha is None:
        maxa = float(np.max(alpha)) if alpha.size else 0.0
        tol_alpha = max(1e-12, 1e-6 * max(1.0, maxa))

    sv = alpha > tol_alpha
    sv_on = sv & (np.abs(y * f - 1.0) <= tol_margin)  # on the margin
    sv_in = sv & (y * f < 1.0 - tol_margin)           # inside the margin (or misclassified)
    return sv_on, sv_in

# ================================================================================
# 1) Separable — points only (tight zoom)
# ================================================================================
X_sep, y_sep = make_blobs(n_samples=46,
                          centers=[(-1.5, -1.5), (1.5, 1.5)],
                          cluster_std=[0.50, 0.50],
                          random_state=42)
fig1, ax1 = plt.subplots()
plot_points(ax1, X_sep, y_sep, title="Separable – points only")
set_limits(ax1, X_sep, pad=PAD_SMALL)
plt.tight_layout(); maybe_save(fig1, "svm_1_separable_points.png"); plt.show()

# ================================================================================
# 2) Non-separable — points only (tight zoom)
# ================================================================================
X_nsep, y_nsep = make_moons(n_samples=70, noise=0.22, random_state=42)
fig2, ax2 = plt.subplots()
plot_points(ax2, X_nsep, y_nsep, title="Non-separable – points only (no boundary)")
set_limits(ax2, X_nsep, pad=PAD_SMALL)
plt.tight_layout(); maybe_save(fig2, "svm_2_nonseparable_points.png"); plt.show()

# ================================================================================
# 3) Quasi-separable — 2 crossed points (tight zoom)
# ================================================================================
centers_qs = np.array([[-1.35, -1.35], [1.35, 1.35]])
X_qs, y_qs = make_blobs(n_samples=54, centers=centers_qs,
                        cluster_std=[0.50, 0.50], random_state=7)
idx0 = rng.choice(np.where(y_qs == 0)[0], size=2, replace=False)
idx1 = rng.choice(np.where(y_qs == 1)[0], size=2, replace=False)
X_qs[idx0] = centers_qs[1] + rng.normal(scale=0.10, size=(2, 2))
X_qs[idx1] = centers_qs[0] + rng.normal(scale=0.10, size=(2, 2))
fig3, ax3 = plt.subplots()
plot_points(ax3, X_qs, y_qs, title="Quasi-separable – 2 crossed points (no hyperplane)")
set_limits(ax3, X_qs, pad=PAD_SMALL)
plt.tight_layout(); maybe_save(fig3, "svm_3_quasi_separable_points.png"); plt.show()

# ================================================================================
# 4) Separable — hyperplane (colored background)
# ================================================================================
clf_sep_lin = SVC(kernel="linear", C=1e6).fit(X_sep, y_sep)  # hard-margin approx
fig4, ax4 = plt.subplots()
plot_decision_regions(ax4, clf_sep_lin, X_sep, pad=PAD_SMALL, fill=True, boundary=True, margins=False)
plot_points(ax4, X_sep, y_sep, title="Separable – hyperplane (linear SVM)")
plt.tight_layout(); maybe_save(fig4, "svm_4_separable_hyperplan_fond.png"); plt.show()

# ================================================================================
# 5) Perceptron — multiple hyperplanes
# ================================================================================
fig5, ax5 = plt.subplots()
sc0, sc1 = plot_points(ax5, X_sep, y_sep,
                       title="Perceptron – multiple hyperplanes (non-uniqueness)",
                       with_legend=False)
colors = ["#111111", "#6B7280", "#9CA3AF", "#374151", "#4B5563"]
for i, rs in enumerate([0, 1, 2, 3, 4]):
    percep = Perceptron(max_iter=2000, tol=None, random_state=rs, fit_intercept=True)
    percep.fit(X_sep, y_sep)
    w = percep.coef_[0]; b = percep.intercept_[0]
    xx = np.linspace(X_sep[:, 0].min()-PAD_DEFAULT, X_sep[:, 0].max()+PAD_DEFAULT, 200)
    if abs(w[1]) < 1e-12:
        continue
    yy = -(w[0]/w[1])*xx - b/w[1]
    ax5.plot(xx, yy, lw=1.6, alpha=0.9 - 0.12*i, c=colors[i])
proxy_line = Line2D([0], [0], color="#111111", lw=1.6, label="Possible hyperplanes (Perceptron)")
if SHOW_LEGENDS:
    leg5 = ax5.legend(handles=[sc0, sc1, proxy_line], loc=LEGEND_LOC,
                      framealpha=LEGEND_FRAME_ALPHA)
    tune_legend_alpha(leg5)
set_limits(ax5, X_sep, pad=PAD_DEFAULT)
plt.tight_layout(); maybe_save(fig5, "svm_5_perceptron_multi_hyperplans.png"); plt.show()

# ================================================================================
# 6) Separable — margins & support vectors + segment M
#       File: svm_6_marges_sv_M.png
# ================================================================================
C_hard = 1e6
clf_margin = SVC(kernel="linear", C=C_hard).fit(X_sep, y_sep)
w = clf_margin.coef_[0]; b = clf_margin.intercept_[0]
fig6, ax6 = plt.subplots()
plot_decision_regions(ax6, clf_margin, X_sep, pad=PAD_DEFAULT, fill=True, boundary=True, margins=True)
sc0, sc1 = plot_points(ax6, X_sep, y_sep,
                       title="Separable – margins, SVs and length M (1/||w||)",
                       with_legend=False)

# Circling SVs: black = on margin ; red = inside margin
sv_on_6, sv_in_6 = sv_on_in_masks(clf_margin, X_sep, y_sep, C_hard)
if np.any(sv_on_6):
    ax6.scatter(X_sep[sv_on_6, 0], X_sep[sv_on_6, 1], s=110, facecolors='none',
                edgecolors="#111111", linewidths=1.8, label="SV on margin")
if np.any(sv_in_6):
    ax6.scatter(X_sep[sv_in_6, 0], X_sep[sv_in_6, 1], s=110, facecolors='none',
                edgecolors="#ef4444", linewidths=1.8, label="Vectors inside margin")

# Annotate one SV + perpendicular segment M
sv_indices = np.where(sv_on_6 | sv_in_6)[0]
if sv_indices.size > 0:
    sv0 = X_sep[sv_indices[0]]
    ax6.annotate("Support vector",
                 xy=(sv0[0], sv0[1]), xytext=(sv0[0]+0.40, sv0[1]+0.40),
                 arrowprops=dict(arrowstyle="->", lw=1.2, color="#111111"),
                 fontsize=11)
draw_margin_perpendicular(ax6, w, b, X_sep, color="blue")

# Legend
if SHOW_LEGENDS:
    boundary_proxy = Line2D([0], [0], color="#111111", lw=2.0, label="Hyperplane f(x)=0")
    margin_proxy   = Line2D([0], [0], color="#111111", lw=1.6, ls="--", label="Margin (f(x)=±1)")
    handles = [sc0, sc1, boundary_proxy, margin_proxy]
    if np.any(sv_on_6):
        handles.append(Line2D([], [], marker="o", lw=0, markerfacecolor='none',
                              markeredgecolor="#111111", markersize=9, label="SV on margin"))
    if np.any(sv_in_6):
        handles.append(Line2D([], [], marker="o", lw=0, markerfacecolor='none',
                              markeredgecolor="#ef4444", markersize=9, label="Vectors inside margin"))
    leg6 = ax6.legend(handles=handles, loc=LEGEND_LOC, framealpha=LEGEND_FRAME_ALPHA)
    tune_legend_alpha(leg6)

set_limits(ax6, X_sep, pad=PAD_DEFAULT)
plt.tight_layout(); maybe_save(fig6, "svm_6_marges_sv_M.png"); plt.show()

# ================================================================================
# 7) Soft margin (linear SVM, moderate C) + slack vectors ξ_i
#       File: svm_7_soft_margin_slack.png
# ================================================================================
C_soft = 1.0
clf_soft = SVC(kernel="linear", C=C_soft).fit(X_qs, y_qs)
w_soft = clf_soft.coef_[0]; b_soft = clf_soft.intercept_[0]

fig7, ax7 = plt.subplots()
plot_decision_regions(ax7, clf_soft, X_qs, pad=PAD_DEFAULT, fill=True, boundary=True, margins=True)
sc0, sc1 = plot_points(ax7, X_qs, y_qs,
                       title=f"Soft margin – linear SVM (C={C_soft}) with slack vectors $\\xi_i$",
                       with_legend=False)

# Circling: black = on margin ; red = inside margin
sv_on_7, sv_in_7 = sv_on_in_masks(clf_soft, X_qs, y_qs, C_soft)
if np.any(sv_on_7):
    ax7.scatter(X_qs[sv_on_7, 0], X_qs[sv_on_7, 1], s=110, facecolors='none',
                edgecolors="#111111", linewidths=1.8, label="SV on margin")
if np.any(sv_in_7):
    ax7.scatter(X_qs[sv_in_7, 0], X_qs[sv_in_7, 1], s=110, facecolors='none',
                edgecolors="#ef4444", linewidths=1.8, label="Vectors inside margin")

# Slack arrows
_ = draw_slack_vectors(ax7, X_qs, y_qs, w_soft, b_soft,
                       max_k=MAX_SLACK_ARROWS,
                       min_xi=MIN_SLACK_TO_DRAW,
                       min_sep_ratio=SLACK_MIN_SEP,
                       color=SLACK_COLOR,
                       annotate=SHOW_SLACK_LABELS)

# Legend
if SHOW_LEGENDS:
    boundary_proxy = Line2D([0], [0], color="#111111", lw=2.0, label="Hyperplane f(x)=0")
    margin_proxy   = Line2D([0], [0], color="#111111", lw=1.6, ls="--", label="Margin (f(x)=±1)")
    slack_proxy    = Line2D([0], [0], color=SLACK_COLOR, lw=2.0, label=r"Slack vectors $\xi_i$")
    handles = [sc0, sc1, boundary_proxy, margin_proxy, slack_proxy]
    if np.any(sv_on_7):
        handles.append(Line2D([], [], marker="o", lw=0, markerfacecolor='none',
                              markeredgecolor="#111111", markersize=9, label="SV on margin"))
    if np.any(sv_in_7):
        handles.append(Line2D([], [], marker="o", lw=0, markerfacecolor='none',
                              markeredgecolor="#ef4444", markersize=9, label="Vectors inside margin"))
    leg7 = ax7.legend(handles=handles, loc=LEGEND_LOC, framealpha=LEGEND_FRAME_ALPHA)
    tune_legend_alpha(leg7)

set_limits(ax7, X_qs, pad=PAD_DEFAULT)
plt.tight_layout(); maybe_save(fig7, "svm_7_soft_margin_slack.png"); plt.show()

# ================================================================================
# KERNEL TRICK — Circles + polynomial (separate figures)
# 8) 2D points only  -> kernel_circles_points_2D.png
# 9) 3D lifted points (z = x1² + x2²) + linear hyperplane f=0
#    -> kernel_circles_points_3D_with_plane.png
# ================================================================================
X_circ, y_circ = make_circles(n_samples=300, factor=0.45, noise=0.06, random_state=1)

# --- Figure 8: 2D points only ---
fig2d, ax2d = plt.subplots(figsize=(6.8, 6.1))
ax2d.scatter(X_circ[y_circ == 0, 0], X_circ[y_circ == 0, 1], s=30, c=palette["neg"],
             edgecolor="white", linewidth=0.6, label="Class 0")
ax2d.scatter(X_circ[y_circ == 1, 0], X_circ[y_circ == 1, 1], s=30, c=palette["pos"],
             edgecolor="white", linewidth=0.6, label="Class 1")
ax2d.set_xlabel("x₁"); ax2d.set_ylabel("x₂"); ax2d.set_aspect("equal", adjustable="box")
if SHOW_TITLES:
    ax2d.set_title("Circles — 2D view (points only)")
if SHOW_LEGENDS:
    leg2d = ax2d.legend(loc=LEGEND_LOC, framealpha=LEGEND_FRAME_ALPHA)
    tune_legend_alpha(leg2d)
plt.tight_layout(); maybe_save(fig2d, "kernel_circles_points_2D.png"); plt.show()

# --- Explicit polynomial lifting: z = r^2 = x1^2 + x2^2 ---
r2 = np.sum(X_circ**2, axis=1, keepdims=True)
X_lift = np.c_[X_circ, r2]  # (x1, x2, r^2)

# Linear separator in 3D (hyperplane f=0 only)
C_lin = 50.0
clf_lin = SVC(kernel="linear", C=C_lin).fit(X_lift, y_circ)
w3d, b3d = clf_lin.coef_[0], clf_lin.intercept_[0]  # w = (w1, w2, w3)

# Grid for the hyperplane: w1*x + w2*y + w3*z + b = 0  ->  z = -(b + w1*x + w2*y)/w3
x1 = np.linspace(X_circ[:, 0].min()-0.2, X_circ[:, 0].max()+0.2, 90)
x2 = np.linspace(X_circ[:, 1].min()-0.2, X_circ[:, 1].max()+0.2, 90)
X1m, X2m = np.meshgrid(x1, x2)
Z_plane = None if abs(w3d[2]) < 1e-12 else -(b3d + w3d[0]*X1m + w3d[1]*X2m) / w3d[2]

# --- Figure 9: 3D lifted points + linear hyperplane f=0 ---
fig3d = plt.figure(figsize=(9.8, 8.2))
ax3d = fig3d.add_subplot(111, projection="3d")
ax3d.view_init(elev=24, azim=-60)

# Lifted points
ax3d.scatter(X_circ[y_circ == 0, 0], X_circ[y_circ == 0, 1], r2[y_circ == 0, 0], s=26, c=palette["neg"],
             edgecolor="white", linewidth=0.6, label="Class 0")
ax3d.scatter(X_circ[y_circ == 1, 0], X_circ[y_circ == 1, 1], r2[y_circ == 1, 0], s=26, c=palette["pos"],
             edgecolor="white", linewidth=0.6, label="Class 1")

# Plane f=0 (no margins, no "bell")
if Z_plane is not None:
    ax3d.plot_surface(X1m, X2m, Z_plane, alpha=0.42, edgecolor="none")

ax3d.set_xlabel("x₁"); ax3d.set_ylabel("x₂"); ax3d.set_zlabel("z = x₁² + x₂²")
if SHOW_TITLES:
    ax3d.set_title("Circles — 3D (lifted points + linear hyperplane)")
if SHOW_LEGENDS:
    plane_proxy = Line2D([0], [0], color="#111111", lw=2, label="Hyperplane f=0")
    leg3d = ax3d.legend(handles=[plane_proxy], loc=LEGEND_LOC, framealpha=LEGEND_FRAME_ALPHA)
    tune_legend_alpha(leg3d)

plt.tight_layout(); maybe_save(fig3d, "kernel_circles_points_3D_with_plane.png"); plt.show()

# ================================================================================
# SVR — ε-insensitive tube + slacks ξ / ξ*
# ================================================================================
rng_svr = np.random.RandomState(7)
n = 70
X1d = np.linspace(-2.0, 2.0, n)[:, None]
y_true = 0.5*np.sin(3*X1d[:, 0]) + 0.3*X1d[:, 0]
y = y_true + rng_svr.normal(scale=0.15, size=n)

# SVR model (RBF)
eps = 0.12
C = 8.0
gamma = 2.0
svr = SVR(kernel="rbf", C=C, epsilon=eps, gamma=gamma).fit(X1d, y)

# Grid & train predictions
xx = np.linspace(X1d.min()-0.1, X1d.max()+0.1, 600)[:, None]
f = svr.predict(xx)
tube_top = f + eps
tube_bot = f - eps
f_tr = svr.predict(X1d)              # f(x_i) for slacks
res  = y - f_tr                      # residual
idx_up   = np.where(res >  eps + 1e-12)[0]   # above tube ⇒ ξ
idx_down = np.where(res < -eps - 1e-12)[0]   # below tube ⇒ ξ*

# Pick a few largest slacks (readability)
def pick_top(idxs, vals, k):
    if len(idxs) == 0:
        return []
    order = np.argsort(-vals)
    idxs = idxs[order][:k]
    vals = vals[order][:k]
    return list(zip(idxs, vals))

slack_up   = res[idx_up] - eps
slack_down = -eps - res[idx_down]
sel_up     = pick_top(idx_up,   slack_up,   2)
sel_down   = pick_top(idx_down, slack_down, 2)

selected = []
if len(sel_up)   > 0: selected.append(("up",)   + sel_up[0])  # (typ, i, s)
if len(sel_down) > 0: selected.append(("down",) + sel_down[0])
rest = [("up",)+t for t in sel_up[1:]] + [("down",)+t for t in sel_down[1:]]
rest.sort(key=lambda t: -t[2])
for t in rest:
    if len(selected) >= 3:
        break
    selected.append(t)

# --- SVR figure ---
fig, ax = plt.subplots(figsize=(9.6, 6.0))

# Scatter
ax.scatter(X1d[:, 0], y, s=28, c=palette["neg"], edgecolor="white", linewidth=0.7, label="Data")

# Predicted curve f(x)
line_pred, = ax.plot(xx[:, 0], f, lw=2.2, c="#111111", label=r"Prediction $f(x)$")

# ε-insensitive tube (±ε shading)
ax.fill_between(xx[:, 0], tube_bot, tube_top, alpha=0.20, color="#8b5cf6")
tube_proxy = Rectangle((0, 0), 1, 1, fc="#8b5cf6", alpha=0.20, ec="none")

# Support vectors (points touching/exceeding the tube)
sv = svr.support_
ax.scatter(X1d[sv, 0], y[sv], s=110, facecolors="none", edgecolors="#111111",
           linewidths=1.8, label="Support vectors")

# (Optional) Ground-truth signal
ax.plot(X1d[:, 0], y_true, lw=1.4, ls="--", c="#6B7280", label="Underlying signal", alpha=0.9)

# ε annotation: double vertical arrow centered at f(x0)
x0 = 0.6
y0 = svr.predict([[x0]])[0]
arrow = FancyArrowPatch((x0, y0 - eps), (x0, y0 + eps),
                        arrowstyle="<->", mutation_scale=12,
                        lw=1.6, color="#8b5cf6")
ax.add_patch(arrow)
ax.text(x0 + 0.04, y0, r"$\varepsilon$", color="#8b5cf6",
        fontsize=13, va="center", ha="left",
        bbox=dict(boxstyle="round,pad=0.15", fc="white", ec="none", alpha=0.85))

# Slacks ξ / ξ* (2–3 examples)
COLOR_UP = "salmon"    # ξ (above)
COLOR_DN = "#10b981"   # ξ* (below)
for j, (typ, i, s) in enumerate(selected):
    xi = X1d[i, 0]; yi = y[i]; fi = f_tr[i]
    if typ == "up":
        y_tube = fi + eps
        col = COLOR_UP
        label = r"$\xi$"
    else:
        y_tube = fi - eps
        col = COLOR_DN
        label = r"$\xi^\ast$"
    ax.add_patch(FancyArrowPatch((xi, y_tube), (xi, yi),
                                 arrowstyle="->", mutation_scale=11,
                                 lw=1.8, color=col))
    y_mid = 0.5*(y_tube + yi)
    x_off = 0.06 * (1 if j % 2 == 0 else -1)
    ax.text(xi + x_off, y_mid, label,
            fontsize=12, color=col, va="center", ha="center",
            bbox=dict(boxstyle="round,pad=0.15", fc="white", ec="none", alpha=0.85))

ax.set_xlabel("x"); ax.set_ylabel("y")
if SHOW_TITLES:
    ax.set_title(f"SVR — ε-insensitive tube (RBF, C={C}, ε={eps}, γ={gamma})")
if SHOW_LEGENDS:
    leg = ax.legend(handles=[Line2D([], [], c="#111111", lw=2, label=r"Prediction $f(x)$"),
                             tube_proxy,
                             Line2D([], [], marker="o", lw=0, c=palette["neg"],
                                    markerfacecolor=palette["neg"], markersize=6, label="Data"),
                             Line2D([], [], marker="o", lw=0, c="#111111",
                                    markerfacecolor="none", markeredgecolor="#111111",
                                    markersize=9, label="Support vectors"),
                             Line2D([], [], c="#6B7280", lw=1.4, ls="--", label="Underlying signal")],
                    loc=LEGEND_LOC)
    tune_legend_alpha(leg)

plt.tight_layout(); maybe_save(fig, "svr_epsilon_tube_demo.png"); plt.show()

# ================================
# 7bis) Compare margin parameter (small C vs large C)
#      -> Two files: svm_soft_margin_C_small.png / svm_soft_margin_C_large.png
# ================================
def soft_margin_compare_C(X, y, C_vals=(0.25, 80.0),
                          names=("svm_soft_margin_C_small.png", "svm_soft_margin_C_large.png"),
                          pad=PAD_DEFAULT):
    for C_val, fname in zip(C_vals, names):
        clf = SVC(kernel="linear", C=C_val).fit(X, y)
        fig, ax = plt.subplots()
        plot_decision_regions(ax, clf, X, pad=pad, fill=True, boundary=True, margins=True)
        sc0, sc1 = plot_points(ax, X, y, with_legend=False)

        # SVs: black = on margin ; red = inside margin
        sv_on, sv_in = sv_on_in_masks(clf, X, y, C_val)
        if np.any(sv_on):
            ax.scatter(X[sv_on, 0], X[sv_on, 1], s=110, facecolors='none',
                       edgecolors="#111111", linewidths=1.8, label="SV on margin")
        if np.any(sv_in):
            ax.scatter(X[sv_in, 0], X[sv_in, 1], s=110, facecolors='none',
                       edgecolors="#ef4444", linewidths=1.8, label="Vectors inside margin")

        if SHOW_LEGENDS:
            boundary_proxy = Line2D([0], [0], color="#111111", lw=2.0, label="Hyperplane f(x)=0")
            margin_proxy   = Line2D([0], [0], color="#111111", lw=1.6, ls="--", label="Margin (f(x)=±1)")
            handles = [sc0, sc1, boundary_proxy, margin_proxy]
            if np.any(sv_on):
                handles.append(Line2D([], [], marker="o", lw=0, markerfacecolor='none',
                                      markeredgecolor="#111111", markersize=9, label="SV on margin"))
            if np.any(sv_in):
                handles.append(Line2D([], [], marker="o", lw=0, markerfacecolor='none',
                                      markeredgecolor="#ef4444", markersize=9, label="Vectors inside margin"))
            leg = ax.legend(handles=handles, loc=LEGEND_LOC, framealpha=LEGEND_FRAME_ALPHA)
            tune_legend_alpha(leg)

        set_limits(ax, X, pad=pad)
        plt.tight_layout(); maybe_save(fig, fname); plt.show()

# Call (same quasi-separable cloud as Fig. 7)
soft_margin_compare_C(X_qs, y_qs,
                      C_vals=(0.010, 10000.0),
                      names=("svm_soft_margin_C_small.png", "svm_soft_margin_C_large.png"))

# ================================
# 7quater) RBF – varying γ (fixed C)
#      -> svm_rbf_gamma_small.png / svm_rbf_gamma_large.png
# ================================
def rbf_compare_gamma(X, y,
                      gammas=(0.2, 5.0),
                      C_fixed=8.0,
                      names=("svm_rbf_gamma_small.png", "svm_rbf_gamma_large.png"),
                      pad=PAD_DEFAULT):
    for gamma, fname in zip(gammas, names):
        clf = SVC(kernel="rbf", C=C_fixed, gamma=gamma).fit(X, y)

        fig, ax = plt.subplots()
        plot_decision_regions(ax, clf, X, pad=pad, fill=True, boundary=True, margins=True)
        sc0, sc1 = plot_points(ax, X, y, with_legend=False)

        # SVs: black = on margin ; red = inside margin
        sv_on, sv_in = sv_on_in_masks(clf, X, y, C_fixed)
        if np.any(sv_on):
            ax.scatter(X[sv_on, 0], X[sv_on, 1], s=110, facecolors='none',
                       edgecolors="#111111", linewidths=1.8, label="SV on margin")
        if np.any(sv_in):
            ax.scatter(X[sv_in, 0], X[sv_in, 1], s=110, facecolors='none',
                       edgecolors="#ef4444", linewidths=1.8, label="Vectors inside margin")

        if SHOW_LEGENDS:
            boundary_proxy = Line2D([0], [0], color="#111111", lw=2.0, label="Hyperplane f(x)=0")
            margin_proxy   = Line2D([0], [0], color="#111111", lw=1.6, ls="--", label="Margin (f(x)=±1)")
            handles = [sc0, sc1, boundary_proxy, margin_proxy]
            if np.any(sv_on):
                handles.append(Line2D([], [], marker="o", lw=0, markerfacecolor='none',
                                      markeredgecolor="#111111", markersize=9, label="SV on margin"))
            if np.any(sv_in):
                handles.append(Line2D([], [], marker="o", lw=0, markerfacecolor='none',
                                      markeredgecolor="#ef4444", markersize=9, label="Vectors inside margin"))
            leg = ax.legend(handles=handles, loc=LEGEND_LOC, framealpha=LEGEND_FRAME_ALPHA)
            tune_legend_alpha(leg)

        set_limits(ax, X, pad=pad)
        plt.tight_layout(); maybe_save(fig, fname); plt.show()

# Example call (same 'moons' as the previous comparison)
X_mo, y_mo = make_moons(n_samples=180, noise=0.18, random_state=42)
rbf_compare_gamma(X_mo, y_mo,
                  gammas=(0.2, 5.0),
                  C_fixed=8.0,
                  names=("svm_rbf_gamma_small.png", "svm_rbf_gamma_large.png"))
