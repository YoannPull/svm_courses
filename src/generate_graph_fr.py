# -*- coding: utf-8 -*-
"""
SVM – Figures pédagogiques (version compacte)
- Fonds colorés sur hyperplans
- Segment M perpendiculaire à l'hyperplan (longueur M = 1/||w||)
- Soft margin + vecteurs de slack ξ_i (avec sélection lisible, SV correctement entourés)
- "Kernel trick" (cercles) : vue 2D des points + vue 3D (lift explicite r^2) avec hyperplan linéaire
- SVR : tube ε-insensible + slacks ξ / ξ*

Catalogue des figures (fichiers générés dans FIG_DIR) :
1) svm_1_separable_points.png
   - Nuage de points séparables (seulement les points, zoom rapproché)
2) svm_2_nonseparable_points.png
   - Nuage de points non séparables (moons), sans frontière
3) svm_3_quasi_separable_points.png
   - Quasi-séparable : 2 points croisés empêchent l’hyperplan parfait
4) svm_4_separable_hyperplan_fond.png
   - SVM linéaire (hard-margin approx) : hyperplan + fonds colorés des régions
5) svm_5_perceptron_multi_hyperplans.png
   - Perceptron : exemples d’hyperplans possibles (non unicité)
6) svm_6_marges_sv_M.png
   - Marges f(x)=±1, vecteurs de support entourés, segment M perpendiculaire à f(x)=0
7) svm_7_soft_margin_slack.png
   - Soft margin (C modéré) : hyperplan + marges + flèches de slacks ξ_i
8) kernel_circles_points_2D.png
   - Données "cercles" en 2D (points uniquement)
9) kernel_circles_points_3D_with_plane.png
   - Lifting polynomial z=r² et hyperplan f=0 en 3D (pas de "cloche")
10) svr_epsilon_tube_demo.png
    - SVR (RBF) : prédiction f(x), tube ε-insensible ±ε, SV, slacks ξ / ξ*
"""

# =========================
# Imports (dédupliqués)
# =========================
import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import FancyArrowPatch, Rectangle
from sklearn.datasets import make_blobs, make_moons, make_circles
from sklearn.svm import SVC, SVR
from sklearn.linear_model import Perceptron
# (Matplotlib >= 3 gère projection="3d" sans import explicite d'Axes3D)

# =========================
# Options globales
# =========================
SAVE_FIGS     = True           # Enregistrer les figures dans FIG_DIR
FIG_DIR       = "figures_fr"   # Dossier de sortie pour les images
FIG_DPI       = 300            # DPI des fichiers enregistrés
SHOW_TITLES   = False          # Afficher les titres des figures
SHOW_LEGENDS  = True           # Afficher les légendes
LEGEND_LOC    = "upper left"   # Position des légendes (partout)
PLOT_TRUE     = False          # False = "je veux juste sauvegarder" (pas d'affichage)

# Opacité/esthétique des légendes
LEGEND_FRAME_ALPHA = 0.55      # fond/cadre de la légende
LEGEND_ITEM_ALPHA  = 0.85      # marqueurs/lignes dans la légende
LEGEND_TEXT_ALPHA  = 1.00      # texte de la légende

# Options spécifiques "Slack" (Figure 7)
SHOW_SLACK_LABELS  = True      # afficher les libellés ξ_i sur les flèches
MAX_SLACK_ARROWS   = 4         # nb max de vecteurs de slack affichés (réduit)
MIN_SLACK_TO_DRAW  = 0.12      # seuil sur ξ_i pour tracer (plus exigeant)
SLACK_MIN_SEP      = 0.18      # distance mini relative entre flèches sélectionnées
SLACK_COLOR        = "#8b5cf6" # violet doux pour les flèches de slack

# Zooms (axes)
PAD_SMALL   = 0.10   # pour figs 1→4 (zoom serré)
PAD_DEFAULT = 0.22

# Sortie
if SAVE_FIGS:
    os.makedirs(FIG_DIR, exist_ok=True)

def maybe_save(fig, filename):
    """Enregistre la figure si SAVE_FIGS=True."""
    if SAVE_FIGS:
        fig.savefig(os.path.join(FIG_DIR, filename), dpi=FIG_DPI, bbox_inches="tight")
    if not PLOT_TRUE:
        # On ferme pour libérer la mémoire (et éviter toute fenêtre)
        plt.close(fig)
        plt.show = lambda *args, **kwargs: None

# =========================
# Style global Matplotlib
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

# Couleurs + RNG
rng = np.random.RandomState(42)
palette = {"neg": "#1f77b4", "pos": "#ff7f0e"}  # Bleu / Orange

# =========================
# Helpers (réutilisés partout)
# =========================
def tune_legend_alpha(leg,
                      alpha_items=LEGEND_ITEM_ALPHA,
                      alpha_text=LEGEND_TEXT_ALPHA,
                      alpha_frame=LEGEND_FRAME_ALPHA):
    """Affinage homogène de l’opacité des éléments de légende."""
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
    """Cadre des axes avec marge 'pad' autour des données."""
    ax.set_xlim(X[:, 0].min()-pad, X[:, 0].max()+pad)
    ax.set_ylim(X[:, 1].min()-pad, X[:, 1].max()+pad)

def make_grid(X, pad=0.28, n=400):
    """Maillage régulier 2D autour du nuage de points X (±pad)."""
    x_min, x_max = X[:, 0].min()-pad, X[:, 0].max()+pad
    y_min, y_max = X[:, 1].min()-pad, X[:, 1].max()+pad
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, n),
                         np.linspace(y_min, y_max, n))
    return xx, yy, (x_min, x_max, y_min, y_max)

def plot_points(ax, X, y, title="", with_legend=True):
    """
    Dispersion 2 classes (0/1) avec couleurs fixes.
    Graphique : nuage de points ; utile pour comparer avant/après séparation.
    """
    sc0 = ax.scatter(X[y == 0, 0], X[y == 0, 1], s=36, c=palette["neg"],
                     edgecolor="white", linewidth=0.7, label="Classe 0")
    sc1 = ax.scatter(X[y == 1, 0], X[y == 1, 1], s=36, c=palette["pos"],
                     edgecolor="white", linewidth=0.7, label="Classe 1")
    ax.set_xlabel("x₁"); ax.set_ylabel("x₂")
    if SHOW_TITLES and title:
        ax.set_title(title)
    if with_legend and SHOW_LEGENDS:
        leg = ax.legend(loc=LEGEND_LOC, framealpha=LEGEND_FRAME_ALPHA)
        tune_legend_alpha(leg)
    return sc0, sc1

def plot_decision_regions(ax, clf, X, pad=0.18, fill=False, boundary=True, margins=False):
    """
    Affiche : (i) fonds colorés (classes prédites) ; (ii) frontière f(x)=0 ; (iii) marges f(x)=±1
    - pad : zoom du maillage autour des données
    - fill=True : remplit les régions de décision
    - boundary=True : trace f(x)=0
    - margins=True : trace f(x)=±1
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
    Trace un petit segment perpendiculaire à l'hyperplan f(x)=0, centré dans la zone des données,
    et l'annote 'M' pour illustrer la distance marge ~ 1/||w||.
    Graphique : segment court montrant la direction perpendiculaire à l’hyperplan.
    """
    w = np.asarray(w)
    norm_w = np.linalg.norm(w)
    if norm_w < 1e-12:
        return
    # point proche du centre de masse des axes, posé sur f(x)=0
    x_mid = 0.5 * (X[:, 0].min() + X[:, 0].max())
    y_mid = 0.5 * (X[:, 1].min() + X[:, 1].max())
    if abs(w[1]) > abs(w[0]):
        x0 = x_mid
        y0 = -(w[0]/w[1]) * x0 - b / w[1]
    else:
        y0 = y_mid
        x0 = -(w[1] * y0 + b) / w[0]
    p0 = np.array([x0, y0])            # sur f(x)=0
    delta = w / (norm_w ** 2)          # longueur ~ 1/||w||
    p1 = p0 + delta
    ax.plot([p0[0], p1[0]], [p0[1], p1[1]], lw=2.0, color=color)
    # Annotation 'M' décalée tangentiel
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
    """Calcule ξ_i = max(0, 1 - y_i f(x_i)) avec y_i ∈ {-1, +1} (à partir de y ∈ {0,1})."""
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
    Pour quelques points violant la marge (ξ_i > 0), trace la flèche jusqu'à la marge f(x)=y_i.
    Sélection gloutonne : on évite les flèches trop proches (lisibilité).
    Graphique : flèches vers la marge, annotées ξ_i.
    """
    w = np.asarray(w)
    ww = np.dot(w, w)
    if ww <= 1e-14:
        return [], []

    xi, f, y_pm1 = compute_slack(X, y01, w, b)
    cand = np.where(xi > min_xi)[0]
    if cand.size == 0:
        return [], []

    # Distance minimale relative (en % de l'étendue max)
    scale = max(np.ptp(X[:, 0]), np.ptp(X[:, 1]))
    min_sep = max(1e-8, float(min_sep_ratio) * float(scale))

    # Tri par ξ décroissant, puis sélection espacée
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
        delta_f = (y_pm1[i] - f[i])           # cible - actuel en espace f
        delta_x = (delta_f / ww) * w          # vecteur en 2D
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

# ---------- NOUVEAU : masques SV sur/dans la marge ----------
def sv_on_in_masks(clf, X, y01, C, tol_alpha=None, tol_margin=1e-3):
    """
    Renvoie deux masques booléens (len(X)):
      - sv_on : SV sur la marge (y f(x) ≈ 1)
      - sv_in : Vecteurs dans la marge (y f(x) < 1)
    Détection des SV via α_i>0 avec une tolérance robuste (relative au max(α)).
    """
    y = 2*y01 - 1
    f = clf.decision_function(X)  # f(x)
    alpha = np.zeros(len(X))
    if getattr(clf, "support_", None) is not None:
        alpha[clf.support_] = np.abs(clf.dual_coef_).ravel()

    # Tolérance robuste: relative à l'échelle observée des alpha (et non à C)
    if tol_alpha is None:
        maxa = float(np.max(alpha)) if alpha.size else 0.0
        tol_alpha = max(1e-12, 1e-6 * max(1.0, maxa))

    sv = alpha > tol_alpha
    sv_on = sv & (np.abs(y * f - 1.0) <= tol_margin)  # sur la marge
    sv_in = sv & (y * f < 1.0 - tol_margin)           # à l’intérieur de la marge (ou mal classé)
    return sv_on, sv_in

# ================================================================================
# 1) Séparable — points uniquement (zoom rapproché)
#    -> Graphique : nuage de points (2 classes). Fichier : svm_1_separable_points.png
# ================================================================================
X_sep, y_sep = make_blobs(n_samples=46,
                          centers=[(-1.5, -1.5), (1.5, 1.5)],
                          cluster_std=[0.50, 0.50],
                          random_state=42)
fig1, ax1 = plt.subplots()
plot_points(ax1, X_sep, y_sep, title="Séparable – points uniquement")
set_limits(ax1, X_sep, pad=PAD_SMALL)
plt.tight_layout(); maybe_save(fig1, "svm_1_separable_points.png"); plt.show()

# ================================================================================
# 2) Non séparables — points uniquement (zoom rapproché)
#    -> Graphique : nuage de points (moons). Fichier : svm_2_nonseparable_points.png
# ================================================================================
X_nsep, y_nsep = make_moons(n_samples=70, noise=0.22, random_state=42)
fig2, ax2 = plt.subplots()
plot_points(ax2, X_nsep, y_nsep, title="Non séparables – points uniquement (pas de frontière)")
set_limits(ax2, X_nsep, pad=PAD_SMALL)
plt.tight_layout(); maybe_save(fig2, "svm_2_nonseparable_points.png"); plt.show()

# ================================================================================
# 3) Quasi-séparable — 2 points croisés (zoom rapproché)
#    -> Graphique : nuage de points où 2 points "croisés" détruisent la séparabilité parfaite.
#       Fichier : svm_3_quasi_separable_points.png
# ================================================================================
centers_qs = np.array([[-1.35, -1.35], [1.35, 1.35]])
X_qs, y_qs = make_blobs(n_samples=54, centers=centers_qs,
                        cluster_std=[0.50, 0.50], random_state=7)
idx0 = rng.choice(np.where(y_qs == 0)[0], size=2, replace=False)
idx1 = rng.choice(np.where(y_qs == 1)[0], size=2, replace=False)
X_qs[idx0] = centers_qs[1] + rng.normal(scale=0.10, size=(2, 2))
X_qs[idx1] = centers_qs[0] + rng.normal(scale=0.10, size=(2, 2))
fig3, ax3 = plt.subplots()
plot_points(ax3, X_qs, y_qs, title="Quasi-séparable – 2 points croisés (pas d’hyperplan)")
set_limits(ax3, X_qs, pad=PAD_SMALL)
plt.tight_layout(); maybe_save(fig3, "svm_3_quasi_separable_points.png"); plt.show()

# ================================================================================
# 4) Séparable — hyperplan (fonds colorés)
#    -> Graphique : hyperplan f(x)=0 + régions (fonds colorés). Fichier : svm_4_separable_hyperplan_fond.png
# ================================================================================
clf_sep_lin = SVC(kernel="linear", C=1e6).fit(X_sep, y_sep)  # hard-margin approx
fig4, ax4 = plt.subplots()
plot_decision_regions(ax4, clf_sep_lin, X_sep, pad=PAD_SMALL, fill=True, boundary=True, margins=False)
plot_points(ax4, X_sep, y_sep, title="Séparable – hyperplan (SVM linéaire)")
plt.tight_layout(); maybe_save(fig4, "svm_4_separable_hyperplan_fond.png"); plt.show()

# ================================================================================
# 5) Perceptron — hyperplans multiples
#    -> Graphique : plusieurs frontières issues de seeds différentes (non unicité).
#       Fichier : svm_5_perceptron_multi_hyperplans.png
# ================================================================================
fig5, ax5 = plt.subplots()
sc0, sc1 = plot_points(ax5, X_sep, y_sep,
                       title="Perceptron – hyperplans multiples (non unicité)",
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
proxy_line = Line2D([0], [0], color="#111111", lw=1.6, label="Hyperplans possibles (Perceptron)")
if SHOW_LEGENDS:
    leg5 = ax5.legend(handles=[sc0, sc1, proxy_line], loc=LEGEND_LOC,
                      framealpha=LEGEND_FRAME_ALPHA)
    tune_legend_alpha(leg5)
set_limits(ax5, X_sep, pad=PAD_DEFAULT)
plt.tight_layout(); maybe_save(fig5, "svm_5_perceptron_multi_hyperplans.png"); plt.show()

# ================================================================================
# 6) Séparable — marges & vecteurs de support + segment M
#    -> Graphique : f(x)=0, marges f(x)=±1, SV sur/dans la marge (noir/rouge), segment M.
#       Fichier : svm_6_marges_sv_M.png
# ================================================================================
C_hard = 1e6
clf_margin = SVC(kernel="linear", C=C_hard).fit(X_sep, y_sep)
w = clf_margin.coef_[0]; b = clf_margin.intercept_[0]
fig6, ax6 = plt.subplots()
plot_decision_regions(ax6, clf_margin, X_sep, pad=PAD_DEFAULT, fill=True, boundary=True, margins=True)
sc0, sc1 = plot_points(ax6, X_sep, y_sep,
                       title="Séparable – marges, SV et longueur M (1/||w||)",
                       with_legend=False)

# Cerclage SV : noir = sur marge ; rouge = dans marge
sv_on_6, sv_in_6 = sv_on_in_masks(clf_margin, X_sep, y_sep, C_hard)
if np.any(sv_on_6):
    ax6.scatter(X_sep[sv_on_6, 0], X_sep[sv_on_6, 1], s=110, facecolors='none',
                edgecolors="#111111", linewidths=1.8, label="SV sur la marge")
if np.any(sv_in_6):
    ax6.scatter(X_sep[sv_in_6, 0], X_sep[sv_in_6, 1], s=110, facecolors='none',
                edgecolors="#ef4444", linewidths=1.8, label="Vecteurs dans la marge")

# Annotation d'un SV + segment perpendiculaire M
sv_indices = np.where(sv_on_6 | sv_in_6)[0]
if sv_indices.size > 0:
    sv0 = X_sep[sv_indices[0]]
    ax6.annotate("Vecteur de support",
                 xy=(sv0[0], sv0[1]), xytext=(sv0[0]+0.40, sv0[1]+0.40),
                 arrowprops=dict(arrowstyle="->", lw=1.2, color="#111111"),
                 fontsize=11)
draw_margin_perpendicular(ax6, w, b, X_sep, color="blue")

# Légende
if SHOW_LEGENDS:
    boundary_proxy = Line2D([0], [0], color="#111111", lw=2.0, label="Hyperplan f(x)=0")
    margin_proxy   = Line2D([0], [0], color="#111111", lw=1.6, ls="--", label="Marge (f(x)=±1)")
    handles = [sc0, sc1, boundary_proxy, margin_proxy]
    if np.any(sv_on_6):
        handles.append(Line2D([], [], marker="o", lw=0, markerfacecolor='none',
                              markeredgecolor="#111111", markersize=9, label="SV sur la marge"))
    if np.any(sv_in_6):
        handles.append(Line2D([], [], marker="o", lw=0, markerfacecolor='none',
                              markeredgecolor="#ef4444", markersize=9, label="Vecteurs dans la marge"))
    leg6 = ax6.legend(handles=handles, loc=LEGEND_LOC, framealpha=LEGEND_FRAME_ALPHA)
    tune_legend_alpha(leg6)

set_limits(ax6, X_sep, pad=PAD_DEFAULT)
plt.tight_layout(); maybe_save(fig6, "svm_6_marges_sv_M.png"); plt.show()

# ================================================================================
# 7) Soft margin (SVM linéaire, C modéré) + vecteurs de slack ξ_i
#    -> Graphique : f(x)=0, marges ±1, cerclage SV sur/dans la marge + flèches ξ_i
#       Fichier : svm_7_soft_margin_slack.png
# ================================================================================
C_soft = 1.0
clf_soft = SVC(kernel="linear", C=C_soft).fit(X_qs, y_qs)
w_soft = clf_soft.coef_[0]; b_soft = clf_soft.intercept_[0]

fig7, ax7 = plt.subplots()
plot_decision_regions(ax7, clf_soft, X_qs, pad=PAD_DEFAULT, fill=True, boundary=True, margins=True)
sc0, sc1 = plot_points(ax7, X_qs, y_qs,
                       title=f"Soft margin – SVM linéaire (C={C_soft}) et vecteurs de slack $\\xi_i$",
                       with_legend=False)

# Cerclage : noir = sur marge ; rouge = dans marge
sv_on_7, sv_in_7 = sv_on_in_masks(clf_soft, X_qs, y_qs, C_soft)
if np.any(sv_on_7):
    ax7.scatter(X_qs[sv_on_7, 0], X_qs[sv_on_7, 1], s=110, facecolors='none',
                edgecolors="#111111", linewidths=1.8, label="SV sur la marge")
if np.any(sv_in_7):
    ax7.scatter(X_qs[sv_in_7, 0], X_qs[sv_in_7, 1], s=110, facecolors='none',
                edgecolors="#ef4444", linewidths=1.8, label="Vecteurs dans la marge")

# Vecteurs de slack (flèches)
_ = draw_slack_vectors(ax7, X_qs, y_qs, w_soft, b_soft,
                       max_k=MAX_SLACK_ARROWS,
                       min_xi=MIN_SLACK_TO_DRAW,
                       min_sep_ratio=SLACK_MIN_SEP,
                       color=SLACK_COLOR,
                       annotate=SHOW_SLACK_LABELS)

# Légende dédiée
if SHOW_LEGENDS:
    boundary_proxy = Line2D([0], [0], color="#111111", lw=2.0, label="Hyperplan f(x)=0")
    margin_proxy   = Line2D([0], [0], color="#111111", lw=1.6, ls="--", label="Marge (f(x)=±1)")
    slack_proxy    = Line2D([0], [0], color=SLACK_COLOR, lw=2.0, label=r"Vecteurs de slack $\xi_i$")
    handles = [sc0, sc1, boundary_proxy, margin_proxy, slack_proxy]
    if np.any(sv_on_7):
        handles.append(Line2D([], [], marker="o", lw=0, markerfacecolor='none',
                              markeredgecolor="#111111", markersize=9, label="SV sur la marge"))
    if np.any(sv_in_7):
        handles.append(Line2D([], [], marker="o", lw=0, markerfacecolor='none',
                              markeredgecolor="#ef4444", markersize=9, label="Vecteurs dans la marge"))
    leg7 = ax7.legend(handles=handles, loc=LEGEND_LOC, framealpha=LEGEND_FRAME_ALPHA)
    tune_legend_alpha(leg7)

set_limits(ax7, X_qs, pad=PAD_DEFAULT)
plt.tight_layout(); maybe_save(fig7, "svm_7_soft_margin_slack.png"); plt.show()

# ================================================================================
# KERNEL TRICK — Cercles + polynomial (figures séparées)
# 8) 2D points seuls  -> kernel_circles_points_2D.png
# 9) 3D points liftés (z = x1² + x2²) + hyperplan linéaire f=0
#    -> kernel_circles_points_3D_with_plane.png
# ================================================================================
X_circ, y_circ = make_circles(n_samples=300, factor=0.45, noise=0.06, random_state=1)

# --- Figure 8 : 2D points uniquement ---
fig2d, ax2d = plt.subplots(figsize=(6.8, 6.1))
ax2d.scatter(X_circ[y_circ == 0, 0], X_circ[y_circ == 0, 1], s=30, c=palette["neg"],
             edgecolor="white", linewidth=0.6, label="Classe 0")
ax2d.scatter(X_circ[y_circ == 1, 0], X_circ[y_circ == 1, 1], s=30, c=palette["pos"],
             edgecolor="white", linewidth=0.6, label="Classe 1")
ax2d.set_xlabel("x₁"); ax2d.set_ylabel("x₂"); ax2d.set_aspect("equal", adjustable="box")
if SHOW_TITLES:
    ax2d.set_title("Cercles — vue 2D (points uniquement)")
if SHOW_LEGENDS:
    leg2d = ax2d.legend(loc=LEGEND_LOC, framealpha=LEGEND_FRAME_ALPHA)
    tune_legend_alpha(leg2d)
plt.tight_layout(); maybe_save(fig2d, "kernel_circles_points_2D.png"); plt.show()

# --- Lifting polynomial explicite : z = r^2 = x1^2 + x2^2 ---
r2 = np.sum(X_circ**2, axis=1, keepdims=True)
X_lift = np.c_[X_circ, r2]  # (x1, x2, r^2)

# Séparateur linéaire en 3D (uniquement l'hyperplan f=0)
C_lin = 50.0
clf_lin = SVC(kernel="linear", C=C_lin).fit(X_lift, y_circ)
w3d, b3d = clf_lin.coef_[0], clf_lin.intercept_[0]  # w = (w1, w2, w3)

# Grille pour l'hyperplan : w1*x + w2*y + w3*z + b = 0  ->  z = -(b + w1*x + w2*y)/w3
x1 = np.linspace(X_circ[:, 0].min()-0.2, X_circ[:, 0].max()+0.2, 90)
x2 = np.linspace(X_circ[:, 1].min()-0.2, X_circ[:, 1].max()+0.2, 90)
X1m, X2m = np.meshgrid(x1, x2)
Z_plane = None if abs(w3d[2]) < 1e-12 else -(b3d + w3d[0]*X1m + w3d[1]*X2m) / w3d[2]

# --- Figure 9 : 3D points liftés + hyperplan linéaire f=0 ---
fig3d = plt.figure(figsize=(9.8, 8.2))
ax3d = fig3d.add_subplot(111, projection="3d")
ax3d.view_init(elev=24, azim=-60)

# Points liftés
ax3d.scatter(X_circ[y_circ == 0, 0], X_circ[y_circ == 0, 1], r2[y_circ == 0, 0], s=26, c=palette["neg"],
             edgecolor="white", linewidth=0.6, label="Classe 0")
ax3d.scatter(X_circ[y_circ == 1, 0], X_circ[y_circ == 1, 1], r2[y_circ == 1, 0], s=26, c=palette["pos"],
             edgecolor="white", linewidth=0.6, label="Classe 1")

# Plan f=0 (pas de marges ni "cloche")
if Z_plane is not None:
    ax3d.plot_surface(X1m, X2m, Z_plane, alpha=0.42, edgecolor="none")

ax3d.set_xlabel("x₁"); ax3d.set_ylabel("x₂"); ax3d.set_zlabel("z = x₁² + x₂²")
if SHOW_TITLES:
    ax3d.set_title("Cercles — 3D (points liftés + hyperplan linéaire)")
if SHOW_LEGENDS:
    plane_proxy = Line2D([0], [0], color="#111111", lw=2, label="Hyperplan f=0")
    leg3d = ax3d.legend(handles=[plane_proxy], loc=LEGEND_LOC, framealpha=LEGEND_FRAME_ALPHA)
    tune_legend_alpha(leg3d)

plt.tight_layout(); maybe_save(fig3d, "kernel_circles_points_3D_with_plane.png"); plt.show()

# ================================================================================
# SVR — Tube ε-insensible + slacks ξ / ξ*
#    -> Graphique : nuage de données 1D, prédiction f(x),
#       tube ±ε ombré, SV entourés, flèches des slacks (2–3 exemples)
#       Fichier : svr_epsilon_tube_demo.png
# ================================================================================
rng_svr = np.random.RandomState(7)
n = 70
X1d = np.linspace(-2.0, 2.0, n)[:, None]
y_true = 0.5*np.sin(3*X1d[:, 0]) + 0.3*X1d[:, 0]
y = y_true + rng_svr.normal(scale=0.15, size=n)

# Modèle SVR (RBF)
eps = 0.12
C = 8.0
gamma = 2.0
svr = SVR(kernel="rbf", C=C, epsilon=eps, gamma=gamma).fit(X1d, y)

# Prédictions grille & train
xx = np.linspace(X1d.min()-0.1, X1d.max()+0.1, 600)[:, None]
f = svr.predict(xx)
tube_top = f + eps
tube_bot = f - eps
f_tr = svr.predict(X1d)              # f(x_i) pour les slacks
res  = y - f_tr                      # résidu
idx_up   = np.where(res >  eps + 1e-12)[0]   # au-dessus du tube ⇒ ξ
idx_down = np.where(res < -eps - 1e-12)[0]   # au-dessous du tube ⇒ ξ*

# Sélectionne quelques plus gros slacks (lisibilité)
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

# --- Figure SVR ---
fig, ax = plt.subplots(figsize=(9.6, 6.0))

# Nuage
ax.scatter(X1d[:, 0], y, s=28, c=palette["neg"], edgecolor="white", linewidth=0.7, label="Données")

# Courbe prédite f(x)
line_pred, = ax.plot(xx[:, 0], f, lw=2.2, c="#111111", label=r"Prédiction $f(x)$")

# Tube ε-insensible (ombrage ±ε)
ax.fill_between(xx[:, 0], tube_bot, tube_top, alpha=0.20, color="#8b5cf6")
tube_proxy = Rectangle((0, 0), 1, 1, fc="#8b5cf6", alpha=0.20, ec="none")

# Vecteurs de support (points qui touchent/dépassent le tube)
sv = svr.support_
ax.scatter(X1d[sv, 0], y[sv], s=110, facecolors="none", edgecolors="#111111",
           linewidths=1.8, label="Vecteurs de support")

# (Optionnel) Signal générateur
ax.plot(X1d[:, 0], y_true, lw=1.4, ls="--", c="#6B7280", label="Signal sous-jacent", alpha=0.9)

# Annotation de ε : double flèche verticale centrée sur f(x0)
x0 = 0.6
y0 = svr.predict([[x0]])[0]
arrow = FancyArrowPatch((x0, y0 - eps), (x0, y0 + eps),
                        arrowstyle="<->", mutation_scale=12,
                        lw=1.6, color="#8b5cf6")
ax.add_patch(arrow)
ax.text(x0 + 0.04, y0, r"$\varepsilon$", color="#8b5cf6",
        fontsize=13, va="center", ha="left",
        bbox=dict(boxstyle="round,pad=0.15", fc="white", ec="none", alpha=0.85))

# Slacks ξ / ξ* (2–3 exemples)
COLOR_UP = "salmon"    # ξ (au-dessus)
COLOR_DN = "#10b981"   # ξ* (au-dessous)
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
    # flèche verticale du bord du tube -> point
    ax.add_patch(FancyArrowPatch((xi, y_tube), (xi, yi),
                                 arrowstyle="->", mutation_scale=11,
                                 lw=1.8, color=col))
    # label au milieu, léger décalage horizontal alterné
    y_mid = 0.5*(y_tube + yi)
    x_off = 0.06 * (1 if j % 2 == 0 else -1)
    ax.text(xi + x_off, y_mid, label,
            fontsize=12, color=col, va="center", ha="center",
            bbox=dict(boxstyle="round,pad=0.15", fc="white", ec="none", alpha=0.85))

# Axes / titres / légende
ax.set_xlabel("x"); ax.set_ylabel("y")
if SHOW_TITLES:
    ax.set_title(f"SVR — tube ε-insensible (RBF, C={C}, ε={eps}, γ={gamma})")
if SHOW_LEGENDS:
    leg = ax.legend(handles=[Line2D([], [], c="#111111", lw=2, label=r"Prédiction $f(x)$"),
                             tube_proxy,
                             Line2D([], [], marker="o", lw=0, c=palette["neg"],
                                    markerfacecolor=palette["neg"], markersize=6, label="Données"),
                             Line2D([], [], marker="o", lw=0, c="#111111",
                                    markerfacecolor="none", markeredgecolor="#111111",
                                    markersize=9, label="Vecteurs de support"),
                             Line2D([], [], c="#6B7280", lw=1.4, ls="--", label="Signal sous-jacent")],
                    loc=LEGEND_LOC)
    tune_legend_alpha(leg)

plt.tight_layout(); maybe_save(fig, "svr_epsilon_tube_demo.png"); plt.show()


# ================================
# 7bis) Comparaison du paramètre de marge (C petit vs grand)
#      -> Deux fichiers pour LaTeX : svm_soft_margin_C_small.png / svm_soft_margin_C_large.png
# ================================
def soft_margin_compare_C(X, y, C_vals=(0.25, 80.0),
                          names=("svm_soft_margin_C_small.png", "svm_soft_margin_C_large.png"),
                          pad=PAD_DEFAULT):
    for C_val, fname in zip(C_vals, names):
        clf = SVC(kernel="linear", C=C_val).fit(X, y)
        fig, ax = plt.subplots()
        plot_decision_regions(ax, clf, X, pad=pad, fill=True, boundary=True, margins=True)
        sc0, sc1 = plot_points(ax, X, y, with_legend=False)

        # Vecteurs de support : noir = sur marge ; rouge = dans marge
        sv_on, sv_in = sv_on_in_masks(clf, X, y, C_val)
        if np.any(sv_on):
            ax.scatter(X[sv_on, 0], X[sv_on, 1], s=110, facecolors='none',
                       edgecolors="#111111", linewidths=1.8, label="SV sur la marge")
        if np.any(sv_in):
            ax.scatter(X[sv_in, 0], X[sv_in, 1], s=110, facecolors='none',
                       edgecolors="#ef4444", linewidths=1.8, label="Vecteurs dans la marge")

        if SHOW_LEGENDS:
            boundary_proxy = Line2D([0], [0], color="#111111", lw=2.0, label="Hyperplan f(x)=0")
            margin_proxy   = Line2D([0], [0], color="#111111", lw=1.6, ls="--", label="Marge (f(x)=±1)")
            handles = [sc0, sc1, boundary_proxy, margin_proxy]
            if np.any(sv_on):
                handles.append(Line2D([], [], marker="o", lw=0, markerfacecolor='none',
                                      markeredgecolor="#111111", markersize=9, label="SV sur la marge"))
            if np.any(sv_in):
                handles.append(Line2D([], [], marker="o", lw=0, markerfacecolor='none',
                                      markeredgecolor="#ef4444", markersize=9, label="Vecteurs dans la marge"))
            leg = ax.legend(handles=handles, loc=LEGEND_LOC, framealpha=LEGEND_FRAME_ALPHA)
            tune_legend_alpha(leg)

        set_limits(ax, X, pad=pad)
        plt.tight_layout(); maybe_save(fig, fname); plt.show()

# Appel (même nuage quasi-séparable que la Fig. 7)
soft_margin_compare_C(X_qs, y_qs,
                      C_vals=(0.010, 10000.0),
                      names=("svm_soft_margin_C_small.png", "svm_soft_margin_C_large.png"))


# ================================
# 7quater) RBF – variation de γ (C fixé)
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

        # Vecteurs de support : noir = sur marge ; rouge = dans marge
        sv_on, sv_in = sv_on_in_masks(clf, X, y, C_fixed)
        if np.any(sv_on):
            ax.scatter(X[sv_on, 0], X[sv_on, 1], s=110, facecolors='none',
                       edgecolors="#111111", linewidths=1.8, label="SV sur la marge")
        if np.any(sv_in):
            ax.scatter(X[sv_in, 0], X[sv_in, 1], s=110, facecolors='none',
                       edgecolors="#ef4444", linewidths=1.8, label="Vecteurs dans la marge")

        if SHOW_LEGENDS:
            boundary_proxy = Line2D([0], [0], color="#111111", lw=2.0, label="Hyperplan f(x)=0")
            margin_proxy   = Line2D([0], [0], color="#111111", lw=1.6, ls="--", label="Marge (f(x)=±1)")
            handles = [sc0, sc1, boundary_proxy, margin_proxy]
            if np.any(sv_on):
                handles.append(Line2D([], [], marker="o", lw=0, markerfacecolor='none',
                                      markeredgecolor="#111111", markersize=9, label="SV sur la marge"))
            if np.any(sv_in):
                handles.append(Line2D([], [], marker="o", lw=0, markerfacecolor='none',
                                      markeredgecolor="#ef4444", markersize=9, label="Vecteurs dans la marge"))
            leg = ax.legend(handles=handles, loc=LEGEND_LOC, framealpha=LEGEND_FRAME_ALPHA)
            tune_legend_alpha(leg)

        set_limits(ax, X, pad=pad)
        plt.tight_layout(); maybe_save(fig, fname); plt.show()

# Exemple d’appel (mêmes 'moons' que la comparaison précédente)
X_mo, y_mo = make_moons(n_samples=180, noise=0.18, random_state=42)
rbf_compare_gamma(X_mo, y_mo,
                  gammas=(0.2, 5.0),
                  C_fixed=8.0,
                  names=("svm_rbf_gamma_small.png", "svm_rbf_gamma_large.png"))
