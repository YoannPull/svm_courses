# SVM Teaching Figures 

Small, self-contained scripts to generate classroom-ready figures for **SVM** (linear / soft-margin / kernel) and **SVR** (ε-insensitive tube).
All images are saved to `figures/` with consistent styling and clear legends.

## What’s included

* **Linear classification:** separability, margins `f(x)=±1`, support vectors, perpendicular segment **M ≈ 1/‖w‖**, Perceptron (non-uniqueness).
* **Soft margin:** slack vectors (ξ_i), **free SVs** ((0<λ<C)) vs **saturated SVs** ((λ≈C)).
* **Kernel trick (circles):** 2D points + explicit 3D polynomial lift ((x_1, x_2, r^2)) with a linear hyperplane.
* **SVR (RBF):** prediction (f(x)), ε-tube, SV highlights, slack arrows (ξ, ξ^*).

---

## Install

### Using `requirements.txt`

```bash
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### Using `pyproject.toml`

```bash
python -m venv .venv
source .venv/bin/activate
pip install .
```

> Python 3.10+ recommended (works on 3.9–3.12).
> Main deps: `numpy`, `matplotlib`, `scikit-learn`.

---

## Run

Generate **all** figures:

```bash
python svm_figures.py
```

Defaults:

* `SAVE_FIGS=True` → files saved in `figures/`
* `PLOT_TRUE=True` → figures also shown on screen

**Headless / save-only mode:**

```python
PLOT_TRUE = False  # don’t show, only save PNGs
```

This disables `plt.show()` and closes figures after saving.

---

## Key settings (top of script)

```python
SAVE_FIGS   = True
FIG_DIR     = "figures"
FIG_DPI     = 300
PLOT_TRUE   = True          # set False to save-only

SHOW_TITLES  = False
SHOW_LEGENDS = True
LEGEND_LOC   = "upper left"

# Soft-margin slack arrows
SHOW_SLACK_LABELS = True
MAX_SLACK_ARROWS  = 4
MIN_SLACK_TO_DRAW = 0.12
SLACK_MIN_SEP     = 0.18
SLACK_COLOR       = "#8b5cf6"
```

---

## Output (what each figure shows)

| #  | Topic                   | File                                      |
| -- | ----------------------- | ----------------------------------------- |
| 1  | Separable (points)      | `svm_1_separable_points.png`              |
| 2  | Non-separable (moons)   | `svm_2_nonseparable_points.png`           |
| 3  | Quasi-separable         | `svm_3_quasi_separable_points.png`        |
| 4  | Linear SVM (regions)    | `svm_4_separable_hyperplan_fond.png`      |
| 5  | Perceptron (many HPs)   | `svm_5_perceptron_multi_hyperplans.png`   |
| 6  | Margins, SVs, segment M | `svm_6_marges_sv_M.png`                   |
| 7  | Soft margin + slacks    | `svm_7_soft_margin_slack.png`             |
| 8  | Circles 2D              | `kernel_circles_points_2D.png`            |
| 9  | Circles 3D (lift)       | `kernel_circles_points_3D_with_plane.png` |
| 10 | SVR (RBF), ε-tube       | `svr_epsilon_tube_demo.png`               |

---

## Reproducibility

Random seeds are fixed; decision grids use a consistent resolution and zoom.

## Troubleshooting

* No windows? Check `PLOT_TRUE`.
* No files? Ensure `SAVE_FIGS=True` and write access to `figures/`.
* Slow/OOM? Lower grid density (e.g., `make_grid(..., n=300)`) and use `PLOT_TRUE=False`.

## License

Add your license (e.g., MIT).
