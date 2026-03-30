import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from scipy.stats import linregress

SCRIPT_DIR = Path(__file__).parent
CSV_DIR = SCRIPT_DIR / "../benchmarks/csv"
OUT_DIR = SCRIPT_DIR / "../benchmarks/pdf"
OUT_DIR.mkdir(parents=True, exist_ok=True)

mpl.rcParams.update(
    {
        "text.usetex": True,
        "font.family": "serif",
        "font.serif": ["Computer Modern Roman"],
        "axes.labelsize": 10,
        "axes.titlesize": 10,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "legend.fontsize": 8,
        "legend.title_fontsize": 8,
        "figure.dpi": 150,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.05,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.linewidth": 0.4,
        "grid.alpha": 0.45,
        "lines.linewidth": 1.6,
        "lines.markersize": 5,
    }
)

C_FM = "#1a1a1a"
C_BEST = "#1f77b4"
C_WORST = "#d62728"
C_MEAN = "#2ca02c"

STYLE_FM = dict(color=C_FM, lw=2.2, ls="-", marker="D", ms=5, zorder=6)
STYLE_BEST = dict(color=C_BEST, lw=1.8, ls="-", marker="o", ms=4.5, zorder=5)
STYLE_WORST = dict(color=C_WORST, lw=1.8, ls="--", marker="s", ms=4.5, zorder=5)
STYLE_MEAN = dict(color=C_MEAN, lw=1.8, ls="-.", marker="^", ms=4.5, zorder=5)


def load_csv(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    df.columns = df.columns.str.strip()

    def method_key(row):
        if row["method_type"] == "FM":
            return "FM"
        return f"{row['stencil']}/{row['minimization']}/{row['interpolant']}"

    df["method"] = df.apply(method_key, axis=1)
    return df.sort_values("h_mean")


def latest_csv() -> Path:
    candidates = sorted(CSV_DIR.glob("sphere_convergence_*.csv"))
    if not candidates:
        sys.exit(
            f"No sphere_convergence_*.csv found in {CSV_DIR}.\n"
            "Run bench_sphere_convergence first."
        )
    return candidates[-1]


def log_regression(xs: np.ndarray, ys: np.ndarray):
    mask = (xs > 0) & (ys > 0) & np.isfinite(xs) & np.isfinite(ys)
    if mask.sum() < 2:  # Minimum 2 points pour une droite
        return float("nan"), float("nan"), float("nan")
    lx, ly = np.log(xs[mask]), np.log(ys[mask])
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        slope, intercept, r, *_ = linregress(lx, ly)
    return slope, intercept, r**2


def reg_label(base: str, slope: float, r2: float) -> str:
    if np.isfinite(slope) and np.isfinite(r2):
        return rf"{base}\ \ $\alpha\!=\!{slope:.2f},\,R^2\!=\!{r2:.3f}$"
    return base


def make_summary(df: pd.DataFrame, xkey: str, ykey: str):
    jm = df[df["method"] != "FM"].copy()

    method_curves = {
        m: grp.groupby(xkey)[ykey].mean().reset_index().sort_values(xkey)
        for m, grp in jm.groupby("method")
    }

    extreme_x = jm[xkey].max() if xkey == "n_vertices" else jm[xkey].min()
    at_extreme = {}
    for m, curve in method_curves.items():
        sub = curve[curve[xkey] == extreme_x][ykey]
        if not sub.empty:
            at_extreme[m] = sub.mean()
    if not at_extreme:
        at_extreme = {m: curve[ykey].mean() for m, curve in method_curves.items()}

    best_method = min(at_extreme, key=at_extreme.get)
    worst_method = max(at_extreme, key=at_extreme.get)

    mean_curve = jm.groupby(xkey)[ykey].mean().reset_index().sort_values(xkey)
    fm_curve = (
        df[df["method"] == "FM"]
        .groupby(xkey)[ykey]
        .mean()
        .reset_index()
        .sort_values(xkey)
    )

    return {
        "fm": (fm_curve, "FM"),
        "best": (method_curves[best_method], best_method),
        "worst": (method_curves[worst_method], worst_method),
        "mean": (mean_curve, "mean JM"),
    }


def short_latex(method: str) -> str:
    if method == "FM":
        return r"\textsc{fm}"
    if method == "mean JM":
        return r"moyenne \textsc{jm}"
    parts = method.split("/")
    if len(parts) != 3:
        return method
    s, m, i = parts
    smap = {
        "Mesh": r"\textsc{mesh}",
        "Ell1(0.5)": r"$\ell^1_{0.5}$",
        "Ell1(2.0)": r"$\ell^1_{2.0}$",
        "MeshEll1(0.5)": r"$\mathcal{M}{\oplus}\ell^1_{0.5}$",
        "MeshEll1(2.0)": r"$\mathcal{M}{\oplus}\ell^1_{2.0}$",
    }
    mmap = {
        "FermatIntegral": "Fermat",
        "EikonalEquation": "Eikonal",
        "CellBasedMarching": "Cell",
        "QuadraticCurve": "Quad",
    }
    imap = {"Cubic": r"\textit{cub}", "Graph": r"\textit{grph}"}
    return rf"{smap.get(s,s)}/{mmap.get(m,m)}/{imap.get(i,i)}"


def outside_legend(ax, handles, labels, title=None):
    return ax.legend(
        handles,
        labels,
        title=title,
        loc="upper left",
        bbox_to_anchor=(1.01, 1.0),
        borderaxespad=0,
        frameon=True,
        framealpha=0.95,
        edgecolor="#cccccc",
        fontsize=7.5,
        title_fontsize=8,
    )


def add_slopes(ax, anchor_x_log, anchor_y_log, orders=(1, 2, 3)):
    if not (np.isfinite(anchor_x_log) and np.isfinite(anchor_y_log)):
        return
    colors = {1: "#aaaaaa", 2: "#666666", 3: "#222222"}
    offsets = {1: 0.55, 2: 0.15, 3: -0.30}
    dx = 0.28
    for k in orders:
        # Conversion depuis l'espace log vers l'espace data pour ax.plot sur axes log-log
        x0 = 10**anchor_x_log
        y0 = 10 ** (anchor_y_log + offsets[k])
        x1 = 10 ** (anchor_x_log + dx)
        y1 = 10 ** (anchor_y_log + offsets[k] + k * dx)

        col = colors[k]
        ax.plot(
            [x0, x1, x1, x0], [y0, y0, y1, y0], color=col, lw=0.9, ls="--", zorder=1
        )
        ax.text(
            x1 * 1.05, (y0 * y1) ** 0.5, rf"$h^{k}$", fontsize=7, color=col, va="center"
        )


def plot_convergence(df, ykey, ylabel, title, filename, out):
    summary = make_summary(df, "h_mean", ykey)

    fig, ax = plt.subplots(figsize=(7.5, 4.0))

    entries = [
        ("fm", r"\textsc{fm}", STYLE_FM),
        ("best", r"\textsc{jm} meilleur", STYLE_BEST),
        ("worst", r"\textsc{jm} pire", STYLE_WORST),
        ("mean", r"\textsc{jm} moyenne", STYLE_MEAN),
    ]

    handles, labels = [], []
    for key, base_label, style in entries:
        curve, _ = summary[key]
        if curve.empty:
            continue
        xs = curve["h_mean"].values
        ys = curve[ykey].values
        a, _, r2 = log_regression(xs, ys)
        label = reg_label(base_label, a, r2)
        (line,) = ax.loglog(xs, ys, label=label, **style)
        handles.append(line)
        labels.append(label)

    fm_curve = summary["fm"][0]
    if not fm_curve.empty:
        # On s'assure d'avoir un point valide pour l'ancrage
        mask = (fm_curve["h_mean"] > 0) & (fm_curve[ykey] > 0)
        if mask.any():
            lx = np.log10(fm_curve.loc[mask, "h_mean"].iloc[0])
            ly = np.log10(fm_curve.loc[mask, ykey].iloc[0])
            add_slopes(ax, lx, ly)

    ax.set_xlabel(r"Taille caractéristique $h$")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.invert_xaxis()
    ax.xaxis.set_major_formatter(ticker.LogFormatterMathtext())
    ax.yaxis.set_major_formatter(ticker.LogFormatterMathtext())

    outside_legend(ax, handles, labels, title=r"\textit{Méthode} $(\alpha,\,R^2)$")

    fig.savefig(out / filename, bbox_inches="tight")
    plt.close(fig)


def plot_time(df, xkey, xlabel, title, filename, out, invert_x=False, ref_lines=False):
    summary = make_summary(df, xkey, "time_s")

    fig, ax = plt.subplots(figsize=(7.5, 4.0))

    entries = [
        ("fm", r"\textsc{fm}", STYLE_FM),
        ("best", r"\textsc{jm} meilleur", STYLE_BEST),
        ("worst", r"\textsc{jm} pire", STYLE_WORST),
        ("mean", r"\textsc{jm} moyenne", STYLE_MEAN),
    ]

    handles, labels = [], []
    for key, base_label, style in entries:
        curve, _ = summary[key]
        if curve.empty:
            continue
        xs = curve[xkey].values
        ys = curve["time_s"].values
        a, _, r2 = log_regression(xs, ys)
        label = reg_label(base_label, a, r2)
        (line,) = ax.loglog(xs, ys, label=label, **style)
        handles.append(line)
        labels.append(label)

    if ref_lines:
        fm_curve = summary["fm"][0]
        if len(fm_curve) >= 2:
            n0 = float(fm_curve[xkey].iloc[0])
            t0 = float(fm_curve["time_s"].iloc[0])
            ns = np.geomspace(
                float(fm_curve[xkey].min()), float(fm_curve[xkey].max()), 80
            )
            (l1,) = ax.plot(
                ns,
                t0 * ns / n0,
                ls=":",
                lw=1.0,
                color="#999999",
                label=r"$\mathcal{O}(N)$",
            )
            (l2,) = ax.plot(
                ns,
                t0 * (ns / n0) * np.log(ns / n0 + 1),
                ls="-.",
                lw=1.0,
                color="#555555",
                label=r"$\mathcal{O}(N\log N)$",
            )
            handles += [l1, l2]
            labels += [r"$\mathcal{O}(N)$", r"$\mathcal{O}(N\log N)$"]

    if invert_x:
        ax.invert_xaxis()

    ax.set_xlabel(xlabel)
    ax.set_ylabel(r"Temps d'exécution (s)")
    ax.set_title(title)
    ax.xaxis.set_major_formatter(ticker.LogFormatterMathtext())
    ax.yaxis.set_major_formatter(ticker.LogFormatterMathtext())

    outside_legend(ax, handles, labels, title=r"\textit{Méthode} $(\alpha,\,R^2)$")

    fig.savefig(out / filename, bbox_inches="tight")
    plt.close(fig)


def plot_order_heatmap(df: pd.DataFrame, out: Path):
    jm_methods = sorted(m for m in df["method"].unique() if m != "FM")
    all_methods = ["FM"] + jm_methods

    mae_orders, merr_orders = [], []
    for m in all_methods:
        g = df[df["method"] == m]
        xs = g.groupby("h_mean")["mae_vs_gt"].mean().reset_index()
        a_mae, _, _ = log_regression(xs["h_mean"].values, xs["mae_vs_gt"].values)
        xs2 = g.groupby("h_mean")["max_err_vs_gt"].mean().reset_index()
        a_merr, _, _ = log_regression(xs2["h_mean"].values, xs2["max_err_vs_gt"].values)
        mae_orders.append(a_mae)
        merr_orders.append(a_merr)

    mae_arr = np.array(mae_orders)
    merr_arr = np.array(merr_orders)
    n = len(all_methods)
    data = np.vstack([mae_arr, merr_arr])

    fig_w = max(7.0, n * 0.55)
    fig, ax = plt.subplots(figsize=(fig_w, 2.8))

    im = ax.imshow(data, aspect="auto", cmap="RdYlGn", vmin=0, vmax=4)

    ax.set_xticks(range(n))
    ax.set_xticklabels(
        [short_latex(m) for m in all_methods],
        rotation=55,
        ha="right",
        fontsize=6.5,
    )
    ax.set_yticks([0, 1])
    ax.set_yticklabels([r"MAE", r"$\|\cdot\|_\infty$"], fontsize=8)

    for j in range(n):
        for i, v in enumerate([mae_arr[j], merr_arr[j]]):
            txt = f"{v:.2f}" if np.isfinite(v) else "---"
            col = "black" if 0.7 < v < 3.3 else "white"
            ax.text(j, i, txt, ha="center", va="center", fontsize=6, color=col)

    fig.colorbar(
        im, ax=ax, label=r"Ordre de convergence $\alpha$", shrink=0.85, pad=0.02
    )
    ax.set_title(
        r"Ordres de convergence estimés "
        r"($\log\,\mathrm{err} \approx \alpha\,\log h + c$)",
        fontsize=9,
    )
    fig.savefig(out / "sphere_order_heatmap.pdf", bbox_inches="tight")
    plt.close(fig)


def plot_fm_vs_best_jm(df: pd.DataFrame, out: Path):
    jm = df[df["method"] != "FM"]
    if jm.empty:
        return

    finest_h = jm["h_mean"].min()
    at_finest = jm[jm["h_mean"] == finest_h].groupby("method")["mae_vs_gt"].mean()
    if at_finest.empty:
        return
    best_jm = at_finest.idxmin()

    fig, axes = plt.subplots(2, 2, figsize=(12.0, 7.0))
    axes = axes.ravel()

    panels = [
        ("h_mean", "mae_vs_gt", axes[0], r"MAE vs.\ $h$", r"Taille $h$", r"MAE", True),
        (
            "h_mean",
            "max_err_vs_gt",
            axes[1],
            r"$\|e\|_\infty$ vs.\ $h$",
            r"Taille $h$",
            r"$\|e\|_\infty$",
            True,
        ),
        (
            "h_mean",
            "time_s",
            axes[2],
            r"Temps vs.\ $h$",
            r"Taille $h$",
            r"Temps (s)",
            True,
        ),
        (
            "n_vertices",
            "time_s",
            axes[3],
            r"Temps vs.\ $N$",
            r"Sommets $N$",
            r"Temps (s)",
            False,
        ),
    ]

    for xk, yk, ax, title, xlabel, ylabel, inv in panels:
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_title(title, fontsize=9)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)

        for m, g in jm.groupby("method"):
            if m == best_jm:
                continue
            agg = g.groupby(xk)[yk].mean().reset_index().sort_values(xk)
            ax.loglog(agg[xk], agg[yk], color="#bbbbbb", lw=0.55, alpha=0.4, zorder=1)

        # FM
        fm_agg = (
            df[df["method"] == "FM"]
            .groupby(xk)[yk]
            .mean()
            .reset_index()
            .sort_values(xk)
        )
        if not fm_agg.empty:
            xs_fm, ys_fm = fm_agg[xk].values, fm_agg[yk].values
            a_fm, _, r2_fm = log_regression(xs_fm, ys_fm)
            lbl_fm = reg_label(r"\textsc{fm}", a_fm, r2_fm)
            (l_fm,) = ax.loglog(xs_fm, ys_fm, label=lbl_fm, **STYLE_FM)
        else:
            l_fm = None

        # Best JM
        bj_agg = (
            jm[jm["method"] == best_jm]
            .groupby(xk)[yk]
            .mean()
            .reset_index()
            .sort_values(xk)
        )
        xs_bj, ys_bj = bj_agg[xk].values, bj_agg[yk].values
        a_bj, _, r2_bj = log_regression(xs_bj, ys_bj)
        lbl_bj = reg_label(r"\textsc{jm} meilleur", a_bj, r2_bj)
        (l_bj,) = ax.loglog(xs_bj, ys_bj, label=lbl_bj, **STYLE_BEST)

        if inv:
            ax.invert_xaxis()

        ax.xaxis.set_major_formatter(ticker.LogFormatterMathtext())
        ax.yaxis.set_major_formatter(ticker.LogFormatterMathtext())

        h_list = [l for l in [l_fm, l_bj] if l is not None]
        l_list = [
            lbl for lbl, l in zip([lbl_fm, lbl_bj], [l_fm, l_bj]) if l is not None
        ]
        outside_legend(ax, h_list, l_list, title=r"$(\alpha,\,R^2)$")

    fig.suptitle(
        r"\textsc{fm} vs.\ meilleur \textsc{jm} "
        r"--- les autres variantes \textsc{jm} sont en gris",
        fontsize=10,
    )
    fig.savefig(out / "sphere_fm_vs_jm_summary.pdf", bbox_inches="tight")
    plt.close(fig)


def main():
    csv_path = Path(sys.argv[1]) if len(sys.argv) > 1 else latest_csv()

    print(f"\nLoading {csv_path} ...")
    df = load_csv(csv_path)
    print()
    print("Generating figures ...")

    plot_convergence(
        df,
        "mae_vs_gt",
        ylabel=r"Erreur moyenne absolue (MAE)",
        title=r"Convergence en $h$ --- MAE vs.\ distance géodésique exacte",
        filename="sphere_mae_convergence.pdf",
        out=OUT_DIR,
    )
    plot_convergence(
        df,
        "max_err_vs_gt",
        ylabel=r"Erreur maximale $\|e\|_\infty$",
        title=(
            r"Convergence en $h$ --- " r"$\|e\|_\infty$ vs.\ distance géodésique exacte"
        ),
        filename="sphere_maxerr_convergence.pdf",
        out=OUT_DIR,
    )
    plot_time(
        df,
        "n_vertices",
        xlabel=r"Nombre de sommets $N$",
        title=r"Complexité algorithmique --- temps vs.\ $N$",
        filename="sphere_time_complexity.pdf",
        out=OUT_DIR,
        ref_lines=True,
    )
    plot_time(
        df,
        "h_mean",
        xlabel=r"Taille caractéristique $h$",
        title=r"Complexité algorithmique --- temps vs.\ $h$",
        filename="sphere_time_vs_h.pdf",
        out=OUT_DIR,
        invert_x=True,
    )
    plot_order_heatmap(df, OUT_DIR)
    plot_fm_vs_best_jm(df, OUT_DIR)


if __name__ == "__main__":
    main()
