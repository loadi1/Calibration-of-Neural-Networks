# ────────── utils_plot.py ──────────
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Optional, Iterable
import pandas as pd

def plot_metric_ds(
    df: pd.DataFrame,
    metric: str,
    datasets: Optional[Iterable[str]] = None,
    hue: str = "method",
    smooth_window: int = 1,
    max_epoch: Optional[int] = None,
    show_legend: bool = True,
    save_pdf: bool = True,
    path_to_dir: Path = Path("plots")
):
    """
    Рисует кривые <metric> по эпохам отдельно для каждого датасета.

    Parameters
    ----------
    df : DataFrame  — hist_df с колонками epoch, metric, dataset, <hue>
    metric : str    — имя столбца для оси Y (например 'val_acc')
    datasets : list — какие датасеты рисовать (None → все)
    hue : str       — чем раскрашивать кривые (обычно 'method')
    smooth_window : int  — ширина скользящего среднего
    max_epoch : int — обрезать график
    save_pdf : bool — сохранять ли PDF+SVG
    """
    if datasets is None:
        datasets = df["dataset"].unique()

    palette = sns.color_palette("tab10", n_colors=df[hue].nunique())

    for ds in datasets:
        sub_ds = df[df["dataset"] == ds]
        if sub_ds.empty:
            continue

        plt.figure(figsize=(6, 4))
        for name, run in sub_ds.groupby("config"):
            x = run["epoch"]
            y = run[metric]
            if smooth_window > 1:
                y = y.rolling(smooth_window, center=True,
                              min_periods=1).mean()
            if max_epoch is not None:
                m = x <= max_epoch
                x, y = x[m], y[m]
            col = palette[sub_ds[hue].cat.codes.loc[run.index[0]]]
            plt.plot(x, y, color=col, alpha=.85, linewidth=1)

        plt.title(f"{ds}: {metric} vs. epochs")
        plt.xlabel("Epoch")
        plt.ylabel(metric)
        if show_legend:
            handles, labels = [], []
            for m, col in zip(sub_ds[hue].cat.categories, palette):
                handles.append(plt.Line2D([], [], color=col, lw=3))
                labels.append(m)
            plt.legend(handles, labels, title=hue, bbox_to_anchor=(1.04, 1),
                       loc="upper left", fontsize="small")
        plt.grid(alpha=.3)
        plt.tight_layout()

        if save_pdf:
            path_to_dir.mkdir(exist_ok=True, parents=True)
            base = f"{ds}_{metric}"
            plt.savefig(path_to_dir / f"{base}.pdf",  bbox_inches="tight")
        plt.show()