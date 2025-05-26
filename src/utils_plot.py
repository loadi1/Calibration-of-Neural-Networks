# ────────── utils_plot.py ──────────
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Optional, Iterable
import pandas as pd
from typing import List

METRICS        = [
    # "accuracy",
    # "ece",
    # "adaece",
    "classece",
    "brier"
    ]
CALIBS         = ["pre",
                  "TS",
                  "Platt",
                  "IR",
                #   "BBQ"
                  ]
CALIBS_DICT    = {
    "temperature":"TS",
    "platt":"Platt",
    "isotonic":"IR",
    # "bbq":"BBQ"
    }
REG_NAMES      = {
    "ls"    : "Label Smoothing",
    "mixup" : "MixUp",
    "augmix": "AugMix"
    }

def _detect_reg(row):
    """Возвращает имя регуляризации, если она включена, иначе ''."""
    if row["ls"] > 0:        return REG_NAMES["ls"]
    if row["mixup"] > 0:     return REG_NAMES["mixup"]
    if row["augmix"]:        return REG_NAMES["augmix"]
    return ""

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



        
def build_summary_table(
        test_df : pd.DataFrame,
        calib_df: pd.DataFrame,
        METRICS: List[str] = METRICS,
        CALIBS_DICT = CALIBS_DICT,
        save_latex: bool = True,
        out_path : Path = Path("table_calibration_summary.tex")
    ) -> pd.DataFrame:
    """Builds Dataset × Method summary with pre / post-hoc results."""

    def _dedup(df: pd.DataFrame) -> pd.DataFrame:
        """Удаляет дублирующиеся названия столбцов (оставляет первое)."""
        return df.loc[:, ~df.columns.duplicated()].copy()
    def bold_best(series, better="max"):
        """Return styler mask that bolds row-wise best."""
        if better == "max":
            best = series == series.max()
        else:
            best = series == series.min()
        return ['\\textbf{' + f'{v:.2f}' + '}' if b else f'{v:.2f}'
                for v, b in zip(series, best)]


    # ---------------- 1. базовые («pre») ----------------
    base = _dedup(test_df)
    base["calibrator"] = "pre"
    base["method_name"] = base["loss"].str.replace("_", " ").str.title()

    # ---------------- 2. регуляризации ------------------
    reg_rows = base[base.apply(_detect_reg, axis=1) != ""].copy()
    reg_rows["method_name"] = reg_rows.apply(_detect_reg, axis=1)
    reg_rows = reg_rows[reg_rows["loss"] == "cross_entropy"]
    reg_rows = _dedup(reg_rows)

    base = pd.concat([base, reg_rows], ignore_index=True)

    # ---------------- 3. калиброванные ------------------
    cal = _dedup(calib_df)
    cal["method_name"] = cal["loss"].str.replace("_", " ").str.title()
    cal["calibrator"] = cal["calibrator"].replace(CALIBS_DICT)

    cal_reg = cal[cal.apply(_detect_reg, axis=1) != ""].copy()
    cal_reg["method_name"] = cal_reg.apply(_detect_reg, axis=1)
    cal_reg = cal_reg[cal_reg["loss"] == "cross_entropy"]
    cal_reg = _dedup(cal_reg)

    cal = pd.concat([cal, cal_reg], ignore_index=True)

    # ---------------- 4. объединяем ---------------------
    all_cols = ["dataset", "method_name", "calibrator"] + METRICS
    full = pd.concat([base[all_cols], cal[all_cols]], ignore_index=True)

    #  проверка дублей ещё раз
    if full.columns.duplicated().any():
        raise ValueError("Still duplicated columns: "
                         f"{full.columns[full.columns.duplicated()].unique()}")

    # ---------------- 5. усредняем ----------------------
    full = (
        full
        .groupby(["dataset", "method_name", "calibrator"], as_index=False)[METRICS]
        .mean()
    )

    # ---------------- 6. сводная таблица ----------------
    summary = (
        full
        .set_index(["dataset", "method_name", "calibrator"])
        .unstack("calibrator")
        .reindex(columns=CALIBS, level=1)
        .sort_index(axis=0, level=[0, 1])
    )
    

    # ---------------- 7. LaTeX --------------------------
    if save_latex:
        sty = (
            summary
            .style
            .format(precision=2, escape="latex")
            # .apply(bold_best, subset=pd.IndexSlice[:, pd.IndexSlice['accuracy', :]], better="max", axis=1)
            # .apply(bold_best, subset=pd.IndexSlice[:, pd.IndexSlice['ece',      :]], better="min", axis=1)
            # .apply(bold_best, subset=pd.IndexSlice[:, pd.IndexSlice['adaece',   :]], better="min", axis=1)
            .apply(bold_best, subset=pd.IndexSlice[:, pd.IndexSlice['classece', :]], better="min", axis=1)
            .apply(bold_best, subset=pd.IndexSlice[:, pd.IndexSlice['brier',    :]], better="min", axis=1)
        )
        
        
        summary.round(2).to_latex(
            out_path,
            multirow=True,
            float_format="%.2f",
            na_rep="--",
            caption="Calibration quality before and after post-hoc methods.",
            label="tab:calibration_summary",
            escape=False,               # allow \multicolumn text
            multicolumn=True, multicolumn_format='c'
        )
        print(f"LaTeX table saved to {out_path}")

    return summary
