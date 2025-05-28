# ────────── utils_plot.py ──────────
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Optional, Iterable
import pandas as pd
from typing import List
import math

METRICS        = [
    # "accuracy",
    "ece",
    "adaece",
    # "classece",
    # "brier"
    ]
CALIBS         = ["pre",
                  "TS",
                  "Platt",
                  "IR",
                #   "BBQ"
                  ]
CALIBS_DICT    = {
    "pre":"pre",
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
    datasets : list — какие датасеты рисовать (None -> все)
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
        return df.loc[:, ~df.columns.duplicated()].copy()

    # ---- 0. вспом. ф-ция, распознаём регуляризацию --------------------
    def _detect_reg(row):
        if row.get("ls", 0)     > 0:   return "Label Smoothing"
        if row.get("mixup", 0)  > 0:   return "MixUp"
        if bool(row.get("augmix", False)): return "AugMix"
        return ""

    CAL_MAP = {
        "pre":        "pre",
        "temperature":"TS",
        "platt":      "Platt",
        "isotonic":   "IR",
        "bbq":        "BBQ",
    }

    # ------------------------------------------------------------
    # 1) базовые метки
    base = _dedup(test_df)
    base["calibrator"]  = "pre"            # уже правильная метка
    base["method_name"] = base["loss"].str.replace("_", " ").str.title()

    # 2) калиброванные модели
    cal = _dedup(calib_df).copy()
    cal["calibrator"]  = cal["calibrator"].str.lower().map(CAL_MAP)   # !!! NEW
    cal["method_name"] = cal["loss"].str.replace("_", " ").str.title()

    # ---- 1a. ДОБАВЛЯЕМ RegAug-строки  【★】 ---------------------------
    reg_base = base[base.apply(_detect_reg, axis=1) != ""].copy()
    reg_base["method_name"] = reg_base.apply(_detect_reg, axis=1)
    base = pd.concat([base, reg_base], ignore_index=True)

    # ---- 2a. ДОБАВЛЯЕМ RegAug-строки к калиброванным  【★】 -----------
    reg_cal = cal[cal.apply(_detect_reg, axis=1) != ""].copy()
    reg_cal["method_name"] = reg_cal.apply(_detect_reg, axis=1)
    cal = pd.concat([cal, reg_cal], ignore_index=True)

    # ---- 3. объединяем, удаляем дубли --------------------------------
    all_cols = ["dataset", "method_name", "calibrator"] + METRICS
    full = (
        pd.concat([base[all_cols], cal[all_cols]], ignore_index=True)
        .drop_duplicates(subset=["dataset", "method_name", "calibrator"])
    )

    # 4) усредняем
    full = (
        full
        .groupby(["dataset", "method_name", "calibrator"], as_index=False)[METRICS]
        .mean()
    )

    # 5) pivot — теперь CALIBS берём из той же карты
    CALIBS = ["pre", "TS", "Platt", "IR"]

    summary = (
        full
        .set_index(["dataset", "method_name", "calibrator"])
        .unstack("calibrator")
        .reindex(columns=CALIBS, level=1)      # имена гарантированно существуют
        .sort_index(axis=0, level=[0, 1])
    )
    if save_latex:
        summary_str = summary.copy().round(3).astype(str)
        # для каждой метрики (уровень 0 в MultiIndex columns)
        for met in METRICS:
            cols = summary_str.columns.get_level_values(0) == met
            # slice DataFrame по этим столбцам
            sub = summary_str.loc[:, cols].astype(float)
            # найдем минимальные в каждой строке
            mins = sub.min(axis=1)
            # обернём в \textbf
            for idx in sub.index:
                for cal in sub.columns:
                    val = float(sub.loc[idx, cal])
                    if math.isclose(val, mins.loc[idx]):
                        summary_str.loc[idx, cal] = rf"\textbf{{{val:.3f}}}"
                        
        latex_body = summary_str.to_latex(
        buf=None,
        multirow=True,
        column_format="l l " + " ".join(["r" * len(CALIBS) for _ in METRICS]),
        escape=False,            # чтобы \textbf{} не экранировалось
        na_rep="--"
    )


        # --- 6c. Обёртка в table ------------------------------
        final_tex = rf"""
        \begin{{table}}[htbp]
        \centering
        \scriptsize
        \setlength{{\tabcolsep}}{{4pt}}
        \caption{{Calibration quality before and after post-hoc methods.}}
        \label{{tab:calibration_summary}}
        {latex_body}
        \end{{table}}
        """
        out_path.write_text(final_tex, encoding="utf8")
        print(f"LaTeX table saved to {out_path}")

    return summary
