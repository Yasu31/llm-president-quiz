#!/usr/bin/env python3
import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import textwrap
from pathlib import Path
from typing import Optional

TOKEN_FIELDS = ("input_tokens", "output_tokens", "reasoning_tokens")
TOKEN_LABELS = {
    "input_tokens": "Input",
    "output_tokens": "Output",
    "reasoning_tokens": "Reasoning",
}

def parse_special_paired_format(df: pd.DataFrame, token_stats_df: pd.DataFrame):
    cols = list(df.columns)
    meta = {"question": "N/A", "system_prompt": "N/A", "model": "N/A"}
    labels = []
    counts = np.array([])
    token_stats = {}

    if "question" in cols:
        q_idx = cols.index("question")
        if q_idx + 1 < len(cols):
            q_text_col = cols[q_idx + 1]
            meta["question"] = q_text_col
            labels = (
                df["question"].astype(str).str.strip().str.replace(r"[,\s]+$", "", regex=True).tolist()
            )
            counts = df[q_text_col].fillna(0).astype(float).values

    if "instructions" in cols:
        i_idx = cols.index("instructions")
        if i_idx + 1 < len(cols):
            meta["system_prompt"] = cols[i_idx + 1]

    if "model" in cols:
        m_idx = cols.index("model")
        if m_idx + 1 < len(cols):
            meta["model"] = cols[m_idx + 1]

    metric_col = "metric" if "metric" in token_stats_df.columns else token_stats_df.columns[0]
    for _, row in token_stats_df.iterrows():
        metric = str(row.get(metric_col, "")).strip().lower()
        if not metric:
            continue
        metric_stats = {}
        for field in TOKEN_FIELDS:
            val = row.get(field, np.nan)
            if isinstance(val, str):
                val = val.strip()
            if val in ("", None):
                metric_stats[field] = np.nan
                continue
            try:
                metric_stats[field] = float(val)
            except (TypeError, ValueError):
                metric_stats[field] = np.nan
        token_stats[metric] = metric_stats

    n_samples = int(np.nansum(counts).round()) if counts.size > 0 else 0
    return meta, labels, counts, n_samples, token_stats

def wrap(s, width=60, max_lines=6):
    if not isinstance(s, str):
        s = str(s)
    lines = textwrap.wrap(s, width=width)
    if len(lines) > max_lines:
        lines = lines[:max_lines] + ["…"]
    return "\n".join(lines)

def make_chart(csv_path: str, output: str):
    csv_path_obj = Path(csv_path)
    if output is None:
        output = f"{csv_path_obj.with_suffix('').name}_infographic.png"
    token_stats_path = csv_path_obj.with_name(f"{csv_path_obj.stem}_token_stats{csv_path_obj.suffix}")
    print(f"Loading CSV from: {csv_path_obj} and token stats from: {token_stats_path}")
    df = pd.read_csv(csv_path_obj)
    token_stats_df = pd.read_csv(token_stats_path)
    meta, labels, counts, n_samples, token_stats = parse_special_paired_format(df, token_stats_df)

    if counts.size:
        # Remove zero-count labels
        positive_mask = counts > 0
        labels = [label for label, keep in zip(labels, positive_mask) if keep]
        counts = counts[positive_mask]

    fig = plt.figure(figsize=(8, 5))
    fig.patch.set_facecolor("#ecf2f9")

    bg_ax = fig.add_axes([0, 0, 1, 1], zorder=0)
    gradient = np.linspace(0.75, 1, 256)
    bg_ax.imshow(
        np.tile(gradient, (256, 1)),
        cmap=plt.get_cmap("Blues"),
        alpha=0.22,
        aspect="auto",
        extent=[0, 1, 0, 1],
        origin="lower",
    )
    bg_ax.axis("off")

    ax = fig.add_axes([0, 0, 1, 1], zorder=1)
    ax.set_axis_off()

    title_style = dict(ha="left", va="top", fontsize=12, fontweight="bold", color="#1f2a44")
    label_style = dict(ha="left", va="top", fontsize=13, fontweight="bold", color="#1f2a44")
    body_style = dict(
        ha="left",
        va="top",
        fontsize=12,
        color="#334155",
        bbox=dict(boxstyle="round,pad=0.4", facecolor="none", edgecolor="none"),
    )
    highlight_box = dict(boxstyle="round,pad=0.4", facecolor="none", edgecolor="none")
    timestamp_style = dict(
        ha="right",
        va="top",
        fontsize=10,
        color="#475569",
        bbox=dict(boxstyle="round,pad=0.35", facecolor="none", edgecolor="none"),
    )

    title = f"Result of {n_samples} LLM Responses"

    left_x = 0.07
    y_top = 0.88
    line_sp = 0.045

    fig.text(left_x, y_top, title, **title_style)
    fig.text(0.93, 0.93, "https://yasunori.jp", **timestamp_style)

    y = y_top - line_sp * 1.5
    fig.text(left_x, y, "Question", **label_style)
    fig.text(left_x, y - line_sp * 0.9, wrap(meta.get("question", "N/A"), 40, 8), **body_style)

    y -= line_sp * 3.0
    fig.text(left_x, y, "Model", **label_style)
    fig.text(left_x, y - line_sp * 0.9, wrap(meta.get("model", "N/A"), 40, 2), **body_style)

    y -= line_sp * 2.2
    fig.text(
        left_x,
        y,
        f"Samples: {n_samples}",
        ha="left",
        va="top",
        fontsize=12,
        fontweight="bold",
        color="#1f2a44",
        bbox=highlight_box,
    )

    stats_order = ("count", "sum", "mean", "std", "min", "max")
    has_stats = any(token_stats.get(metric) for metric in stats_order)
    if has_stats:
        mean_stats = token_stats.get("mean", {})
        std_stats = token_stats.get("std", {})
        field_labels = []
        means = []
        stds = []
        for field in TOKEN_FIELDS:
            mean_val = mean_stats.get(field)
            if mean_val is None:
                continue
            mean_val = float(mean_val)
            if np.isnan(mean_val):
                continue
            std_val = std_stats.get(field)
            std_val = float(std_val) if std_val is not None else 0.0
            if np.isnan(std_val):
                std_val = 0.0
            field_labels.append(TOKEN_LABELS[field])
            means.append(mean_val)
            stds.append(std_val)
        if field_labels:
            token_ax = fig.add_axes([0.07, 0.08, 0.28, 0.22], zorder=2)
            x = np.arange(len(field_labels))
            pastel = plt.get_cmap("Pastel2")(np.linspace(0.15, 0.85, len(field_labels)))
            bars = token_ax.bar(
                x,
                means,
                yerr=stds,
                capsize=6,
                color=pastel,
                edgecolor="#f8fafc",
                linewidth=1.6,
            )
            token_ax.set_facecolor("#edf2fb")
            token_ax.set_title("Token Stats", loc="left", fontsize=12, fontweight="bold", color="#1f2a44", pad=8)
            token_ax.set_ylabel("Mean tokens", color="#475569", fontweight="bold")
            token_ax.set_xticks(x)
            token_ax.set_xticklabels(field_labels, color="#1f2a44", fontweight="bold")
            token_ax.tick_params(axis="y", colors="#475569")
            token_ax.spines["top"].set_visible(False)
            token_ax.spines["right"].set_visible(False)
            token_ax.spines["left"].set_color("#cbd5f5")
            token_ax.spines["bottom"].set_color("#cbd5f5")
            token_ax.set_axisbelow(True)
            token_ax.grid(axis="y", color="#e0e7ff", linestyle="--", linewidth=0.6, alpha=0.8)
            for idx, bar in enumerate(bars):
                height = bar.get_height()
                token_ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    height + stds[idx] + max(means) * 0.02,
                    f"{height:.1f}",
                    ha="center",
                    va="bottom",
                    fontsize=9,
                    fontweight="bold",
                    color="#1f2a44",
                )

    colors = plt.get_cmap("Pastel2")(np.linspace(0, 1, len(labels))) if len(labels) else None

    if len(labels) and np.sum(counts) > 0:
        pie_ax = fig.add_axes([0.45, 0.3, 0.5, 0.7], zorder=2)
        pie_ax.set_facecolor("none")
        pie_ax.axis("equal")
        pie_ax.set_axis_off()

        pie_kwargs = dict(
            startangle=90,
            colors=colors,
            wedgeprops=dict(width=0.5, linewidth=1.5, edgecolor="#f8fafc"),
        )
        try:
            wedges, _ = pie_ax.pie(counts, radius=1, **pie_kwargs)
        except TypeError:
            fallback_kwargs = {k: v for k, v in pie_kwargs.items() if k not in {"wedgeprops"}}
            wedges, _ = pie_ax.pie(counts, radius=1, **fallback_kwargs)

        legend_labels = []
        for label, count in zip(labels, counts):
            rounded = int(round(count)) if np.isclose(count, round(count)) else round(float(count), 2)
            legend_labels.append(f"{label} ({rounded})")

        legend = pie_ax.legend(
            wedges,
            legend_labels,
            loc="upper left",
            bbox_to_anchor =(0., 0.1),
            frameon=False,
            fontsize=11,
            ncol=1,
        )
        for text in legend.get_texts():
            text.set_color("#1f2a44")

        if pie_kwargs.get("wedgeprops", {}).get("width"):
            inner_radius = 1 - pie_kwargs["wedgeprops"]["width"]
            centre_circle = plt.Circle((0, 0), inner_radius, color="#ecf2f9", zorder=3)
            pie_ax.add_artist(centre_circle)
            pie_ax.text(
                0,
                0.08,
                f"{n_samples}",
                ha="center",
                va="center",
                fontsize=18,
                fontweight="bold",
                color="#1f2a44",
            )
            pie_ax.text(
                0,
                -0.16,
                "total responses",
                ha="center",
                va="center",
                fontsize=10,
                color="#475569",
            )
    else:
        fig.text(
            0.65,
            0.5,
            "No tally data found in CSV.",
            ha="center",
            va="center",
            fontsize=12,
            fontweight="bold",
            color="#1f2a44",
            bbox=highlight_box,
        )

    Path(output).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Infographic saved to: {output}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Create an infographic-style pie chart from a tally CSV.")
    parser.add_argument("csv", help="Path to the CSV file.")
    parser.add_argument("-o", "--output", default=None, help="Output image path (PNG).")
    args = parser.parse_args()
    make_chart(args.csv, args.output)
