#!/usr/bin/env python3
import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import textwrap
from datetime import datetime
from pathlib import Path

def parse_special_paired_format(df: pd.DataFrame):
    cols = list(df.columns)
    meta = {"question": "N/A", "system_prompt": "N/A", "model": "N/A"}
    labels = []
    counts = np.array([])
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
    n_samples = int(np.nansum(counts).round()) if counts.size > 0 else 0
    return meta, labels, counts, n_samples

def wrap(s, width=60, max_lines=6):
    if not isinstance(s, str):
        s = str(s)
    lines = textwrap.wrap(s, width=width)
    if len(lines) > max_lines:
        lines = lines[:max_lines] + ["…"]
    return "\n".join(lines)

def make_chart(csv_path: str, output: str):
    if output is None:
        base, _ = Path(csv_path).with_suffix("").name, Path(csv_path).suffix
        output = f"{base}_infographic.png"
    df = pd.read_csv(csv_path)
    meta, labels, counts, n_samples = parse_special_paired_format(df)

    if counts.size:
        # Remove zero-count labels
        positive_mask = counts > 0
        labels = [label for label, keep in zip(labels, positive_mask) if keep]
        counts = counts[positive_mask]

    fig = plt.figure(figsize=(13, 5))
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

    title_style = dict(ha="left", va="top", fontsize=20, fontweight="bold", color="#1f2a44")
    label_style = dict(ha="left", va="top", fontsize=12, fontweight="bold", color="#1f2a44")
    body_style = dict(
        ha="left",
        va="top",
        fontsize=11,
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
    fig.text(left_x, y - line_sp * 0.9, wrap(meta.get("question", "N/A"), 60, 8), **body_style)

    y -= line_sp * 3.0
    fig.text(left_x, y, "Model", **label_style)
    fig.text(left_x, y - line_sp * 0.9, wrap(meta.get("model", "N/A"), 60, 2), **body_style)

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

    def autopct_with_counts(pct):
        total = counts.sum() if isinstance(counts, np.ndarray) else float(np.sum(counts))
        count = int(round(pct / 100.0 * total))
        return "" if count == 0 else f"{count}"

    colors = plt.get_cmap("Pastel2")(np.linspace(0, 1, len(labels))) if len(labels) else None

    if len(labels) and np.sum(counts) > 0:
        pie_ax = fig.add_axes([0.45, 0.08, 0.5, 0.84], zorder=2)
        pie_ax.set_facecolor("none")
        pie_ax.axis("equal")
        pie_ax.set_axis_off()

        pie_kwargs = dict(
            labels=labels,
            autopct=autopct_with_counts,
            startangle=90,
            colors=colors,
            textprops=dict(color="#1f2a44", fontweight="bold", fontsize=11),
            wedgeprops=dict(width=0.5, linewidth=1.5, edgecolor="#f8fafc"),
            pctdistance=0.72,
        )
        try:
            wedges, texts, autotexts = pie_ax.pie(counts, radius=1, **pie_kwargs)
        except TypeError:
            fallback_kwargs = {
                k: v for k, v in pie_kwargs.items() if k not in {"wedgeprops", "pctdistance", "textprops"}
            }
            wedges, texts, autotexts = pie_ax.pie(counts, radius=1, **fallback_kwargs)

        for text in texts:
            text.set_color("#1f2a44")
            text.set_fontsize(11)
        for autotext in autotexts:
            autotext.set_color("#1f2a44")
            autotext.set_fontsize(10)

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
