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
    df = pd.read_csv(csv_path)
    meta, labels, counts, n_samples = parse_special_paired_format(df)

    fig = plt.figure(figsize=(13, 5))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_axis_off()

    title = "Results Overview"
    subtitle = datetime.now().strftime("%Y-%m-%d %H:%M")

    left_x = 0.07
    y_top = 0.88
    line_sp = 0.045

    fig.text(left_x, y_top, title, ha="left", va="top", fontsize=20, fontweight="bold")

    y = y_top - line_sp*1.5
    fig.text(left_x, y, "Question", ha="left", va="top", fontsize=12, fontweight="bold")
    fig.text(left_x, y - line_sp*0.9, wrap(meta.get("question", "N/A"), 60, 8), ha="left", va="top", fontsize=11)


    y -= line_sp*3.0
    fig.text(left_x, y, "Model", ha="left", va="top", fontsize=12, fontweight="bold")
    fig.text(left_x, y - line_sp*0.9, wrap(meta.get("model", "N/A"), 60, 2), ha="left", va="top", fontsize=11)

    y -= line_sp*2.2
    fig.text(left_x, y, f"Samples: {int(np.nansum(counts)) if isinstance(counts, np.ndarray) else 0}", ha="left", va="top", fontsize=12, fontweight="bold")

    def autopct_with_counts(pct):
        total = counts.sum() if isinstance(counts, np.ndarray) else float(np.sum(counts))
        count = int(round(pct/100.0 * total))
        # return f"{pct:.1f}%\n({count})"
        return f"{count}"

    if len(labels) and np.sum(counts) > 0:
        try:
            ax.pie(
                counts,
                labels=labels,
                autopct=autopct_with_counts,
                startangle=90,
                radius=0.6,
                center=(0.73, 0.5)
            )
        except TypeError:
            ax.pie(
                counts,
                labels=labels,
                autopct=autopct_with_counts,
                startangle=90,
                radius=0.6
            )
    else:
        fig.text(0.65, 0.5, "No tally data found in CSV.", ha="center", va="center", fontsize=12, fontweight="bold")

    Path(output).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=200, bbox_inches="tight")
    plt.close(fig)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Create an infographic-style pie chart from a tally CSV.")
    parser.add_argument("csv", help="Path to the CSV file.")
    parser.add_argument("-o", "--output", default="pie_infographic.png", help="Output image path (PNG).")
    args = parser.parse_args()
    make_chart(args.csv, args.output)
