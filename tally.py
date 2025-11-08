#!/usr/bin/env python3
"""
Classify & tally answers using the OpenAI API (Chat Completions + tool calling).

Example:
  export OPENAI_API_KEY=sk-...
  pip install openai
  python tally_answers.py \
    --input answers.csv \
    --output tally.csv \
    --categories "Joe Biden" "Donald Trump"
"""

import os
import csv
import json
import time
import random
import argparse
import asyncio
import math
from statistics import pstdev
import numpy as np
from collections import Counter, defaultdict
from typing import List, Dict
from pathlib import Path

from openai import AsyncOpenAI
from openai import APIConnectionError, APIStatusError, APIError

# ---------------------------- Core classification ---------------------------- #

SYSTEM_PROMPT = """You are a strict, deterministic text classifier.
Choose exactly ONE label from the allowed categories for the given sentence (which is an answer in response to a question).
Rules:
- Pick a category ONLY if the answer clearly asserts that option.
- If the answer is unclear, hedged, contains multiple possibilities, refuses to answer,
  or does not correspond to any allowed category, return "Other".
Return the result by calling the provided function."""

def build_tools(categories: List[str]) -> List[Dict]:
    """Define a function/tool with a strict enum of allowed labels (includes 'Other')."""
    enum = list(categories) + ["Other"]
    return [{
        "type": "function",
        "function": {
            "name": "record_classification",
            "description": "Record the single best category for the answer.",
            "parameters": {
                "type": "object",
                "properties": {
                    "category": {"type": "string", "enum": enum},
                    "reason": {"type": "string", "description": "Short justification (optional)."}
                },
                "required": ["category"],
                "additionalProperties": False
            }
        }
    }]

async def classify_once(client: AsyncOpenAI, model: str,
                        answer: str, categories: List[str],
                        retries: int = 5) -> str:
    """Classify one answer into a category (or Other) with simple exponential backoff."""
    tools = build_tools(categories)
    wait = 0.5
    while True:
        try:
            resp = await client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": (
                        f"Allowed categories: {', '.join(categories)}; plus Other.\n"
                        f"Statement: {answer.strip()}"
                    )}
                ],
                tools=tools,
                tool_choice={"type": "function", "function": {"name": "record_classification"}},
            )
            tool_calls = resp.choices[0].message.tool_calls or []
            if not tool_calls:
                return "Other"
            args = tool_calls[0].function.arguments
            data = json.loads(args) if isinstance(args, str) else args
            label = data.get("category", "Other")
            return label if label in (set(categories) | {"Other"}) else "Other"

        except APIConnectionError:
            pass  # transient; retry
        except APIStatusError as e:
            if getattr(e, "status_code", 500) not in (429, 500, 502, 503, 504):
                return "Other"
        except APIError:
            pass

        retries -= 1
        if retries < 0:
            return "Other"
        await asyncio.sleep(wait + random.uniform(0, wait * 0.3))
        wait = min(wait * 2, 8)

# ------------------------------- CSV utilities ------------------------------ #

TOKEN_FIELDS = ("input_tokens", "output_tokens", "reasoning_tokens")

def _coerce_token(value):
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    try:
        return float(text)
    except ValueError:
        return None

def _format_metric(value):
    if value is None:
        return ""
    if isinstance(value, float) and math.isnan(value):
        return ""
    if isinstance(value, (int, float)):
        numeric = float(value)
        if numeric.is_integer():
            return str(int(numeric))
        return f"{numeric:.2f}"
    return str(value)

def _compute_stats(values):
    cleaned = [v for v in values if v is not None]
    if not cleaned:
        return {}
    total = sum(cleaned)
    count = len(cleaned)
    mean = total / count
    std = pstdev(cleaned) if count > 1 else 0.0
    return {
        "count": count,
        "sum": total,
        "mean": mean,
        "std": std,
        "min": min(cleaned),
        "max": max(cleaned),
    }


async def run(args):
    if not os.getenv("OPENAI_API_KEY"):
        raise SystemExit("Please set OPENAI_API_KEY")

    # Read answers
    with open(args.input, newline="", encoding="utf-8") as f:
        # Extract first row (metadata)
        meta_reader = csv.reader(f)
        try:
            csv_metadata = next(meta_reader)  # stored for potential later use
        except StopIteration:
            raise SystemExit("Input CSV appears empty.")

        # Now use the rest as a table, with the next row as the header
        reader = csv.DictReader(f)
        rows = list(reader)
        if not rows:
            raise SystemExit("No data rows found after metadata/header row.")
        answer_col = "answer"
        token_accumulator = {field: [] for field in TOKEN_FIELDS if field in (reader.fieldnames or [])}
        raw_answers = []
        for r in rows:
            raw_answers.append(r.get(answer_col, "").strip())
            for field in token_accumulator:
                token_value = _coerce_token(r.get(field))
                if token_value is not None:
                    token_accumulator[field].append(token_value)

    # Deduplicate to save tokens
    unique_answers = {}
    for a in raw_answers:
        unique_answers.setdefault(a, None)

    client = AsyncOpenAI(max_retries=0)

    # Classify in parallel (limit concurrency)
    sem = asyncio.Semaphore(args.concurrency)

    async def classify_guarded(ans: str):
        async with sem:
            return await classify_once(
                client=client,
                model=args.model,
                answer=ans,
                categories=args.categories,
                retries=args.retries,
            )

    tasks = {ans: asyncio.create_task(classify_guarded(ans)) for ans in unique_answers}
    for ans, task in tasks.items():
        unique_answers[ans] = await task

    # Tally
    counts = Counter()
    allowed = list(args.categories) + ["Other"]
    for a in raw_answers:
        label = unique_answers.get(a, "Other")
        if label not in allowed:
            label = "Other"
        counts[label] += 1

    # Compute token statistics
    token_stats = {field: _compute_stats(token_accumulator.get(field, [])) for field in TOKEN_FIELDS}
    metrics_order = ("count", "sum", "mean", "std", "min", "max")

    if args.output is None:
        base, ext = os.path.splitext(args.input)
        args.output = f"{base}_tally.csv"
    output_path = Path(args.output)
    with output_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(csv_metadata)
        for cat in args.categories:
            writer.writerow([cat, counts.get(cat, 0)])
        writer.writerow(["Other", counts.get("Other", 0)])

    token_rows = []
    for metric in metrics_order:
        row = [metric]
        for field in TOKEN_FIELDS:
            value = token_stats.get(field, {}).get(metric)
            row.append(_format_metric(value))
        if any(cell for cell in row[1:]):
            token_rows.append(row)

    if token_rows:
        token_stats_path = output_path.with_name(f"{output_path.stem}_token_stats{output_path.suffix}")
        with token_stats_path.open("w", newline="", encoding="utf-8") as tf:
            writer = csv.writer(tf)
            writer.writerow(["metric", *TOKEN_FIELDS])
            writer.writerows(token_rows)
        print(f"Token stats saved to: {token_stats_path}")

def parse_args():
    p = argparse.ArgumentParser(description="Tally answers into predefined categories using OpenAI.")
    p.add_argument("--input", "-i", default="answers.csv", help="Path to input CSV (with an 'answer' column or similar).")
    p.add_argument("--output", "-o", default=None, help="Where to write the summary CSV.")
    p.add_argument("--model", "-m", default="gpt-5-chat-latest", help="Model to use for classification.")
    p.add_argument("--categories", nargs="+", required=True, help="List of category labels (Other is added automatically).")
    p.add_argument("--concurrency", "-c", type=int, default=8, help="Max concurrent API calls.")
    p.add_argument("--retries", type=int, default=5, help="Retries per classification on transient errors.")
    return p.parse_args()

if __name__ == "__main__":
    asyncio.run(run(parse_args()))
