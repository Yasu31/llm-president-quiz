#!/usr/bin/env python3
"""
Burst-query an OpenAI model N times in parallel and save answers to CSV.

Usage:
  pip install openai
  export OPENAI_API_KEY=sk-...
  python burst_query.py -q "Who is the current president of the United States?" -n 100 -o answers.csv
"""

import os
import csv
import time
import random
import argparse
import asyncio
from datetime import datetime

from openai import AsyncOpenAI, APIError, APIConnectionError, APIStatusError

DEFAULT_MODEL = os.getenv("OPENAI_MODEL", "gpt-5-chat-latest")

async def ask_once(
    client: AsyncOpenAI,
    idx: int,
    question: str,
    semaphore: asyncio.Semaphore,
    model: str,
    retries: int = 5,
    instructions: str = "",
) -> dict:
    """
    Send one request with simple exponential backoff on transient errors.
    Returns dict with index, text, and latency_ms.
    """
    wait = 0.5  # initial backoff seconds
    while True:
        try:
            async with semaphore:
                t0 = time.perf_counter()
                # Using the Responses API; output_text gives the concatenated text
                resp = await client.responses.create(
                    model=model,
                    instructions=instructions,
                    input=question,
                )
            latency_ms = int((time.perf_counter() - t0) * 1000)
            text = (resp.output_text or "").strip()
            usage = getattr(resp, "usage", None)
            input_tokens = getattr(usage, "input_tokens", 0) if usage else 0
            output_tokens = getattr(usage, "output_tokens", 0) if usage else 0
            model = getattr(resp, "model", model)
            reasoning_tokens = 0
            if usage:
                details = getattr(usage, "output_tokens_details", None)
                reasoning_tokens = getattr(details, "reasoning_tokens", 0) if details else 0
            return {
                "index": idx,
                "text": text,
                "latency_ms": latency_ms,
                "model": model,
                "input_tokens": input_tokens,
                "output_tokens": output_tokens,
                "reasoning_tokens": reasoning_tokens,
            }

        except APIConnectionError as e:
            # network/timeouts => retry
            if retries <= 0:
                return {"index": idx, "text": f"ERROR: {e}", "latency_ms": None, "input_tokens": 0, "output_tokens": 0, "reasoning_tokens": 0}

        except APIStatusError as e:
            # Retry typical transient statuses
            status = getattr(e, "status_code", None)
            if status not in (429, 500, 502, 503, 504):
                return {"index": idx, "text": f"ERROR {status}: {e}", "latency_ms": None, "input_tokens": 0, "output_tokens": 0, "reasoning_tokens": 0}
            if retries <= 0:
                return {"index": idx, "text": f"ERROR {status}: {e}", "latency_ms": None, "input_tokens": 0, "output_tokens": 0, "reasoning_tokens": 0}

        except APIError as e:
            # Unknown API error — retry a few times, then give up
            if retries <= 0:
                return {"index": idx, "text": f"ERROR: {e}", "latency_ms": None, "input_tokens": 0, "output_tokens": 0, "reasoning_tokens": 0}

        # Backoff with jitter
        retries -= 1
        await asyncio.sleep(wait + random.uniform(0, wait * 0.3))
        wait = min(wait * 2.0, 8.0)


async def main():
    parser = argparse.ArgumentParser(description="Parallel OpenAI calls -> CSV")
    parser.add_argument(
        "-q", "--question",
        default="Who is the current president of the United States of America?",
        help="Question to send on every request."
    )
    parser.add_argument(
        "-n", "--num",
        type=int, default=10,
        help="Number of parallel requests to send."
    )
    parser.add_argument(
        "-c", "--concurrency",
        type=int, default=10,
        help="Max number of in-flight requests."
    )
    parser.add_argument(
        "-o", "--output",
        default=f"answers_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
        help="Output CSV path."
    )
    parser.add_argument(
        "-m", "--model",
        default=DEFAULT_MODEL,
        help=f"Model to use (default from OPENAI_MODEL or {DEFAULT_MODEL})."
    )
    parser.add_argument(
        "--retries",
        type=int, default=5,
        help="Max retries per request on transient errors."
    )

    args = parser.parse_args()
    if not os.environ.get("OPENAI_API_KEY"):
        parser.error("Please set OPENAI_API_KEY in your environment.")
    instruction_text = f"Answer the question as accurately and concisely as possible. Today is {datetime.now().strftime('%B %d, %Y')}."

    client = AsyncOpenAI(max_retries=0)  # we implement our own backoff above

    sem = asyncio.Semaphore(args.concurrency)

    print(
        f"Starting {args.num} requests "
        f"(concurrency={args.concurrency}, model={args.model})..."
    )

    tasks = [
        ask_once(
            client=client,
            idx=i + 1,
            question=args.question,
            semaphore=sem,
            model=args.model,
            retries=args.retries,
            instructions=instruction_text
        )
        for i in range(args.num)
    ]

    results = await asyncio.gather(*tasks)
    results.sort(key=lambda r: r["index"])

    # Write CSV: one row per call
    with open(args.output, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["question", args.question, "instructions", instruction_text, "model", args.model, "date", datetime.now().isoformat()])
        writer.writerow(["index", "answer", "input_tokens", "output_tokens", "reasoning_tokens", "model"])
        for row in results:
            writer.writerow([
                row["index"],
                row["text"],
                row["input_tokens"],
                row["output_tokens"],
                row["reasoning_tokens"],
                row["model"],
            ])

    print(f"Wrote {len(results)} rows to {args.output}")

if __name__ == "__main__":
    asyncio.run(main())
