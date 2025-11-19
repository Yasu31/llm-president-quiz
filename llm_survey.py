#!/usr/bin/env python3
"""
Burst-query an LLM (OpenAI or Google's Gemini) N times in parallel and save
answers to CSV.

Usage examples:
  export OPENAI_API_KEY=sk-...
  python llm_survey.py -q "Who is the current president?" --provider openai --model gpt-5.1

  export GEMINI_API_KEY=...   # or GOOGLE_API_KEY
  python llm_survey.py --provider gemini --model gemini-1.5-flash
"""

import os
import csv
import time
import random
import argparse
import asyncio
from datetime import datetime

try:
    from openai import AsyncOpenAI, APIError, APIConnectionError, APIStatusError

    OPENAI_AVAILABLE = True
except ImportError:  # pragma: no cover - optional dependency
    AsyncOpenAI = None  # type: ignore
    APIError = APIConnectionError = APIStatusError = Exception  # type: ignore
    OPENAI_AVAILABLE = False

try:
    from google import genai
    GEMINI_AVAILABLE = True
except ImportError:  # pragma: no cover - optional dependency
    genai = None  # type: ignore
    GEMINI_AVAILABLE = False

DEFAULT_OPENAI_MODEL = "gpt-5.1"
DEFAULT_GEMINI_MODEL = "gemini-2.5-flash"


async def ask_once_openai(
    client: AsyncOpenAI,
    idx: int,
    question: str,
    semaphore: asyncio.Semaphore,
    model: str,
    retries: int = 5,
    instructions: str = "",
) -> dict:
    """Send one OpenAI request with exponential backoff."""

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
                    reasoning={
                        # try also: low, medium, high
                        "effort": "none",
                    }
                )
            latency_ms = int((time.perf_counter() - t0) * 1000)
            text = (resp.output_text or "").strip()
            usage = getattr(resp, "usage", None)
            input_tokens = getattr(usage, "input_tokens", 0) if usage else 0
            output_tokens = getattr(usage, "output_tokens", 0) if usage else 0
            model_name = getattr(resp, "model", model)
            reasoning_tokens = 0
            if usage:
                details = getattr(usage, "output_tokens_details", None)
                reasoning_tokens = getattr(details, "reasoning_tokens", 0) if details else 0
            return {
                "index": idx,
                "text": text,
                "latency_ms": latency_ms,
                "model": model_name,
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


async def ask_once_gemini(
    client: "genai.Client",
    idx: int,
    question: str,
    semaphore: asyncio.Semaphore,
    model: str,
    retries: int = 5,
    instructions: str = "",
) -> dict:
    """Send one Gemini request with retries and return metadata dict."""

    wait = 0.5
    prompt = f"{instructions}\n\nQuestion: {question}" if instructions else question

    while True:
        try:
            async with semaphore:
                t0 = time.perf_counter()
                resp = await asyncio.to_thread(client.models.generate_content,
                                               model=model,
                                               contents=prompt,)
            latency_ms = int((time.perf_counter() - t0) * 1000)
            text = getattr(resp, "text", "").strip()
            usage = getattr(resp, "usage_metadata", None)
            input_tokens = getattr(usage, "prompt_token_count", 0) if usage else 0
            output_tokens = getattr(usage, "candidates_token_count", 0) if usage else 0
            model_version = getattr(resp, "model_version", model)

            return {
                "index": idx,
                "text": text,
                "latency_ms": latency_ms,
                "model": model_version,
                "input_tokens": input_tokens,
                "output_tokens": output_tokens,
                "reasoning_tokens": 0,
            }

        except Exception as e:  # Broad catch due to varied client exceptions
            if retries <= 0:
                return {"index": idx, "text": f"ERROR: {e}", "latency_ms": None, "input_tokens": 0, "output_tokens": 0, "reasoning_tokens": 0}

        retries -= 1
        await asyncio.sleep(wait + random.uniform(0, wait * 0.3))
        wait = min(wait * 2.0, 8.0)


def _default_model_for_provider(provider: str) -> str:
    return DEFAULT_OPENAI_MODEL if provider == "openai" else DEFAULT_GEMINI_MODEL


async def main():
    parser = argparse.ArgumentParser(description="Parallel LLM calls -> CSV")
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
        default=None,
        help="Model to use. Defaults depend on provider (OpenAI or Gemini).",
    )
    parser.add_argument(
        "--retries",
        type=int, default=5,
        help="Max retries per request on transient errors.",
    )
    parser.add_argument(
        "-p", "--provider",
        choices=("openai", "gemini"),
        default="openai",
        help="LLM provider to use (default from LLM_PROVIDER or openai).",
    )

    args = parser.parse_args()

    if args.model is None:
        args.model = _default_model_for_provider(args.provider)

    if args.provider == "openai":
        if not OPENAI_AVAILABLE:
            parser.error("OpenAI provider selected but openai package is not installed.")
        if not os.environ.get("OPENAI_API_KEY"):
            parser.error("Please set OPENAI_API_KEY in your environment.")
        client = AsyncOpenAI(max_retries=0)  # type: ignore[arg-type]
        ask_fn = ask_once_openai
    else:
        if not GEMINI_AVAILABLE:
            parser.error("Gemini provider selected but google-genai package is not installed.")
        api_key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
        if not api_key:
            parser.error("Please set GEMINI_API_KEY or GOOGLE_API_KEY in your environment.")
        client = genai.Client(api_key=api_key)
        ask_fn = ask_once_gemini

    instruction_text = f"Answer the question as accurately and concisely as possible. Today is {datetime.now().strftime('%B %d, %Y')}."

    sem = asyncio.Semaphore(args.concurrency)

    print(
        f"Starting {args.num} requests "
        f"(concurrency={args.concurrency}, provider={args.provider}, model={args.model})..."
    )

    tasks = [
        ask_fn(
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
    print(results)  # make it easier to debug

    # Write CSV: one row per call
    with open(args.output, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "question", args.question,
            "instructions", instruction_text,
            "provider", args.provider,
            "model", args.model,
            "date", datetime.now().isoformat()
        ])
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
