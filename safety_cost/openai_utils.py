"""Helpers for the OpenAI API (client creation, Batch API jobs, cost estimates).

The API key is read from the ``OPENAI_API_KEY`` environment variable (or a ``.env``
file in the repository root, see ``.env.example``). Keys are never stored in code.
"""

from __future__ import annotations

import json
import os
import time
from datetime import datetime
from pathlib import Path

from .config import OUTPUT_DIR, REPO_ROOT

# USD per 1M tokens: (input, cached input, output). Used only for the cost estimate
# printed before a batch job is submitted (Batch API is billed at a discount).
MODEL_PRICING = {
    "gpt-5.2": (1.75, 0.175, 14.00),
    "gpt-5-mini": (0.25, 0.025, 2.00),
    "gpt-4.1": (2.00, 0.50, 8.00),
    "gpt-4.1-mini": (0.40, 0.10, 1.60),
    "gpt-4o": (2.50, 1.25, 10.00),
}


def _load_dotenv() -> None:
    env_file = REPO_ROOT / ".env"
    if not env_file.exists():
        return
    for line in env_file.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))


def get_client():
    from openai import OpenAI

    _load_dotenv()
    if not os.environ.get("OPENAI_API_KEY"):
        raise RuntimeError("Set OPENAI_API_KEY (environment variable or .env file).")
    return OpenAI()


def get_env(name: str) -> str:
    _load_dotenv()
    value = os.environ.get(name)
    if not value:
        raise RuntimeError(f"Set {name} (environment variable or .env file).")
    return value


# ---------------------------------------------------------------------------
# Request construction / response parsing
# ---------------------------------------------------------------------------

def responses_task(custom_id: str, model: str, content: str, reasoning_effort: str | None = None,
                   **extra) -> dict:
    """A Batch API task for the ``/v1/responses`` endpoint."""
    body = {"model": model, "input": [{"role": "user", "content": content}], **extra}
    if reasoning_effort and model.startswith("gpt-5"):
        body["reasoning"] = {"effort": reasoning_effort}
    return {"custom_id": custom_id, "method": "POST", "url": "/v1/responses", "body": body}


def chat_task(custom_id: str, model: str, content: str, **extra) -> dict:
    """A Batch API task for the ``/v1/chat/completions`` endpoint."""
    body = {"model": model, "messages": [{"role": "user", "content": content}], **extra}
    return {"custom_id": custom_id, "method": "POST", "url": "/v1/chat/completions", "body": body}


def extract_output_text(batch_result: dict) -> str | None:
    """Return the generated text of one Batch API result line (``None`` on error)."""
    response = batch_result.get("response") or {}
    body = response.get("body") or {}
    if isinstance(body, str):
        body = json.loads(body)
    if batch_result.get("error") or body.get("error"):
        return None
    # /v1/responses
    for item in body.get("output", []) or []:
        if item.get("type") == "message":
            for content in item.get("content", []):
                if content.get("type") == "output_text":
                    return content.get("text", "").strip()
    # /v1/chat/completions
    if body.get("choices"):
        return (body["choices"][0].get("message", {}).get("content") or "").strip()
    return None


# ---------------------------------------------------------------------------
# Cost estimate
# ---------------------------------------------------------------------------

def count_input_tokens(tasks: list[dict]) -> int:
    from tiktoken import encoding_for_model

    enc = encoding_for_model("gpt-4")
    total = 0
    for task in tasks:
        body = task.get("body", {})
        for message in body.get("input", body.get("messages", [])):
            total += len(enc.encode(message.get("content", ""))) + 4
    return total


def confirm_cost(tasks: list[dict], model: str, assume_yes: bool = False) -> bool:
    n_tokens = count_input_tokens(tasks)
    price = MODEL_PRICING.get(model, MODEL_PRICING["gpt-5-mini"])[0]
    print(f"{len(tasks):,} requests, {n_tokens:,} input tokens "
          f"(~${n_tokens / 1e6 * price:.2f} for input tokens with {model}, before batch discount)")
    if assume_yes:
        return True
    return input("Submit batch job? [y/N] ").strip().lower() == "y"


# ---------------------------------------------------------------------------
# Batch API
# ---------------------------------------------------------------------------

def new_batch_dir(kind: str, output_dir: Path | None = None) -> Path:
    batch_dir = Path(output_dir or OUTPUT_DIR) / "batches" / kind / datetime.now().strftime("%y%m%d_%H%M%S")
    batch_dir.mkdir(parents=True, exist_ok=True)
    return batch_dir


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def read_jsonl(path: Path) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def run_batch_job(client, request_file: Path, result_file: Path, endpoint: str,
                  batch_job_id: str | None = None, poll_seconds: int = 20) -> list[dict]:
    """Upload ``request_file``, wait for the job to finish, and download the results.

    Pass ``batch_job_id`` to resume waiting for a job that was already submitted.
    """
    if batch_job_id is None:
        with open(request_file, "rb") as f:
            uploaded = client.files.create(file=f, purpose="batch")
        job = client.batches.create(input_file_id=uploaded.id, endpoint=endpoint, completion_window="24h")
        batch_job_id = job.id
        (Path(request_file).parent / "batch_job_id.txt").write_text(batch_job_id)
        print(f"Submitted batch job {batch_job_id} (saved to batch_job_id.txt; "
              f"pass --batch_job_id to resume if interrupted)")

    while True:
        job = client.batches.retrieve(batch_job_id)
        if job.status in ("completed", "failed", "cancelled", "expired", "cancelling"):
            break
        print(f"[{datetime.now():%H:%M:%S}] batch {batch_job_id}: {job.status} {job.request_counts}")
        time.sleep(poll_seconds)

    if job.status != "completed" or not job.output_file_id:
        raise RuntimeError(f"Batch job {batch_job_id} ended with status {job.status}")

    content = client.files.content(job.output_file_id).content
    result_file = Path(result_file)
    result_file.parent.mkdir(parents=True, exist_ok=True)
    result_file.write_bytes(content)
    print(f"Saved batch results to {result_file}")
    return read_jsonl(result_file)
