"""Submit-or-resume wrapper around the OpenAI Batch API used by the judge scripts.

Each run creates ``outputs/batches/<kind>/<timestamp>/`` containing
``requests.jsonl``, ``metadata.json`` (custom_id -> bookkeeping), ``batch_job_id.txt``
and, once finished, ``results.jsonl``. Pass that directory to ``--resume`` to pick up
a job that was interrupted while waiting.
"""

from __future__ import annotations

from pathlib import Path

from .openai_utils import (confirm_cost, get_client, new_batch_dir, read_jsonl, run_batch_job,
                           write_jsonl)
from .utils import load_json, save_json


def execute(tasks: list[dict], metadata: dict[str, dict], kind: str, model: str, endpoint: str,
            assume_yes: bool = False, resume: str | Path | None = None,
            output_dir: Path | None = None):
    """Run (or resume) a batch job and return ``(results, metadata)``.

    Returns ``(None, None)`` if the user declines the cost prompt or there is nothing to do.
    """
    if resume:
        batch_dir = Path(resume)
        metadata = load_json(batch_dir / "metadata.json")
        result_file = batch_dir / "results.jsonl"
        if result_file.exists():
            return read_jsonl(result_file), metadata
        job_file = batch_dir / "batch_job_id.txt"
        job_id = job_file.read_text().strip() if job_file.exists() else None
    else:
        if not tasks:
            print("Nothing to submit.")
            return None, None
        if not confirm_cost(tasks, model, assume_yes):
            print("Aborted.")
            return None, None
        batch_dir = new_batch_dir(kind, output_dir)
        write_jsonl(batch_dir / "requests.jsonl", tasks)
        save_json(metadata, batch_dir / "metadata.json")
        job_id = None

    print(f"Batch directory: {batch_dir}")
    results = run_batch_job(get_client(), batch_dir / "requests.jsonl", batch_dir / "results.jsonl",
                            endpoint=endpoint, batch_job_id=job_id)
    return results, metadata


def add_batch_args(parser) -> None:
    parser.add_argument("--yes", action="store_true", help="Submit without the cost confirmation prompt.")
    parser.add_argument("--resume", default=None,
                        help="Batch directory (outputs/batches/<kind>/<timestamp>) of a job to resume.")
    parser.add_argument("--summarize_only", action="store_true",
                        help="Skip judging and only aggregate existing judgment files.")
