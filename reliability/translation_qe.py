"""Reference-free translation quality estimation of the translated benchmarks (Appendix B).

Every translated prompt (650 per language: 250 XSTest + 200 OR-Bench + 200 OR-Bench-Hard)
is scored against its English source with two QE metrics from ``unbabel-comet`` in
source-only mode (no reference), both on a 0-1 scale where higher is better:

  - CometKiwi-XL  (Unbabel/wmt23-cometkiwi-da-xl; gated on the Hugging Face Hub)
  - xCOMET-XL     (Unbabel/XCOMET-XL)

Requires ``pip install "unbabel-comet>=2.2.4"``, one GPU, and Hugging Face access to the
gated CometKiwi checkpoint (``huggingface-cli login`` or HF_TOKEN).

Input:  data/<benchmark>/<language>.json   (fields ``prompt`` and ``prompt_original``)
Output: outputs/reliability/translation_qe/per_prompt_scores.json
        outputs/reliability/translation_qe/summary.csv   (per language and per (language, benchmark))

Usage:
    python reliability/translation_qe.py
    python reliability/translation_qe.py --skip_cometkiwi   # reuse saved CometKiwi scores, run xCOMET only
"""

from __future__ import annotations

import argparse
import csv
import gc
import statistics
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from reliability.common import reliability_dir
from safety_cost.config import BENCHMARKS, NON_ENGLISH, prompts_path
from safety_cost.utils import load_json, save_json

COMETKIWI_MODEL = "Unbabel/wmt23-cometkiwi-da-xl"
XCOMET_MODEL = "Unbabel/XCOMET-XL"
METRICS = ("cometkiwi", "xcomet")


def collect_pairs(languages: list[str], benchmarks: list[str], data_dir: Path | None) -> list[dict]:
    pairs = []
    for language in languages:
        for benchmark in benchmarks:
            path = prompts_path(benchmark, language, data_dir)
            if not path.exists():
                print(f"WARN: missing {path}", file=sys.stderr)
                continue
            for idx, item in enumerate(load_json(path)):
                src, mt = item.get("prompt_original"), item.get("prompt")
                if not src or not mt:
                    continue
                pairs.append({"lang": language, "benchmark": benchmark, "idx": idx, "src": src, "mt": mt})
    return pairs


def run_qe(model_name: str, pairs: list[dict], batch_size: int, gpus: int) -> list[float]:
    """Score (src, mt) pairs; omitting ``ref`` runs both models in QE (source-only) mode."""
    from comet import download_model, load_from_checkpoint

    print(f"[{model_name}] loading ...", flush=True)
    model = load_from_checkpoint(download_model(model_name))
    inputs = [{"src": p["src"], "mt": p["mt"]} for p in pairs]
    print(f"[{model_name}] scoring {len(inputs)} pairs (batch={batch_size}) ...", flush=True)
    return model.predict(inputs, batch_size=batch_size, gpus=gpus, progress_bar=True).scores


def free_gpu() -> None:
    gc.collect()
    try:
        import torch
        torch.cuda.empty_cache()
    except ImportError:
        pass


def load_prior(pairs: list[dict], per_prompt_file: Path, metric: str) -> None:
    """Fill ``metric`` from a previous run (used with --skip_<metric>)."""
    prior = {}
    if per_prompt_file.exists():
        prior = {(p["lang"], p["benchmark"], p["idx"]): p.get(metric) for p in load_json(per_prompt_file)}
    for p in pairs:
        p[metric] = prior.get((p["lang"], p["benchmark"], p["idx"]))
    print(f"  loaded prior {metric} scores: {sum(p[metric] is not None for p in pairs)}/{len(pairs)}")


def aggregate(pairs: list[dict]) -> list[dict]:
    by_lang = defaultdict(lambda: {m: [] for m in METRICS})
    by_lang_bench = defaultdict(lambda: {m: [] for m in METRICS})
    for p in pairs:
        for m in METRICS:
            if p.get(m) is None:
                continue
            by_lang[p["lang"]][m].append(p[m])
            by_lang_bench[(p["lang"], p["benchmark"])][m].append(p[m])

    def stat(xs):
        if not xs:
            return None, None
        return round(statistics.mean(xs), 4), (round(statistics.stdev(xs), 4) if len(xs) > 1 else None)

    def row(lang, bench, d):
        ck_m, ck_s = stat(d["cometkiwi"])
        xc_m, xc_s = stat(d["xcomet"])
        return {"lang": lang, "benchmark": bench, "n_cometkiwi": len(d["cometkiwi"]), "n_xcomet": len(d["xcomet"]),
                "cometkiwi_mean": ck_m, "cometkiwi_std": ck_s, "xcomet_mean": xc_m, "xcomet_std": xc_s}

    return ([row(lang, "all", d) for lang, d in by_lang.items()]
            + [row(lang, bench, d) for (lang, bench), d in by_lang_bench.items()])


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--languages", nargs="+", default=list(NON_ENGLISH), choices=NON_ENGLISH)
    parser.add_argument("--benchmarks", nargs="+", default=list(BENCHMARKS), choices=BENCHMARKS)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--gpus", type=int, default=1)
    parser.add_argument("--skip_cometkiwi", action="store_true", help="reuse CometKiwi scores of a previous run")
    parser.add_argument("--skip_xcomet", action="store_true", help="only run CometKiwi")
    parser.add_argument("--data_dir", type=Path, default=None)
    parser.add_argument("--output_dir", type=Path, default=None)
    args = parser.parse_args()

    out_dir = reliability_dir("translation_qe", args.output_dir)
    per_prompt_file = out_dir / "per_prompt_scores.json"
    pairs = collect_pairs(args.languages, args.benchmarks, args.data_dir)
    print(f"collected {len(pairs)} (source, translation) pairs")

    if args.skip_cometkiwi:
        load_prior(pairs, per_prompt_file, "cometkiwi")
    else:
        for p, s in zip(pairs, run_qe(COMETKIWI_MODEL, pairs, args.batch_size, args.gpus)):
            p["cometkiwi"] = s
        save_json(pairs, per_prompt_file)  # checkpoint before loading the second model
        free_gpu()

    if args.skip_xcomet:
        for p in pairs:
            p["xcomet"] = None
    else:
        for p, s in zip(pairs, run_qe(XCOMET_MODEL, pairs, args.batch_size, args.gpus)):
            p["xcomet"] = s
    save_json(pairs, per_prompt_file)

    rows = aggregate(pairs)
    with open(out_dir / "summary.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {per_prompt_file} and {out_dir / 'summary.csv'}")
    for r in rows:
        if r["benchmark"] == "all":
            print(f"  {r['lang']:<11} CometKiwi-XL {r['cometkiwi_mean']} (+-{r['cometkiwi_std']})  "
                  f"xCOMET-XL {r['xcomet_mean']} (+-{r['xcomet_std']})")


if __name__ == "__main__":
    main()
