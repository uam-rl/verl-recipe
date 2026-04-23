#!/usr/bin/env python3
"""Prepare APPS stdin problems for VERL GRPO with a Typst harness."""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from pathlib import Path
from typing import Any

import datasets
from huggingface_hub import hf_hub_download


DATA_SOURCE = "typst_apps"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="codeparrot/apps")
    parser.add_argument("--split", default="train")
    parser.add_argument("--local_save_dir", default="/workspace/typst_apps_data")
    parser.add_argument("--harness_dir", default="/workspace/typst_harness")
    parser.add_argument("--val_size", type=int, default=256)
    parser.add_argument("--max_train_samples", type=int, default=0)
    parser.add_argument("--max_val_samples", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--include_example_ids_in_val", action="store_true", default=True)
    parser.add_argument("--no_include_example_ids_in_val", dest="include_example_ids_in_val", action="store_false")
    return parser.parse_args()


def load_example_ids(harness_dir: Path) -> set[int]:
    ids: set[int] = set()
    for path in harness_dir.glob("sol_apps_*.typ"):
        match = re.fullmatch(r"sol_apps_(\d+)\.typ", path.name)
        if match:
            ids.add(int(match.group(1)))
    return ids


def parse_input_output(raw: str) -> dict[str, Any] | None:
    try:
        value = json.loads(raw)
    except Exception:
        return None
    if not isinstance(value, dict):
        return None
    if "inputs" not in value or "outputs" not in value:
        return None
    return value


def make_stdin_tests(input_output: dict[str, Any]) -> list[dict[str, str]] | None:
    if "fn_name" in input_output:
        return None
    inputs = input_output.get("inputs")
    outputs = input_output.get("outputs")
    if not isinstance(inputs, list) or not isinstance(outputs, list):
        return None
    tests = []
    for inp, out in zip(inputs, outputs, strict=False):
        tests.append({"input": str(inp), "expected": str(out), "testtype": "stdin"})
    return tests or None


def problem_id(example: dict[str, Any], idx: int) -> int:
    for key in ("problem_id", "id"):
        value = example.get(key)
        if value is None:
            continue
        try:
            return int(value)
        except Exception:
            pass
    return idx


def title_for(example: dict[str, Any], pid: int) -> str:
    for key in ("name", "title", "question_title"):
        value = example.get(key)
        if value:
            return str(value)
    return f"APPS problem {pid}"


def prompt_for(example: dict[str, Any], pid: int) -> str:
    question = str(example.get("question") or example.get("problem") or "").strip()
    starter = str(example.get("starter_code") or "").strip()
    pieces = [
        "Write a Typst solution for the programming problem below.",
        "",
        "Follow this contract exactly:",
        "- Output only Typst code. No prose, no markdown fences, no explanation, no thinking aloud.",
        "- Define exactly one function named `solve` with signature `#let solve(input) = { ... }`.",
        "- `input` is the raw stdin string. Parse it exactly according to the statement and return the exact stdout string.",
        "- Use only real Typst syntax and real Typst methods. Do not invent helpers, iterators, or language features.",
        "- If the statement has multiple test cases, preserve the exact order and separators required by the statement.",
        "- Put the final answer in a single Typst code block if you use one; the block should contain only the `solve` function.",
        "",
        "Required shape:",
        "#let solve(input) = {",
        "  // parse the raw stdin string",
        "  // compute the answer",
        "  // return stdout as a string value",
        "}",
        "",
        "Example shape only:",
        "```typst",
        "",
        "#let solve(input) = {",
        "  // parse input",
        "  // compute answer",
        "  // return stdout string",
        "}",
        "```",
        "",
        f"Problem id: {pid}",
        "",
        question,
    ]
    if starter:
        pieces.extend(["", "Original starter code:", "", starter])
    return "\n".join(pieces).strip()


def load_raw_rows(dataset_name: str, split: str) -> list[dict[str, Any]]:
    try:
        return list(datasets.load_dataset(dataset_name, split=split, trust_remote_code=True))
    except RuntimeError as exc:
        if "Dataset scripts are no longer supported" not in str(exc):
            raise

    path = hf_hub_download(dataset_name, f"{split}.jsonl", repo_type="dataset")
    rows = []
    with open(path, encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def convert_row(example: dict[str, Any], idx: int, split: str, example_ids: set[int]) -> dict[str, Any] | None:
    io = parse_input_output(str(example.get("input_output", "")))
    if io is None:
        return None
    tests = make_stdin_tests(io)
    if tests is None:
        return None

    pid = problem_id(example, idx)
    title = title_for(example, pid)
    difficulty = str(example.get("difficulty") or "")
    url = str(example.get("url") or "")
    test_json = json.dumps(tests, ensure_ascii=False)

    return {
        "data_source": DATA_SOURCE,
        "prompt": [{"role": "user", "content": prompt_for(example, pid)}],
        "ability": "code",
        "reward_model": {"style": "rule", "ground_truth": test_json},
        "extra_info": {
            "split": split,
            "index": idx,
            "problem_id": pid,
            "title": title,
            "difficulty": difficulty,
            "url": url,
            "test_count": len(tests),
            "is_example_id": pid in example_ids,
        },
    }


def main() -> None:
    sys.set_int_max_str_digits(100000)
    args = parse_args()
    save_dir = Path(args.local_save_dir).expanduser()
    save_dir.mkdir(parents=True, exist_ok=True)

    harness_dir = Path(args.harness_dir).expanduser()
    example_ids = load_example_ids(harness_dir)
    print(f"Loaded {len(example_ids)} APPS example ids from {harness_dir}", flush=True)

    raw = load_raw_rows(args.dataset, args.split)
    rows = []
    skipped = 0
    for idx, example in enumerate(raw):
        row = convert_row(example, idx, args.split, example_ids)
        if row is None:
            skipped += 1
            continue
        rows.append(row)

    if not rows:
        raise SystemExit("No stdin APPS rows were converted.")

    ds = datasets.Dataset.from_list(rows).shuffle(seed=args.seed)
    examples = ds.filter(lambda row: bool(row["extra_info"]["is_example_id"]))
    train_pool = ds.filter(lambda row: not bool(row["extra_info"]["is_example_id"]))

    val_size = min(args.val_size, len(train_pool))
    val = train_pool.select(range(val_size))
    train = train_pool.select(range(val_size, len(train_pool)))

    if args.include_example_ids_in_val and len(examples) > 0:
        val = datasets.concatenate_datasets([examples, val])

    if args.max_train_samples and args.max_train_samples > 0:
        train = train.select(range(min(args.max_train_samples, len(train))))
    if args.max_val_samples and args.max_val_samples > 0:
        val = val.select(range(min(args.max_val_samples, len(val))))

    train_path = save_dir / "train.parquet"
    val_path = save_dir / "validation.parquet"
    train.to_parquet(str(train_path))
    val.to_parquet(str(val_path))

    meta = {
        "dataset": args.dataset,
        "source_split": args.split,
        "converted_rows": len(rows),
        "skipped_rows": skipped,
        "example_ids_excluded_from_train": sorted(example_ids),
        "train_rows": len(train),
        "validation_rows": len(val),
        "data_source": DATA_SOURCE,
    }
    (save_dir / "metadata.json").write_text(json.dumps(meta, indent=2) + "\n")
    (save_dir / "train_example.json").write_text(json.dumps(train[0], indent=2, ensure_ascii=False) + "\n")
    (save_dir / "validation_example.json").write_text(json.dumps(val[0], indent=2, ensure_ascii=False) + "\n")
    print(json.dumps(meta, indent=2), flush=True)


if __name__ == "__main__":
    main()
