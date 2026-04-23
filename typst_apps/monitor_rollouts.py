#!/usr/bin/env python3
"""Summarize incremental Typst GRPO rollouts while a VERL run is active."""

from __future__ import annotations

import argparse
import collections
import json
import os
from pathlib import Path
from statistics import mean


def iter_entries(path: Path):
    for file_path in sorted(path.glob("*.jsonl")):
        with file_path.open("r", encoding="utf-8") as f:
            for line_no, line in enumerate(f, 1):
                if not line.strip():
                    continue
                try:
                    yield file_path, json.loads(line)
                except json.JSONDecodeError as exc:
                    print(f"bad json: {file_path}:{line_no}: {exc}")


def _termination_label(entry: dict) -> str:
    finish = str(entry.get("finish_reason") or "<none>")
    stop = str(entry.get("stop_reason") or "<none>")
    label = f"{finish}/{stop}"
    if entry.get("hit_response_cap"):
        label += "+cap"
    return label


def summarize(entries: list[dict]) -> str:
    if not entries:
        return "no incremental rollout records found"

    by_phase_step: dict[tuple[str, int], list[dict]] = collections.defaultdict(list)
    for entry in entries:
        by_phase_step[(entry.get("phase", "?"), int(entry.get("step", -1)))].append(entry)

    lines = []
    for (phase, step), group in sorted(by_phase_step.items()):
        scores = [float(e["reward_score"]) for e in group if e.get("reward_score") is not None]
        token_counts = [int(e.get("response_token_count", 0)) for e in group]
        termination = collections.Counter(_termination_label(e) for e in group)
        errors = collections.Counter(
            str((e.get("reward_extra_info") or {}).get("error", "")) or "<ok>" for e in group
        )
        reasoning_closed = sum(1 for e in group if e.get("reasoning_close_seen"))
        solve_seen = sum(1 for e in group if e.get("typst_solve_seen"))
        hit_cap = sum(1 for e in group if e.get("hit_response_cap"))
        score_mean = f"{mean(scores):.4f}" if scores else "n/a"
        tokens_avg = f"{mean(token_counts):.1f}" if token_counts else "0.0"
        tokens_max = max(token_counts) if token_counts else 0

        lines.append(
            f"{phase} step {step}: count={len(group)} "
            f"score_mean={score_mean} "
            f"tokens_avg={tokens_avg} "
            f"tokens_max={tokens_max} "
            f"think_closed={reasoning_closed}/{len(group)} "
            f"solve_seen={solve_seen}/{len(group)} "
            f"hit_cap={hit_cap}/{len(group)} "
            f"termination={dict(termination)}"
        )
        lines.append(f"  top errors: {dict(errors.most_common(5))}")

    return "\n".join(lines)


def print_samples(entries: list[dict], limit: int, chars: int) -> None:
    for entry in entries[-limit:]:
        print()
        print(
            f"[{entry.get('phase')} step={entry.get('step')} sample={entry.get('sample_index')} "
            f"rollout={entry.get('rollout_n')} score={entry.get('reward_score')} "
            f"termination={_termination_label(entry)} tokens={entry.get('response_token_count')}]"
        )
        response = entry.get("response_text", "")
        print(response[-chars:])


def main() -> None:
    default_path = os.environ.get("INCREMENTAL_ROLLOUT_DATA_DIR")
    if not default_path:
        workspace_path = Path("/workspace/eval_results/typst_grpo_real/incremental_rollouts")
        default_path = str(workspace_path if workspace_path.exists() else Path.home() / "eval_results/typst_grpo_real/incremental_rollouts")

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "path",
        nargs="?",
        default=default_path,
        help="Directory containing incremental rollout JSONL files.",
    )
    parser.add_argument("--samples", type=int, default=0, help="Print the last N response tails.")
    parser.add_argument("--chars", type=int, default=2000, help="Characters per printed response tail.")
    args = parser.parse_args()

    path = Path(args.path)
    entries = [entry for _, entry in iter_entries(path)]
    print(summarize(entries))
    if args.samples:
        print_samples(entries, args.samples, args.chars)


if __name__ == "__main__":
    main()
