#!/usr/bin/env python3
"""Render incremental Typst GRPO rollouts to a PDF report via Typst."""

from __future__ import annotations

import argparse
import collections
import json
import os
import shutil
import subprocess
import tempfile
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean


@dataclass(frozen=True)
class GroupKey:
    phase: str
    step: int


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


def _json_str(value: object) -> str:
    return json.dumps(value, ensure_ascii=False)


def _typst_raw_block(text: str) -> str:
    return f"#raw({_json_str(text)}, block: true)"


def _format_counter(counter: collections.Counter[str]) -> str:
    if not counter:
        return "<none>"
    items = ", ".join(f"{key}: {value}" for key, value in counter.most_common())
    return items


def _termination_label(entry: dict) -> str:
    finish = str(entry.get("finish_reason") or "<none>")
    stop = str(entry.get("stop_reason") or "<none>")
    label = f"{finish}/{stop}"
    if entry.get("hit_response_cap"):
        label += "+cap"
    return label


def _counter_block(counter: collections.Counter[str], limit: int = 8) -> str:
    if not counter:
        return _typst_raw_block("<none>")
    items = "\n".join(f"{key}: {value}" for key, value in counter.most_common(limit))
    return _typst_raw_block(items)


def _compact_text(text: object, limit: int = 140) -> str:
    compact = " ".join(str(text).split())
    if len(compact) > limit:
        return compact[: limit - 1] + "…"
    return compact


def _select_groups(entries: list[dict], step: int | None, all_steps: bool) -> list[tuple[GroupKey, list[dict]]]:
    grouped: dict[tuple[str, int], list[dict]] = collections.defaultdict(list)
    for entry in entries:
        grouped[(str(entry.get("phase", "?")), int(entry.get("step", -1)))].append(entry)

    if not grouped:
        return []

    if all_steps:
        selected = sorted(grouped.items(), key=lambda item: (item[0][1], item[0][0]))
    else:
        if step is None:
            step = max(key[1] for key in grouped)
        selected = [item for item in sorted(grouped.items(), key=lambda item: (item[0][1], item[0][0])) if item[0][1] == step]

    return [(GroupKey(phase=phase, step=step_no), group) for (phase, step_no), group in selected]


def _summary_lines(entries: list[dict]) -> list[str]:
    if not entries:
        return ["- no incremental rollout records found"]

    by_group = _select_groups(entries, step=None, all_steps=True)
    scores = [float(e["reward_score"]) for e in entries if e.get("reward_score") is not None]
    token_counts = [int(e.get("response_token_count", 0)) for e in entries]
    termination = collections.Counter(_termination_label(e) for e in entries)
    errors = collections.Counter(
        _compact_text((e.get("reward_extra_info") or {}).get("error", "") or "<ok>") for e in entries
    )
    hit_cap = sum(1 for e in entries if e.get("hit_response_cap"))
    reasoning_closed = sum(1 for e in entries if e.get("reasoning_close_seen"))
    solve_seen = sum(1 for e in entries if e.get("typst_solve_seen"))

    lines = [
        f"- total records: {len(entries)}",
        f"- groups: {len(by_group)}",
        f"- score mean: {mean(scores):.4f}" if scores else "- score mean: n/a",
        f"- response tokens mean: {mean(token_counts):.1f}" if token_counts else "- response tokens mean: 0.0",
        f"- response tokens max: {max(token_counts) if token_counts else 0}",
        f"- think closed: {reasoning_closed}/{len(entries)}",
        f"- solve seen: {solve_seen}/{len(entries)}",
        f"- hit cap: {hit_cap}/{len(entries)}",
        f"- termination: {_format_counter(termination)}",
        "- top errors:",
        _counter_block(errors),
    ]
    return lines


def _group_summary(group: list[dict]) -> list[str]:
    scores = [float(e["reward_score"]) for e in group if e.get("reward_score") is not None]
    token_counts = [int(e.get("response_token_count", 0)) for e in group]
    termination = collections.Counter(_termination_label(e) for e in group)
    errors = collections.Counter(
        _compact_text((e.get("reward_extra_info") or {}).get("error", "") or "<ok>") for e in group
    )
    reasoning_closed = sum(1 for e in group if e.get("reasoning_close_seen"))
    solve_seen = sum(1 for e in group if e.get("typst_solve_seen"))
    hit_cap = sum(1 for e in group if e.get("hit_response_cap"))

    lines = [
        f"- count: {len(group)}",
        f"- score mean: {mean(scores):.4f}" if scores else "- score mean: n/a",
        f"- response tokens mean: {mean(token_counts):.1f}" if token_counts else "- response tokens mean: 0.0",
        f"- response tokens max: {max(token_counts) if token_counts else 0}",
        f"- think closed: {reasoning_closed}/{len(group)}",
        f"- solve seen: {solve_seen}/{len(group)}",
        f"- hit cap: {hit_cap}/{len(group)}",
        f"- termination: {_format_counter(termination)}",
        "- top errors:",
        _counter_block(errors),
    ]
    return lines


def _truncate(text: str, chars: int, from_end: bool = True) -> str:
    if chars <= 0 or len(text) <= chars:
        return text
    if from_end:
        return text[-chars:]
    return text[:chars]


def _sample_var_name(entry: dict) -> str:
    phase = str(entry.get("phase", "sample")).replace("-", "_")
    step = int(entry.get("step", -1))
    sample_index = int(entry.get("sample_index", -1))
    rollout_n = int(entry.get("rollout_n", -1))
    return f"{phase}_{step}_{sample_index}_{rollout_n}"


def _render_sample(entry: dict, response_chars: int, prompt_chars: int, include_prompt: bool) -> str:
    var_name = _sample_var_name(entry)
    response = _truncate(str(entry.get("response_text", "")), response_chars)
    prompt = _truncate(str(entry.get("prompt_text", "")), prompt_chars)
    header = (
        f"=== sample {entry.get('sample_index')} / rollout {entry.get('rollout_n')} "
        f"score={entry.get('reward_score')} termination={_termination_label(entry)} "
        f"tokens={entry.get('response_token_count')}"
    )
    lines = [
        header,
        f"- uid: {_json_str(entry.get('uid'))}",
        f"- data source: {_json_str(entry.get('data_source'))}",
        f"- hit cap: {bool(entry.get('hit_response_cap'))}",
        f"- reasoning open: {bool(entry.get('reasoning_prompt_open'))}",
        f"- reasoning closed: {bool(entry.get('reasoning_close_seen'))}",
        f"- solve seen: {bool(entry.get('typst_solve_seen'))}",
        "- reward extra:",
        _typst_raw_block(_truncate(_json_str(entry.get('reward_extra_info') or {}), 1200)),
    ]
    if include_prompt:
        lines.append("- prompt tail:")
        lines.append(_typst_raw_block(prompt))
    lines.append("- response:")
    lines.append(_typst_raw_block(response))
    return "\n".join(lines)


def _typst_report(entries: list[dict], source_dir: Path, selected_groups: list[tuple[GroupKey, list[dict]]], args: argparse.Namespace) -> str:
    now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
    scope = "all steps" if args.all_steps else ("latest step" if args.step is None else f"step {args.step}")

    lines = [
        '#set page(width: 8.5in, margin: 0.65in)',
        '#set text(font: "DejaVu Sans", size: 9pt)',
        f"= Typst GRPO rollout report",
        f"Generated: {now}",
        f"Source: {source_dir}",
        f"Scope: {scope}",
        "",
        "== Overview",
        *_summary_lines(entries),
    ]

    if not selected_groups:
        lines.extend(["", "== Details", "- no selected groups"])
        return "\n".join(lines)

    for index, (group_key, group) in enumerate(selected_groups):
        lines.extend(
            [
                "",
                f"== {group_key.phase} step {group_key.step}",
                *_group_summary(group),
            ]
        )
        for entry in sorted(group, key=lambda e: (int(e.get("sample_index", -1)), int(e.get("rollout_n", -1)))):
            lines.extend(
                [
                    "",
                    _render_sample(
                        entry,
                        response_chars=args.response_chars,
                        prompt_chars=args.prompt_chars,
                        include_prompt=args.include_prompt,
                    ),
                ]
            )
        if index != len(selected_groups) - 1:
            lines.append("#pagebreak()")

    return "\n".join(lines)


def _typst_binary() -> str:
    candidates = [
        shutil.which("typst"),
        str(Path.home() / ".cargo" / "bin" / "typst"),
        "/usr/local/bin/typst",
        "/usr/bin/typst",
    ]
    for candidate in candidates:
        if candidate and Path(candidate).exists():
            return candidate
    return "typst"


def main() -> None:
    default_path = os.environ.get("INCREMENTAL_ROLLOUT_DATA_DIR")
    if not default_path:
        workspace_path = Path("/workspace/eval_results/typst_grpo_real/incremental_rollouts")
        default_path = str(
            workspace_path
            if workspace_path.exists()
            else Path.home() / "eval_results/typst_grpo_real/incremental_rollouts"
        )

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "path",
        nargs="?",
        default=default_path,
        help="Directory containing incremental rollout JSONL files.",
    )
    parser.add_argument("--output", "-o", default=None, help="PDF output path.")
    parser.add_argument("--step", type=int, default=None, help="Render a single training step.")
    parser.add_argument("--all-steps", action="store_true", help="Render every step in the directory.")
    parser.add_argument("--response-chars", type=int, default=6000, help="Tail characters of each response.")
    parser.add_argument("--prompt-chars", type=int, default=1200, help="Tail characters of each prompt when included.")
    parser.add_argument("--include-prompt", action="store_true", help="Render prompt tails too.")
    parser.add_argument("--keep-typst", action="store_true", help="Keep the intermediate Typst source next to the PDF.")
    args = parser.parse_args()

    source_dir = Path(args.path)
    entries = [entry for _, entry in iter_entries(source_dir)]
    selected_groups = _select_groups(entries, step=args.step, all_steps=args.all_steps)

    if args.output:
        output_pdf = Path(args.output)
    else:
        reports_dir = source_dir.parent / "reports"
        reports_dir.mkdir(parents=True, exist_ok=True)
        if args.all_steps:
            stem = "rollout_report_all_steps"
        elif args.step is None:
            step_tag = max(int(entry.get("step", -1)) for entry in entries) if entries else "empty"
            stem = f"rollout_report_step_{step_tag:08d}" if isinstance(step_tag, int) else "rollout_report_empty"
        else:
            stem = f"rollout_report_step_{args.step:08d}"
        output_pdf = reports_dir / f"{stem}.pdf"

    output_pdf.parent.mkdir(parents=True, exist_ok=True)

    typst_source = _typst_report(entries, source_dir, selected_groups, args)
    typst_path = output_pdf.with_suffix(".typ")
    typst_path.write_text(typst_source, encoding="utf-8")

    typst_bin = _typst_binary()
    subprocess.run([typst_bin, "compile", str(typst_path), str(output_pdf)], check=True)

    if not args.keep_typst:
        typst_path.unlink(missing_ok=True)

    print(output_pdf)


if __name__ == "__main__":
    main()
