"""Typst APPS reward function for VERL.

Scores completions by the fraction of APPS harness tests passed.
"""

from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Any


def _extract_typst_code(solution_str: str) -> str:
    text = solution_str.strip()
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()

    answer_blocks = re.findall(r"<answer>(.*?)</answer>", text, flags=re.DOTALL | re.IGNORECASE)
    if answer_blocks:
        text = max((block.strip() for block in answer_blocks), key=len)

    fenced = re.findall(r"```(?:typst|typ|text)?\s*(.*?)```", text, flags=re.DOTALL | re.IGNORECASE)
    if fenced:
        fenced = [block.strip() for block in fenced]
        preferred = [block for block in fenced if "#let solve" in block]
        if preferred:
            text = max(preferred, key=len)
        else:
            text = max(fenced, key=len)

    marker = "#let solve"
    pos = text.find(marker)
    if pos >= 0:
        text = text[pos:]

    return text.strip()


def _typst_binary(explicit: str | None = None) -> str:
    candidates = [
        explicit,
        os.environ.get("TYPST_BIN"),
        shutil.which("typst"),
        str(Path.home() / ".cargo" / "bin" / "typst"),
        "/usr/local/bin/typst",
        "/usr/bin/typst",
    ]
    for candidate in candidates:
        if candidate and Path(candidate).exists():
            return candidate
    return "typst"


def _harness_file(harness_dir: str | None) -> Path:
    candidates = [
        harness_dir,
        os.environ.get("TYPST_HARNESS_DIR"),
        "/workspace/typst_harness",
        "/home/user/typst_harness",
    ]
    for candidate in candidates:
        if not candidate:
            continue
        path = Path(candidate).expanduser() / "harness.typ"
        if path.exists():
            return path
    raise FileNotFoundError("Could not find harness.typ; set harness_dir or TYPST_HARNESS_DIR.")


def _load_tests(ground_truth: Any, extra_info: dict[str, Any] | None) -> list[dict[str, Any]]:
    if isinstance(ground_truth, str) and ground_truth.strip():
        return json.loads(ground_truth)
    if extra_info and "tests" in extra_info:
        tests = extra_info["tests"]
        if isinstance(tests, str):
            return json.loads(tests)
        if isinstance(tests, list):
            return tests
    raise ValueError("Missing APPS tests in reward_model.ground_truth.")


def _run_typst(solution_code: str, tests: list[dict[str, Any]], harness: Path, typst_bin: str, timeout: int) -> tuple[list[bool], str | None]:
    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        (tmp_path / "solution.typ").write_text(solution_code)
        shutil.copy(harness, tmp_path / "harness.typ")
        result = subprocess.run(
            [
                typst_bin,
                "query",
                str(tmp_path / "harness.typ"),
                "<results>",
                "--field",
                "value",
                "--one",
                f"--input=tests={json.dumps(tests, ensure_ascii=False)}",
            ],
            capture_output=True,
            text=True,
            timeout=timeout,
        )

    if result.returncode != 0:
        return [], result.stderr.strip()[:800]
    try:
        values = json.loads(result.stdout.strip())
    except Exception as exc:
        return [], f"failed to parse Typst harness output: {exc}: {result.stdout[:300]!r}"
    if not isinstance(values, list):
        return [], f"Typst harness returned non-list output: {values!r}"
    return [bool(value) for value in values], None


def compute_score(
    data_source: str,
    solution_str: str,
    ground_truth: Any,
    extra_info: dict[str, Any] | None = None,
    harness_dir: str | None = None,
    typst_bin: str | None = None,
    timeout: int = 10,
) -> dict[str, Any]:
    """Return pass-ratio reward for a generated Typst solution."""
    del data_source
    extra_info = dict(extra_info or {})

    try:
        tests = _load_tests(ground_truth, extra_info)
        code = _extract_typst_code(solution_str)
        if not code:
            return {"score": 0.0, "acc": False, "passed": 0, "total": len(tests), "error": "empty_code"}
        if "#let solve" not in code:
            return {"score": 0.0, "acc": False, "passed": 0, "total": len(tests), "error": "missing_solve"}

        results, err = _run_typst(
            solution_code=code,
            tests=tests,
            harness=_harness_file(harness_dir),
            typst_bin=_typst_binary(typst_bin),
            timeout=int(timeout),
        )
        if err:
            return {"score": 0.0, "acc": False, "passed": 0, "total": len(tests), "error": err}
        total = len(results)
        passed = sum(results)
        score = float(passed / total) if total else 0.0
        return {
            "score": score,
            "acc": passed == total and total > 0,
            "passed": passed,
            "total": total,
            "error": "",
        }
    except subprocess.TimeoutExpired:
        return {"score": 0.0, "acc": False, "passed": 0, "total": 0, "error": "timeout"}
    except Exception as exc:
        return {"score": 0.0, "acc": False, "passed": 0, "total": 0, "error": f"{type(exc).__name__}: {exc}"}
