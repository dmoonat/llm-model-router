"""Open-source benchmark dataset loader and deterministic ground-truth verifiers.

Fetches real test-split rows from Hugging Face Datasets Server REST API:
1. ARC-Easy (`allenai/ai2_arc`, config=`ARC-Easy`, split=`test`) — Simple science/factual MCQs
2. MATH-500 (`HuggingFaceH4/MATH-500`, config=`default`, split=`test`, Level 5) — Olympiad competition math
3. HumanEval (`openai/openai_humaneval`, config=`openai_humaneval`, split=`test`) — Python algorithm & code generation
"""

from __future__ import annotations

import json
import logging
import re
import ssl
import subprocess
import sys
import urllib.parse
import urllib.request
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import certifi

logger = logging.getLogger(__name__)

_EVAL_DIR = Path(__file__).resolve().parent
DEFAULT_CACHE_PATH = _EVAL_DIR / "data_cache" / "benchmark_samples_v2.json"
HF_ROWS_ENDPOINT = "https://datasets-server.huggingface.co/rows"
DATASET_NAMES = ("ARC-Easy", "MATH-500", "HumanEval")


@dataclass
class BenchmarkSample:
    """Single real open-source evaluation item with ground-truth verification data."""

    sample_id: str
    dataset: str  # "ARC-Easy", "MATH-500", or "HumanEval"
    expected_complexity: str  # "simple" or "complex"
    expected_sub_agent: str  # "fast_qa_agent", "math_analyst_agent", or "code_engineer_agent"
    prompt: str
    ground_truth: str
    test_assertions: list[str] = field(default_factory=list)
    test_imports: list[str] = field(default_factory=list)
    source_url: str = ""


def _fetch_hf_rows(
    dataset: str,
    config: str,
    split: str,
    length: int = 20,
    offset: int = 0,
) -> list[dict[str, Any]]:
    """Fetches rows from the public Hugging Face Datasets Server API."""
    params = urllib.parse.urlencode(
        {
            "dataset": dataset,
            "config": config,
            "split": split,
            "offset": offset,
            "length": length,
        }
    )
    url = f"{HF_ROWS_ENDPOINT}?{params}"
    req = urllib.request.Request(
        url,
        headers={"User-Agent": "laya-adk-model-router-eval/1.0"},
    )
    ssl_ctx = ssl.create_default_context(cafile=certifi.where())
    with urllib.request.urlopen(req, timeout=30, context=ssl_ctx) as resp:
        payload = json.loads(resp.read().decode("utf-8"))
    return [item["row"] for item in payload.get("rows", [])]


def _extract_function_signature(code: str) -> str:
    """Extracts the primary `def func_name(...):` signature from a Python prompt or solution."""
    matches = re.findall(r"^def\s+[a-zA-Z_][a-zA-Z0-9_]*\s*\([^)]*\)\s*(?:->\s*[^:]+)?:", code, flags=re.MULTILINE)
    if matches:
        return matches[-1]
    return "def solution():"


def _read_cached_samples(
    cache_path: Path = DEFAULT_CACHE_PATH,
) -> dict[str, list[BenchmarkSample]]:
    """Reads and groups cached BenchmarkSample items by dataset name if the cache file exists."""
    if not cache_path.exists():
        return {}
    try:
        raw = json.loads(cache_path.read_text(encoding="utf-8"))
        if not isinstance(raw, list):
            return {}
        by_ds: dict[str, list[BenchmarkSample]] = {}
        for item in raw:
            sample = BenchmarkSample(**item)
            by_ds.setdefault(sample.dataset, []).append(sample)
        return by_ds
    except Exception as exc:
        logger.warning("Failed reading benchmark cache at %s (%s); will re-download.", cache_path, exc)
        return {}


def load_arc_easy_samples(
    limit: int = 15,
    cache_path: Path = DEFAULT_CACHE_PATH,
    refresh_cache: bool = False,
) -> list[BenchmarkSample]:
    """Loads real test-split samples from local cache or `allenai/ai2_arc` (`ARC-Easy`)."""
    if not refresh_cache:
        cached = _read_cached_samples(cache_path).get("ARC-Easy", [])
        if len(cached) >= limit:
            logger.info("Loaded %d ARC-Easy samples from local cache (%s).", limit, cache_path)
            return cached[:limit]

    logger.info("Downloading %d ARC-Easy samples from Hugging Face Hub...", limit)
    rows = _fetch_hf_rows("allenai/ai2_arc", "ARC-Easy", "test", length=max(limit * 2, 20))
    samples: list[BenchmarkSample] = []
    for row in rows:
        if len(samples) >= limit:
            break
        q_text = str(row.get("question", "")).strip()
        choices = row.get("choices") or {}
        labels = choices.get("label") or []
        texts = choices.get("text") or []
        answer_key = str(row.get("answerKey", "")).strip().upper()
        if not q_text or not labels or not texts or not answer_key:
            continue

        options_str = "\n".join(f"{lbl}) {txt}" for lbl, txt in zip(labels, texts))
        prompt = (
            f"Question: {q_text}\n\n"
            f"Options:\n{options_str}\n\n"
            "Provide a brief 1-sentence answer and end with `FINAL_OPTION: <label>` (e.g. `FINAL_OPTION: A`)."
        )
        samples.append(
            BenchmarkSample(
                sample_id=f"arc_easy_{row.get('id', len(samples))}",
                dataset="ARC-Easy",
                expected_complexity="simple",
                expected_sub_agent="fast_qa_agent",
                prompt=prompt,
                ground_truth=answer_key,
                source_url="https://huggingface.co/datasets/allenai/ai2_arc",
            )
        )
    return samples


def load_math500_samples(
    limit: int = 15,
    cache_path: Path = DEFAULT_CACHE_PATH,
    refresh_cache: bool = False,
) -> list[BenchmarkSample]:
    """Loads real Level-5 competition math samples from local cache or `HuggingFaceH4/MATH-500` (`test` split)."""
    if not refresh_cache:
        cached = _read_cached_samples(cache_path).get("MATH-500", [])
        if len(cached) >= limit:
            logger.info("Loaded %d MATH-500 samples from local cache (%s).", limit, cache_path)
            return cached[:limit]

    logger.info("Downloading %d MATH-500 Level-5 samples from Hugging Face Hub...", limit)
    samples: list[BenchmarkSample] = []
    for offset in (0, 100, 200, 300):
        if len(samples) >= limit:
            break
        rows = _fetch_hf_rows("HuggingFaceH4/MATH-500", "default", "test", length=100, offset=offset)
        for row in rows:
            if len(samples) >= limit:
                break
            level = row.get("level")
            problem = str(row.get("problem", "")).strip()
            answer = str(row.get("answer", "")).strip()
            unique_id = str(row.get("unique_id", f"idx_{len(samples)}")).replace("/", "_").replace(".json", "")
            subject = str(row.get("subject", "Mathematics")).strip()

            # Filter for hardest Level 5 problems with clean exact numeric answers
            if level not in (5, "Level 5"):
                continue
            if not problem or not re.fullmatch(r"-?\d+(?:\.\d+)?", answer):
                continue

            prompt = (
                f"Solve this Level-5 competition mathematics problem ({subject}) step-by-step, "
                "showing all algebraic and analytical derivations, and conclude with "
                "`FINAL_ANSWER: <number>`:\n\n"
                f"{problem}"
            )
            samples.append(
                BenchmarkSample(
                    sample_id=f"math500_{unique_id}",
                    dataset="MATH-500",
                    expected_complexity="complex",
                    expected_sub_agent="math_analyst_agent",
                    prompt=prompt,
                    ground_truth=answer,
                    source_url="https://huggingface.co/datasets/HuggingFaceH4/MATH-500",
                )
            )
    return samples


def load_humaneval_samples(
    limit: int = 15,
    cache_path: Path = DEFAULT_CACHE_PATH,
    refresh_cache: bool = False,
) -> list[BenchmarkSample]:
    """Loads real algorithmic Python coding tasks from local cache or `openai/openai_humaneval` (`test` split)."""
    if not refresh_cache:
        cached = _read_cached_samples(cache_path).get("HumanEval", [])
        if len(cached) >= limit:
            logger.info("Loaded %d HumanEval samples from local cache (%s).", limit, cache_path)
            return cached[:limit]

    logger.info("Downloading %d HumanEval samples from Hugging Face Hub...", limit)
    # Offset 120..163 contains the harder algorithmic/DP/grid/parsing problems in HumanEval
    rows = _fetch_hf_rows("openai/openai_humaneval", "openai_humaneval", "test", length=max(limit + 10, 30), offset=120)
    samples: list[BenchmarkSample] = []
    for row in rows:
        if len(samples) >= limit:
            break
        task_id_raw = str(row.get("task_id", f"HumanEval/{120 + len(samples)}")).strip()
        task_slug = task_id_raw.lower().replace("/", "_")
        code_prompt = str(row.get("prompt", "")).strip()
        test_code = str(row.get("test", "")).strip()
        entry_point = str(row.get("entry_point", "")).strip()
        if not code_prompt or not test_code or not entry_point:
            continue

        sig = _extract_function_signature(code_prompt)
        prompt = (
            "Write a complete, self-contained Python implementation for the following function specification "
            "and explain the algorithm's time and space complexity.\n\n"
            f"Required function signature: `{sig}`\n\n"
            f"Specification and starter code:\n```python\n{code_prompt}\n```\n\n"
            "Enclose your complete Python implementation (including any needed standard library imports) "
            "inside a ```python ... ``` fenced block."
        )
        samples.append(
            BenchmarkSample(
                sample_id=task_slug,
                dataset="HumanEval",
                expected_complexity="complex",
                expected_sub_agent="code_engineer_agent",
                prompt=prompt,
                ground_truth=sig,
                test_assertions=[test_code, f"check({entry_point})"],
                test_imports=["from typing import *", "import math", "import collections", "import itertools", "import functools", "import re"],
                source_url="https://huggingface.co/datasets/openai/openai_humaneval",
            )
        )
    return samples


def load_benchmark_suite(
    samples_per_dataset: int = 15,
    cache_path: Path = DEFAULT_CACHE_PATH,
    refresh_cache: bool = False,
) -> list[BenchmarkSample]:
    """Loads the stratified open-source benchmark suite from local cache if available, otherwise downloads from Hugging Face Hub."""
    if not refresh_cache and cache_path.exists():
        by_ds = _read_cached_samples(cache_path)
        if all(len(by_ds.get(ds, [])) >= samples_per_dataset for ds in DATASET_NAMES):
            selected: list[BenchmarkSample] = []
            for ds in DATASET_NAMES:
                selected.extend(by_ds[ds][:samples_per_dataset])
            logger.info(
                "Using existing local data cache at %s (%d samples across %s); skipping download.",
                cache_path,
                len(selected),
                ", ".join(DATASET_NAMES),
            )
            return selected

    fetch_count = max(samples_per_dataset, 15)
    arc_samples = load_arc_easy_samples(limit=fetch_count, cache_path=cache_path, refresh_cache=refresh_cache)
    math_samples = load_math500_samples(limit=fetch_count, cache_path=cache_path, refresh_cache=refresh_cache)
    he_samples = load_humaneval_samples(limit=fetch_count, cache_path=cache_path, refresh_cache=refresh_cache)

    all_fetched = arc_samples + math_samples + he_samples
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    cache_path.write_text(
        json.dumps([asdict(s) for s in all_fetched], indent=2),
        encoding="utf-8",
    )
    logger.info("Saved %d benchmark samples to local cache at %s.", len(all_fetched), cache_path)

    return (
        arc_samples[:samples_per_dataset]
        + math_samples[:samples_per_dataset]
        + he_samples[:samples_per_dataset]
    )


# ---------------------------------------------------------------------------
# Deterministic Ground-Truth Verifiers
# ---------------------------------------------------------------------------

def verify_arc_easy(response_text: str, answer_key: str) -> tuple[bool, str]:
    """Checks whether the model selected the correct multiple-choice option label."""
    gold = answer_key.strip().upper()
    if not response_text.strip():
        return False, ""

    patterns = [
        r"FINAL_OPTION\s*:\s*\**([A-D1-4])\**",
        r"(?i)\b(?:correct\s+)?(?:option|answer|choice)\s*(?:is|:)\s*\**([A-D1-4])\b",
        r"\*\*([A-D1-4])\)",
        r"(?:^|\s)([A-D1-4])\)\s",
    ]
    for pat in patterns:
        matches = re.findall(pat, response_text)
        if matches:
            pred = matches[-1].strip().upper()
            return pred == gold, pred

    last_line = response_text.strip().splitlines()[-1]
    m = re.search(r"\b([A-D1-4])\b", last_line)
    if m:
        pred = m.group(1).upper()
        return pred == gold, pred
    return False, ""


def _parse_number(val: str) -> float | None:
    cleaned = val.replace(",", "").replace("$", "").strip()
    try:
        return float(cleaned)
    except ValueError:
        return None


def verify_math_numeric(response_text: str, ground_truth: str) -> tuple[bool, str]:
    """Checks whether the extracted numerical answer matches the MATH-500 / GSM8K ground truth."""
    gold_val = _parse_number(ground_truth)
    if gold_val is None or not response_text.strip():
        return False, ""

    patterns = [
        r"FINAL_ANSWER\s*:\s*\$?\**(-?[\d,]+(?:\.\d+)?)\**",
        r"\\boxed\{\s*(-?[\d,]+(?:\.\d+)?)\s*\}",
        r"####\s*\$?\**(-?[\d,]+(?:\.\d+)?)\**",
        r"(?i)(?:final\s+answer|answer\s+is)\s*[:=]?\s*\$?\**(-?[\d,]+(?:\.\d+)?)\**",
    ]
    for pat in patterns:
        matches = re.findall(pat, response_text)
        if matches:
            pred_str = matches[-1]
            pred_val = _parse_number(pred_str)
            if pred_val is not None:
                return abs(pred_val - gold_val) < 1e-3, pred_str

    all_nums = re.findall(r"-?\d[\d,]*(?:\.\d+)?", response_text)
    if all_nums:
        pred_str = all_nums[-1]
        pred_val = _parse_number(pred_str)
        if pred_val is not None:
            return abs(pred_val - gold_val) < 1e-3, pred_str
    return False, ""


def verify_code_assertions(
    response_text: str,
    test_assertions: list[str],
    test_imports: list[str] | None = None,
) -> tuple[bool, str]:
    """Extracts generated Python code and executes ground-truth unit test assertions."""
    if not response_text.strip():
        return False, "empty_response"

    code_blocks = re.findall(r"```(?:python)?\s*\n(.*?)```", response_text, flags=re.DOTALL | re.IGNORECASE)
    code = "\n\n".join(code_blocks).strip() if code_blocks else response_text.strip()

    imports_block = "\n".join(test_imports or [])
    assertions_block = "\n\n".join(test_assertions)
    test_script = f"{imports_block}\n\n{code}\n\n{assertions_block}\n"

    try:
        proc = subprocess.run(
            [sys.executable, "-I", "-c", test_script],
            capture_output=True,
            text=True,
            timeout=8,
            check=False,
        )
        if proc.returncode == 0:
            return True, "passed_all_assertions"
        err_lines = (proc.stderr or proc.stdout or "assertion_failed").strip().splitlines()
        short_err = err_lines[-1] if err_lines else "assertion_failed"
        return False, short_err[:120]
    except subprocess.TimeoutExpired:
        return False, "timeout_8s"
    except Exception as exc:
        return False, f"exec_error: {exc}"


def verify_sample(sample: BenchmarkSample, response_text: str) -> tuple[bool, str]:
    """Dispatches verification to the dataset-specific ground-truth checker."""
    if sample.dataset == "ARC-Easy":
        return verify_arc_easy(response_text, sample.ground_truth)
    if sample.dataset in ("MATH-500", "GSM8K"):
        return verify_math_numeric(response_text, sample.ground_truth)
    if sample.dataset in ("HumanEval", "MBPP"):
        return verify_code_assertions(response_text, sample.test_assertions, sample.test_imports)
    raise ValueError(f"Unknown dataset: {sample.dataset}")
