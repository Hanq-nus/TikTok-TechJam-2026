import json
import os
import re
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()  # reads .env in the current working directory into os.environ

# ---- Config ------------------------------------------------------------

ROOT = Path(__file__).parent
WORKDIR = ROOT / "workdir"
LOG_PATH = ROOT / "run_log.jsonl"
BEST_CODE_PATH = ROOT / "best_pipeline.py"

# Adjust if your data lives elsewhere. Passed to candidate scripts via env var.
DATA_DIR = ROOT / "KuaiRand-Pure" / "data"

MAX_ITERATIONS = 50
WALL_CLOCK_LIMIT_SEC = 6 * 60 * 60
CONVERGENCE_EPS = 0.002
CONVERGENCE_N = 3
PER_ITERATION_TIMEOUT_SEC = 5 * 60  # FM baseline is ~40s; kill slow/hung candidates fast to save budget

# Fill in MODEL once you've checked GET /v1/models for what's available.
MODEL = "qwen3-coder-next"
API_BASE_URL = os.environ.get("SOCLAAS_BASE_URL")
API_KEY = os.environ.get("SOCLAAS_API_KEY")

client = OpenAI(api_key=API_KEY, base_url=API_BASE_URL)

RESULT_MARKER = "RESULT_JSON:"


@dataclass
class Iteration:
    index: int
    hypothesis: str
    code_path: Path
    metrics: dict = field(default_factory=dict)
    error: Optional[str] = None
    duration_sec: float = 0.0


# ---- Bootstrap ------------------------------------------------------

BOOTSTRAP_FOOTER = '''

# --- Appended by agent_loop.py, not part of the original baseline.py ---
if __name__ == "__main__":
    import json, os
    from data import load
    from _agent_utils import to_native
    splits = load(os.environ["KUAIRAND_DATA_DIR"])
    res = run_fm(splits, k=16, lr=0.001, epochs=40, seed=0, verbose=False)
    print("RESULT_JSON:" + json.dumps(to_native(res)))
'''


def build_bootstrap_code() -> str:
    """Build best_pipeline.py by reusing baseline.py's ACTUAL file content
    verbatim (FM class, run_fm, etc.) rather than a hand-copied duplicate --
    this can never drift from the real file, unlike a manually retyped copy.
    Only baseline.py's own CLI entry point (argparse, its own __main__ block)
    is dropped, since we append our own minimal runner instead."""
    src = (WORKDIR / "baseline.py").read_text()
    marker = "if __name__ == '__main__':"
    if marker in src:
        src = src.split(marker)[0]
    return src + BOOTSTRAP_FOOTER


AGENT_UTILS_CODE = '''"""Fixed helper module, not LLM-generated. Available to every candidate
script as `from _agent_utils import to_native`."""

def to_native(x):
    """Recursively cast numpy scalars to native Python types for JSON."""
    if isinstance(x, dict):
        return {k: to_native(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [to_native(v) for v in x]
    if hasattr(x, "item"):  # numpy scalar (float32, int64, etc.)
        return x.item()
    return x
'''


def bootstrap_pipeline():
    """Always overwrite _agent_utils.py and best_pipeline.py with the known-
    good starting point at the start of a run — a fresh run should never
    inherit a previous run's best_pipeline.py, even if one is on disk."""
    (WORKDIR / "_agent_utils.py").write_text(AGENT_UTILS_CODE)
    BEST_CODE_PATH.write_text(build_bootstrap_code())


def load_current_pipeline_code() -> str:
    return BEST_CODE_PATH.read_text()


def profile_dataset() -> str:
    """Run once, using the real loader/encoder — not the LLM's guesswork —
    to extract concrete facts (dtypes, shapes, types, class balance) about
    the actual data. Past iterations kept repeating the same bugs despite
    prose warnings in the system prompt (e.g. treating `users` as a numpy
    array when it's a plain list); grounding the prompt in real, computed
    facts is more reliable than asking the model to remember prose rules."""
    sys.path.insert(0, str(WORKDIR))
    import importlib
    data_mod = importlib.import_module("data")
    splits = data_mod.load(str(DATA_DIR.resolve()))
    enc, dim = data_mod.encode(splits)
    lines = [f"dim (total feature vocab size, sum of all field ranges) = {dim}",
             f"FIELDS (column order in X) = {data_mod.FIELDS}"]
    for name in ("train", "valid", "test"):
        X, y, users = enc[name]
        lines.append(
            f"{name}: X.shape={X.shape} X.dtype={X.dtype} | "
            f"y.shape={y.shape} y.dtype={y.dtype} positive_rate={float(y.mean()):.4f} | "
            f"users: type={type(users).__name__} (NOT a numpy array — wrap with "
            f"np.asarray(users) before any fancy/array indexing) len={len(users)} "
            f"sample_first_3={users[:3]}"
        )
    return "\n".join(lines)


# ---- Execute a candidate pipeline ----------------------------------------

def run_pipeline(code: str) -> dict:
    """Write `code` into workdir, execute it, return {'valid': {...}, 'test': {...}}.
    Raises on failure, timeout, or a missing/malformed RESULT_JSON line —
    the caller logs these as robustness events and moves on."""
    scratch = WORKDIR / "candidate_pipeline.py"
    scratch.write_text(code)

    env = dict(os.environ)
    env["KUAIRAND_DATA_DIR"] = str(DATA_DIR.resolve())

    result = subprocess.run(
        [sys.executable, "candidate_pipeline.py"],
        cwd=WORKDIR,
        capture_output=True,
        text=True,
        timeout=PER_ITERATION_TIMEOUT_SEC,
        env=env,
    )
    if result.returncode != 0:
        raise RuntimeError(result.stderr[-4000:])

    marker_lines = [l for l in result.stdout.splitlines() if l.startswith(RESULT_MARKER)]
    if not marker_lines:
        raise RuntimeError(
            "No RESULT_JSON: line in stdout. Last 2000 chars of stdout:\n"
            + result.stdout[-2000:]
        )
    payload = marker_lines[-1][len(RESULT_MARKER):]
    res = json.loads(payload)
    if "valid" not in res or "primary" not in res["valid"]:
        raise RuntimeError(f"RESULT_JSON missing valid/primary: {payload[:500]}")
    return res


# ---- LLM proposal step ----------------------------------------------------

SYSTEM_PROMPT = """You are an autonomous ML research agent competing on the \
KuaiRand-Pure recommender-systems benchmark. Task: within-user ranking over \
logged impressions. Label: long_view (0/1). Metrics: GAUC and nDCG@5, \
primary = mean of both. You are trying to beat the official FM baseline \
(test primary 0.5946) as far as possible, scored on validation-primary \
convergence then a one-time hidden-test evaluation.

FIXED, DO NOT MODIFY — available via local imports in your candidate script:
- `from data import load, encode, FIELDS` — FIELDS = ['user_id','video_id',
  'author_id','tab','dur_bucket']. load(data_dir) returns
  {'train':[...], 'valid':[...], 'test':[...]} of raw tuples
  (date, user_id, video_id, author_id, tab, duration_ms, label).
  encode(splits) returns (enc, dim) where enc[name] = (X, y, users),
  X is int32 (N, len(FIELDS)) of encoded categorical ids, y is float32 labels.
  ⚠️ `users` is a PLAIN PYTHON LIST, not a numpy array. Fancy/array indexing
  like `users[idx]` where idx is an array will raise
  "TypeError: only integer scalar arrays can be converted to a scalar index".
  If you need to index, shuffle, or batch it like an array, first convert it:
  `users_arr = np.asarray(users)`. This exact bug has broken the majority of
  past iterations that touched user grouping/session logic — check for it
  before submitting any candidate that indexes `users`.
- `from evaluate import evaluate` — evaluate(user_ids, labels, scores, k=5)
  -> {'GAUC':.., 'nDCG@5':.., 'primary':.., 'users':.., 'rows':..}. This is
  the ONLY scoring function that counts — never reimplement it.
- Data directory is at os.environ["KUAIRAND_DATA_DIR"]; call load(that path).
- `from _agent_utils import to_native` — recursively casts numpy scalars to
  native Python types; use before any json.dumps() of your results dict.
- Only numpy + Python stdlib are available (no torch/pandas/sklearn).
- Target Python 3.9. Do NOT use 3.10+ syntax such as `X | None` union
  types or `match` statements — use `Optional[X]` from `typing` instead,
  and stick to if/elif chains. This will crash the run if violated.

CONFIRMED DEAD ENDS — already tested by the organizers, do not retry:
1. Adding more static features (music_id, video_type, upload_type, coarse
   user buckets) — no measurable gain (0.5940 vs 0.5950, within noise).
2. Increasing FM embedding dim k (8/16/32) — barely moves the score
   (0.5895/0.5902/0.5887). user_id x video_id crossing already captures
   most learnable signal; 1.14M training rows can't support more capacity.
3. Structural fact: pure user-side features contribute nothing to ranking,
   since ranking is within-user and a constant-per-user term can't change
   intra-user order. User features only help via interaction with item-side
   features.

RANKED UNEXPLORED DIRECTIONS (organizer's own priority guess — not proven,
you should still verify each empirically against validation primary):
1. Loss function mismatch: training currently optimizes pointwise logloss
   but scoring is rank-based (GAUC/nDCG). Try pairwise (BPR-style) or
   listwise (softmax over one user's session) loss instead. Organizer's
   top guess for where headroom is.
2. User history/sequence modeling — completely unused currently, despite
   each user having hundreds-to-thousands of train interactions.
3. Multi-task learning using is_click/is_like/is_follow/is_comment/
   is_forward/play_time_ms as auxiliary signals alongside long_view.
4. Watch-time modeling (censored regression) — research-depth, lower priority.
5. Different architecture (DeepFM/DCN/xDeepFM) — lower priority since
   capacity was empirically not the bottleneck.
6. Time features / date and train-test distribution drift.
7. Use log_random_*.csv (unbiased random-exposure log) as an extra
   validation set to check for overfitting to biased traffic.

Your candidate script MUST:
- Be a complete, standalone Python file using only numpy + stdlib.
- Import load/encode/evaluate as shown above; call load(os.environ[...]).
- START FROM THE "Current best pipeline code" SHOWN BELOW, EDITED IN PLACE.
  It is the real, working FM implementation (class FM + run_fm), not a
  stub. Make the smallest change that tests your hypothesis -- e.g. edit
  the loss/gradient inside FM.step, or the feature construction before
  encode() -- rather than rewriting training from scratch. Rewriting from
  scratch is the single biggest cause of past iteration failures here:
  wrong assumptions about encode()'s return shape, broadcasting errors in
  the interaction term, and one iteration that accidentally redefined a
  function with the same name as an existing one and recursed infinitely.
- Never define a function with the same name as one you also call
  unchanged -- that causes infinite recursion, not delegation.
- As its LAST line of stdout, print exactly:
  RESULT_JSON:<json with {"valid": {...evaluate() dict...}, "test": {...}}>
- evaluate() returns numpy scalars (e.g. float32), which json.dumps() cannot
  serialize directly. A `_agent_utils.py` module is available in the same
  directory — use `from _agent_utils import to_native` and wrap your result
  dict as `json.dumps(to_native(res))`. Do not reimplement this yourself.
- Complete within a few minutes on a single CPU core (the FM baseline is ~40s).
  Avoid pure-Python loops over the full 1.14M-row training set per epoch --
  vectorize with numpy, the way the existing FM.step() does.

Respond in EXACTLY this plain-text format, no JSON, nothing else outside it:

HYPOTHESIS: <one or two sentences -- what you're changing and why, tied to
one of the ranked directions above or your own reasoning>

```python
<the full candidate script as described above>
```

Do not wrap the whole response in JSON -- source code inside a JSON string
is error-prone (unescaped quotes/newlines routinely break JSON parsing).
Only the code itself goes inside the ```python fence; everything else is
plain text."""


def propose_next_iteration(current_code: str, history: list[Iteration], data_profile: str) -> tuple[str, str]:
    history_summary = "\n".join(
        f"Iter {h.index}: {h.hypothesis} -> "
        + (f"valid.primary={h.metrics.get('valid', {}).get('primary')}" if not h.error
           else f"ERROR: {h.error[:300]}")
        for h in history[-6:]
    )
    user_msg = f"""GROUND-TRUTH DATA PROFILE (computed directly from the real data, not a description —
trust this over any assumption you'd otherwise make):
{data_profile}

Current best pipeline code:
```python
{current_code}
```

Recent iteration history:
{history_summary or '(none yet)'}

Propose the next single focused change."""

    resp = client.chat.completions.create(
        model=MODEL,
        max_tokens=6000,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_msg},
        ],
    )
    text = resp.choices[0].message.content.strip()

    hyp_match = re.search(r"HYPOTHESIS:\s*(.+?)(?=\n```|\Z)", text, re.DOTALL)
    code_match = re.search(r"```(?:python)?\s*\n(.*?)```", text, re.DOTALL)

    if not hyp_match or not code_match:
        raise ValueError(
            "Could not parse LLM response into hypothesis + code "
            f"(hypothesis_found={bool(hyp_match)}, code_found={bool(code_match)}). "
            f"First 500 chars of raw response:\n{text[:500]}"
        )

    hypothesis = hyp_match.group(1).strip()
    candidate_code = code_match.group(1)
    return hypothesis, candidate_code


# ---- Convergence ----------------------------------------------------------

def has_converged(history: list[Iteration]) -> bool:
    scored = [h for h in history if h.error is None]
    if len(scored) < CONVERGENCE_N + 1:
        return False
    recent = scored[-(CONVERGENCE_N + 1):]
    deltas = [
        recent[i + 1].metrics["valid"]["primary"] - recent[i].metrics["valid"]["primary"]
        for i in range(len(recent) - 1)
    ]
    return all(abs(d) <= CONVERGENCE_EPS for d in deltas)


# ---- Main loop --------------------------------------------------------------

def main():
    WORKDIR.mkdir(exist_ok=True)
    bootstrap_pipeline()
    LOG_PATH.write_text("")  # fresh run log every time the script is run
    history: list[Iteration] = []
    start_time = time.time()

    print("Profiling dataset (one-time, grounds the LLM in real facts)...")
    data_profile = profile_dataset()
    print(data_profile)

    # Score iteration 0 (the deterministic baseline) first, before any LLM call.
    print("Scoring bootstrap baseline (iteration 0)...")
    t0 = time.time()
    baseline_res = run_pipeline(load_current_pipeline_code())
    iteration0 = Iteration(
        index=0, hypothesis="Bootstrap: official FM baseline, unmodified.",
        code_path=BEST_CODE_PATH, metrics=baseline_res, duration_sec=time.time() - t0,
    )
    history.append(iteration0)
    best_valid_primary = baseline_res["valid"]["primary"]
    print(f"  baseline valid.primary={best_valid_primary:.4f} test.primary={baseline_res['test']['primary']:.4f}")
    with LOG_PATH.open("a") as f:
        f.write(json.dumps({
            "index": 0, "hypothesis": iteration0.hypothesis,
            "metrics": iteration0.metrics, "error": None,
            "duration_sec": iteration0.duration_sec,
        }) + "\n")

    for i in range(1, MAX_ITERATIONS):
        if time.time() - start_time > WALL_CLOCK_LIMIT_SEC:
            print(f"Wall-clock limit hit at iteration {i}. Stopping.")
            break

        current_code = load_current_pipeline_code()

        t0 = time.time()
        try:
            hypothesis, candidate_code = propose_next_iteration(current_code, history, data_profile)
        except Exception as e:  # noqa: BLE001 -- a bad/malformed LLM response must not kill the run
            iteration = Iteration(
                index=i, hypothesis="(LLM proposal step failed)",
                code_path=WORKDIR / "candidate_pipeline.py",
                error=f"propose_next_iteration failed: {e}",
                duration_sec=time.time() - t0,
            )
            print(f"[iter {i}] PROPOSAL ERROR: {e}")
            history.append(iteration)
            with LOG_PATH.open("a") as f:
                f.write(json.dumps({
                    "index": iteration.index, "hypothesis": iteration.hypothesis,
                    "metrics": iteration.metrics, "error": iteration.error,
                    "duration_sec": iteration.duration_sec,
                }) + "\n")
            continue

        iteration = Iteration(index=i, hypothesis=hypothesis, code_path=WORKDIR / "candidate_pipeline.py")
        try:
            res = run_pipeline(candidate_code)
            iteration.metrics = res
            iteration.duration_sec = time.time() - t0
            vp = res["valid"]["primary"]

            if vp > best_valid_primary:
                best_valid_primary = vp
                BEST_CODE_PATH.write_text(candidate_code)
                print(f"[iter {i}] NEW BEST valid.primary={vp:.4f} test.primary={res['test']['primary']:.4f} -- {hypothesis}")
            else:
                print(f"[iter {i}] valid.primary={vp:.4f} (best={best_valid_primary:.4f}) -- {hypothesis}")

        except Exception as e:  # noqa: BLE001 -- log & continue, this IS the robustness story
            iteration.error = str(e)
            iteration.duration_sec = time.time() - t0
            print(f"[iter {i}] ERROR: {e}")

        history.append(iteration)
        with LOG_PATH.open("a") as f:
            f.write(json.dumps({
                "index": iteration.index,
                "hypothesis": iteration.hypothesis,
                "metrics": iteration.metrics,
                "error": iteration.error,
                "duration_sec": iteration.duration_sec,
            }) + "\n")

        if has_converged(history):
            print(f"Converged after {i + 1} iterations (incl. bootstrap).")
            break

    print(f"\nDone. Best valid.primary: {best_valid_primary:.4f}")
    print(f"Best pipeline saved at: {BEST_CODE_PATH}")
    print(f"Full run log at: {LOG_PATH}")
    print()
    print_run_summary()


def print_run_summary():
    """Pretty-print run_log.jsonl as a readable table. Can also be called
    standalone (python3 -c "from agent_loop import print_run_summary; print_run_summary()")
    to re-view a completed run's log without rerunning the agent."""
    if not LOG_PATH.exists():
        print(f"No log found at {LOG_PATH}")
        return

    iters = []
    with LOG_PATH.open() as f:
        for line in f:
            line = line.strip()
            if line:
                iters.append(json.loads(line))

    def fmt_primary(entry, split):
        v = (entry.get("metrics") or {}).get(split, {}).get("primary")
        return f"{v:.4f}" if v is not None else "  --  "

    baseline_valid = None
    best_idx, best_valid = None, -1.0

    print(f"{'#':>3} {'status':<8} {'valid':>8} {'test':>8} {'Δvalid':>8} {'time':>7}  hypothesis")
    print("-" * 100)

    for it in iters:
        idx = it["index"]
        err = it.get("error")
        status = "OK" if not err else "ERROR"
        valid_p = (it.get("metrics") or {}).get("valid", {}).get("primary")
        dur = it.get("duration_sec", 0)

        if idx == 0 and valid_p is not None:
            baseline_valid = valid_p
        if valid_p is not None and valid_p > best_valid:
            best_valid, best_idx = valid_p, idx

        delta = ""
        if valid_p is not None and baseline_valid is not None:
            delta = f"{valid_p - baseline_valid:+.4f}"

        hyp = it.get("hypothesis", "")
        hyp_short = (hyp[:75] + "...") if len(hyp) > 78 else hyp
        if err:
            err_short = err.strip().splitlines()[-1][:75]
            hyp_short = f"{hyp_short}\n{'':>39}⤷ {err_short}"

        print(f"{idx:>3} {status:<8} {fmt_primary(it,'valid'):>8} {fmt_primary(it,'test'):>8} "
              f"{delta:>8} {dur:>6.1f}s  {hyp_short}")

    print("-" * 100)
    n_ok = sum(1 for it in iters if not it.get("error"))
    print(f"{len(iters)} iterations: {n_ok} succeeded, {len(iters) - n_ok} errored.")
    if best_idx is not None:
        print(f"Best so far: iteration {best_idx} (valid primary {best_valid:.4f}, "
              f"baseline was {baseline_valid:.4f})")


if __name__ == "__main__":
    main()