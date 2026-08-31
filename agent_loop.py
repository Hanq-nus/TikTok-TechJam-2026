import difflib
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
LOG_PATH = ROOT / "run_log.jsonl"   # the CURRENT run only (overwritten each run)
RUNS_DIR = ROOT / "runs"            # permanent archive: runs/run_log_<n>_<timestamp>.jsonl
BEST_CODE_PATH = ROOT / "best_pipeline.py"
BEST_TEST_SCORES_PATH = ROOT / "best_test_scores.npy"
SUBMISSION_PATH = ROOT / "submission.csv"

# Adjust if your data lives elsewhere. Passed to candidate scripts via env var.
DATA_DIR = ROOT / "KuaiRand-Pure" / "data"

MAX_ITERATIONS = 50
WALL_CLOCK_LIMIT_SEC = 6 * 60 * 60
CONVERGENCE_EPS = 0.002
CONVERGENCE_N = 3
PER_ITERATION_TIMEOUT_SEC = 5 * 60  # FM baseline is ~40s; kill slow/hung candidates fast to save budget

# Tunable heuristic, NOT a hard scientific cutoff. The honest FM baseline's
# validation-minus-test primary gap is ~0.006. A candidate whose gap is several
# times that has almost certainly computed a statistic / bucket edge / vocabulary
# using valid or test rows (leakage), inflating validation while test stays flat.
# Candidates exceeding this gap are NOT accepted as the new best (see main()).
# Raise or lower it if the guard fires too often or never.
LEAKAGE_GAP_THRESHOLD = 0.02

# Fill in MODEL once you've checked GET /v1/models for what's available.
MODEL = "qwen3-coder-next"
API_BASE_URL = os.environ.get("SOCLAAS_BASE_URL")
API_KEY = os.environ.get("SOCLAAS_API_KEY")

client = OpenAI(api_key=API_KEY, base_url=API_BASE_URL)

RESULT_MARKER = "RESULT_JSON:"

# Filled in by main() at the start of each run. Every run_log line carries these
# so a timestamped archive file under runs/ is self-identifying.
_RUN_META: dict = {}

# Running LLM token totals for the whole run (deliverable: Results Summary needs
# total input+output tokens). Reset by main(); accumulated in
# propose_next_iteration() right after each API call, so a later parse failure
# in the same call still counts. `calls_without_usage` > 0 means the endpoint
# omitted usage on some calls and the totals are a lower bound.
_TOKEN_TOTALS: dict = {"prompt": 0, "completion": 0, "calls": 0, "calls_without_usage": 0}


@dataclass
class Iteration:
    index: int
    hypothesis: str
    code_path: Path
    metrics: dict = field(default_factory=dict)
    error: Optional[str] = None
    # Deterministic one-line verdict for this iteration, derived by our own code
    # (never an extra LLM call). Holds the "REJECTED: suspected leakage, ..."
    # string when the leakage guard fires, otherwise a summary such as
    # "Improved valid primary by +0.0025 -- new best 0.6040" / "Failed: <error>"
    # / "No improvement, within noise (...)". See summarize_evaluation().
    evaluation: Optional[str] = None
    # Source-attribution fields parsed from the LLM's plain-text response
    # (same mechanism as `hypothesis`): a short stable direction label used by
    # the anti-repetition guardrail, where the idea came from, and notes handed
    # to the next iteration.
    direction: Optional[str] = None
    source: Optional[str] = None
    next_notes: Optional[str] = None
    # unified_diff(best_pipeline.py -> this candidate), joined into one string.
    code_diff: Optional[str] = None
    # LLM token usage for this iteration's proposal call (None if the endpoint
    # omitted usage, or if the call failed before usage was captured).
    prompt_tokens: Optional[int] = None
    completion_tokens: Optional[int] = None
    duration_sec: float = 0.0


# ---- Bootstrap ------------------------------------------------------

BOOTSTRAP_FOOTER = '''

# --- Appended by agent_loop.py, not part of the original baseline.py ---
if __name__ == "__main__":
    import json, os
    from data import load
    from _agent_utils import to_native, save_test_scores
    splits = load(os.environ["KUAIRAND_DATA_DIR"])
    res = run_fm(splits, k=16, lr=0.001, epochs=40, seed=0, verbose=False)
    # Raw test predictions are a big ndarray -- not JSON-safe even through
    # to_native() -- so pop them out and persist them separately.
    save_test_scores(res.pop("_test_scores"))
    print("RESULT_JSON:" + json.dumps(to_native(res)))
'''

# The one deliberate additive deviation from baseline.py: run_fm() as shipped
# returns only evaluate() dicts, discarding the raw test-split scores we need
# to build submission.csv. Patch exactly that return block -- everything else
# stays byte-for-byte identical to the real file.
_BASELINE_RETURN_SRC = """    return {'valid': evaluate(uva, yva, m.predict(Xva)),
            'test':  evaluate(ute, yte, m.predict(Xte))}"""

_BASELINE_RETURN_PATCHED = """    _test_scores = m.predict(Xte)
    return {'valid': evaluate(uva, yva, m.predict(Xva)),
            'test':  evaluate(ute, yte, _test_scores),
            '_test_scores': _test_scores}"""


def build_bootstrap_code() -> str:
    """Build best_pipeline.py by reusing baseline.py's ACTUAL file content
    verbatim (FM class, run_fm, etc.) rather than a hand-copied duplicate --
    this can never drift from the real file, unlike a manually retyped copy.
    Only baseline.py's own CLI entry point (argparse, its own __main__ block)
    is dropped, since we append our own minimal runner instead, plus one
    targeted patch to run_fm's return so raw test scores are exposed."""
    src = (WORKDIR / "baseline.py").read_text()
    marker = "if __name__ == '__main__':"
    if marker in src:
        src = src.split(marker)[0]

    if src.count(_BASELINE_RETURN_SRC) != 1:
        raise RuntimeError(
            "Could not patch run_fm's return block in workdir/baseline.py: expected "
            f"exactly one occurrence of the known return statement, found "
            f"{src.count(_BASELINE_RETURN_SRC)}. baseline.py has changed shape -- "
            "update _BASELINE_RETURN_SRC to match it before rerunning."
        )
    src = src.replace(_BASELINE_RETURN_SRC, _BASELINE_RETURN_PATCHED)
    return src + BOOTSTRAP_FOOTER


AGENT_UTILS_CODE = '''"""Fixed helper module, not LLM-generated. Available to every candidate
script as `from _agent_utils import to_native, save_test_scores`."""
import numpy as np

def to_native(x):
    """Recursively cast numpy scalars to native Python types for JSON."""
    if isinstance(x, dict):
        return {k: to_native(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [to_native(v) for v in x]
    if hasattr(x, "item"):  # numpy scalar (float32, int64, etc.)
        return x.item()
    return x

def save_test_scores(scores):
    """Save this script's raw test-split prediction scores, in the same row
    order as data.load(...)['test'], so the winning iteration's scores can
    be turned into submission.csv afterward. Call this once, right before
    printing RESULT_JSON, passing the EXACT SAME array you passed as
    `scores` when calling evaluate(...) on the test split."""
    np.save("test_scores.npy", np.asarray(scores, dtype=np.float64))
'''


def sync_submit_module():
    """Mirror the kit's own submit.py into workdir/ so write_submission_csv()
    can import its write_submission() directly instead of reimplementing the
    submission CSV format. Copied verbatim — submit.py is a fixed kit file."""
    src = ROOT / "submit.py"
    if not src.exists():
        return False
    (WORKDIR / "submit.py").write_bytes(src.read_bytes())
    return True


def bootstrap_pipeline():
    """Always overwrite _agent_utils.py and best_pipeline.py with the known-
    good starting point at the start of a run — a fresh run should never
    inherit a previous run's best_pipeline.py, even if one is on disk."""
    (WORKDIR / "_agent_utils.py").write_text(AGENT_UTILS_CODE)
    BEST_CODE_PATH.write_text(build_bootstrap_code())
    sync_submit_module()


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

    # Clear any previous iteration's scores first: if this candidate fails to
    # write its own, we must NOT silently promote a stale file belonging to a
    # different model as if it were this iteration's output.
    stale_scores = WORKDIR / "test_scores.npy"
    if stale_scores.exists():
        stale_scores.unlink()

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


def capture_best_test_scores(iter_index: int):
    """Promote the iteration that just ran — now the best-scoring one — from
    workdir/test_scores.npy to the persistent ROOT/best_test_scores.npy, so a
    later iteration overwriting workdir can't destroy the winner's scores.
    Plain file-byte copy, so numpy/shutil stay out of this module's imports."""
    scores_path = WORKDIR / "test_scores.npy"
    if not scores_path.exists():
        print(f"[iter {iter_index}] WARNING: no test_scores.npy produced -- this "
              f"iteration's scores won't be available for submission generation.")
        return
    BEST_TEST_SCORES_PATH.write_bytes(scores_path.read_bytes())


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
- `from _agent_utils import to_native, save_test_scores` — to_native()
  recursively casts numpy scalars to native Python types (use before any
  json.dumps() of your results dict); save_test_scores(scores) persists your
  raw test-split scores for submission generation (see the MUST list below).
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
- SAVE TEST-SPLIT SCORES FOR SUBMISSION — MANDATORY, EVERY ITERATION.
  Immediately before printing the RESULT_JSON line, call
  `save_test_scores(scores_test)` (imported from _agent_utils, as shown
  above). `scores_test` MUST be THE EXACT SAME array you passed as the
  `scores` argument to evaluate(...) for the test split — not a
  recomputation, not the valid-split scores, not a rescaled copy. It is a
  1D array of raw scores (any real numbers, higher = more relevant) with
  one entry for EVERY row of the test split, in the exact same order as
  encode(splits)['test'] (which matches splits['test']'s row order).
  Concretely, structure it like this:

      scores_test = model.predict(Xte)          # your test-split scores
      res = {'valid': evaluate(uva, yva, scores_valid),
             'test':  evaluate(ute, yte, scores_test)}
      save_test_scores(scores_test)             # same array as above
      print("RESULT_JSON:" + json.dumps(to_native(res)))

  Do NOT put the score array inside the results dict — a large ndarray is
  not JSON-serializable and will crash json.dumps(). Do this unconditionally
  on every iteration, even though it does not affect your printed metrics:
  it is what turns the winning iteration into an actual competition
  submission file afterward, and an iteration that skips it cannot be
  submitted even if it scores best.
- VERIFY IMPORTS BEFORE FINISHING. Missing imports (os, json, numpy as np,
  to_native, evaluate, load, encode) have been the single most common cause
  of past iteration failures. Before finalizing, check every name you use
  is actually imported at the top of the script.
- NEVER LEAK VALID/TEST DATA INTO STATISTICS, THRESHOLDS, OR ENCODING.
  Any statistic, threshold, quantile/bucket edge, vocabulary, id->index map,
  normalization constant, or lookup table that your candidate COMPUTES and
  then applies to rows MUST be derived from splits['train'] ONLY. Never
  compute one of these from splits['valid'] or splits['test'], from any
  concatenation that includes them (e.g. train+valid, or all three), or
  indirectly (e.g. an expanding per-user accumulator that is updated while
  looping over valid/test rows, so each valid row sees other valid rows).
  The official encode() in data.py already does this correctly for the base
  FIELDS: it fits every field's vocabulary from splits['train'] alone, and
  any category first seen in valid/test maps to a shared UNK slot. ANY new
  feature you add MUST follow that identical pattern -- fit on train, apply
  to valid/test, route unseen values to UNK.
  Concrete examples of LEAKAGE (all forbidden):
    * item or user popularity / long-view rate computed with a Counter over
      splits['train'] + splits['valid'] + splits['test'] (or train+valid)
      instead of splits['train'] only;
    * np.quantile(values, q) to pick bucket edges where `values` spans the
      full dataset (or train+valid) rather than just the train split;
    * building a new categorical feature's id->index map from the union of
      all three splits, so valid/test-only categories get their own index
      instead of falling into UNK;
    * a running / time-decayed / expanding per-user statistic whose
      accumulator keeps updating as you iterate over valid or test rows.
  A candidate whose validation primary is far above its test primary is
  almost always leaking. The loop now AUTO-REJECTS any candidate whose
  (valid primary - test primary) gap exceeds ~0.02 -- the honest baseline
  gap is ~0.006 -- keeping the previous best instead. A leaky candidate
  therefore cannot win; it only wastes an iteration. Build the feature
  correctly (train-only) the first time.
- AVOID LOCAL-OPTIMUM COLLAPSE. If the recent iteration history below shows
  several consecutive attempts clustering within ~0.002 of each other and
  only tuning a numeric knob (temperature, clipping threshold, smoothing
  alpha, learning rate, etc. on the same underlying model), that's a sign
  you've plateaued on calibration, not found real headroom. In that case
  you MUST switch to a structurally different, not-yet-seriously-attempted
  direction from the ranked list above -- prioritize #2 (user history /
  sequence modeling) or #3 (multi-task learning), since those change what
  the model can represent rather than how its existing output is scaled.
- ANTI-REPETITION ACROSS DIRECTIONS (hard rule). Each history line below is
  tagged with the DIRECTION that iteration pursued. If the TWO most recent
  iterations THAT RAN WITHOUT ERROR share the same DIRECTION and NEITHER
  improved on the running best validation primary, you MUST choose a DIFFERENT
  DIRECTION this iteration. A direction that produced two clean, working runs
  without improving is demonstrably not paying off right now, however
  promising it seems in principle or however standard the advice is -- do not
  open your HYPOTHESIS with "since the top-ranked direction is ..." and return
  to it anyway. Iterations that CRASHED do not count toward this -- a bug is
  not evidence the idea is wrong, so a direction that only ever errored still
  deserves a genuine working attempt. If a line in this prompt says
  "ANTI-REPETITION RULE IS ACTIVE", that condition has already been detected
  for you and the named DIRECTION is forbidden this iteration. Your DIRECTION
  line is the on-record evidence of whether you followed this rule.

Respond in EXACTLY this plain-text format, no JSON, nothing else outside it:

DIRECTION: <one line, no prose: a short stable lowercase label for the
research direction this iteration pursues. Reuse one of these EXACT labels
when it fits: loss-mismatch, user-history, multi-task, watch-time,
architecture, time-features, unbiased-eval, calibration. Only invent a new
short hyphenated label if none fit. The anti-repetition rule above compares
these labels, so keep the label for a given approach identical across
iterations.>
SOURCE: <one line: where this specific idea came from -- e.g. "ranked
direction #2", "iter 14 NEXT_NOTES", "own reasoning: <a few words>",
"CONFIRMED DEAD END #3 rules out X so trying Y".>
HYPOTHESIS: <one or two sentences -- what you're changing and why, tied to
the DIRECTION above.>

```python
<the full candidate script as described above>
```

NEXT_NOTES: <one to three sentences addressed to the NEXT iteration: what
this change rules in or out, and the single most promising thing to try
next.>

Do not wrap the whole response in JSON -- source code inside a JSON string
is error-prone (unescaped quotes/newlines routinely break JSON parsing).
Only the code itself goes inside the ```python fence; everything else is
plain text. DIRECTION, SOURCE and HYPOTHESIS come before the fence,
NEXT_NOTES after it."""


HISTORY_RECENT_N = 8  # regular recency window shown to the LLM each iteration


def _history_for_prompt(history: list[Iteration], n_recent: int = HISTORY_RECENT_N) -> list[Iteration]:
    """The last `n_recent` iterations PLUS every iteration ever rejected for
    suspected leakage -- even if it has aged out of the recency window -- so the
    model keeps seeing a leaky pattern it already got caught on and stops
    repeating it. Deduplicated by index, sorted by index."""
    keep = {h.index: h for h in history[-n_recent:]}
    for h in history:
        if h.evaluation and "leakage" in h.evaluation.lower():
            keep[h.index] = h
    return [keep[i] for i in sorted(keep)]


def _history_line(h: Iteration) -> str:
    d = f"[DIRECTION: {h.direction}] " if h.direction else "[DIRECTION: ?] "
    if h.error:
        return f"Iter {h.index}: {d}{h.hypothesis} -> ERROR: {h.error[:200]}"
    vp = h.metrics.get("valid", {}).get("primary")
    tp = h.metrics.get("test", {}).get("primary")
    verdict = f" | {h.evaluation}" if h.evaluation else ""
    return f"Iter {h.index}: {d}{h.hypothesis} -> valid.primary={vp} test.primary={tp}{verdict}"


def _norm_direction(s: Optional[str]) -> str:
    """Loose-normalise a DIRECTION label for equality checks (compare intent,
    not exact spelling): lowercase, non-alphanumerics collapsed to hyphens."""
    return re.sub(r"[^a-z0-9]+", "-", (s or "").strip().lower()).strip("-")


def _anti_repetition_note(history: list[Iteration]) -> str:
    """If the two most recently EXECUTED iterations (error is None -- errored
    ones are skipped entirely) share a DIRECTION and neither became a new best
    (evaluation starts with 'Improved'), return a line telling the model that
    DIRECTION is off-limits this iteration. A direction that only ever crashed
    has not had a real attempt, so a bug alone must not rule it out."""
    executed = [h for h in history if h.error is None]
    if len(executed) < 2:
        return ""
    a, b = executed[-2], executed[-1]
    da, db = _norm_direction(a.direction), _norm_direction(b.direction)
    if not da or da != db:
        return ""
    if any(str(h.evaluation or "").startswith("Improved") for h in (a, b)):
        return ""
    return (
        f'\nANTI-REPETITION RULE IS ACTIVE: the last two EXECUTED attempts at '
        f'DIRECTION "{a.direction}" (iterations {a.index} and {b.index}) both ran '
        f'without improving on the best validation primary. You MUST pick a '
        f'DIRECTION whose normalised label differs from "{a.direction}" this '
        f'iteration.\n'
    )


def propose_next_iteration(current_code: str, history: list[Iteration], data_profile: str) -> tuple[dict, str]:
    history_summary = "\n".join(_history_line(h) for h in _history_for_prompt(history))
    user_msg = f"""GROUND-TRUTH DATA PROFILE (computed directly from the real data, not a description —
trust this over any assumption you'd otherwise make):
{data_profile}

Current best pipeline code:
```python
{current_code}
```

Recent iteration history:
{history_summary or '(none yet)'}
{_anti_repetition_note(history)}
Propose the next single focused change."""

    resp = client.chat.completions.create(
        model=MODEL,
        max_tokens=6000,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_msg},
        ],
    )

    # Capture token usage FIRST -- before parsing, which may raise -- so a
    # malformed response still counts toward the run total. Some OpenAI-compatible
    # endpoints omit `usage`; treat that as "unknown", not an error.
    usage = getattr(resp, "usage", None)
    prompt_tokens = getattr(usage, "prompt_tokens", None) if usage is not None else None
    completion_tokens = getattr(usage, "completion_tokens", None) if usage is not None else None
    _TOKEN_TOTALS["calls"] += 1
    if prompt_tokens is None and completion_tokens is None:
        _TOKEN_TOTALS["calls_without_usage"] += 1
    _TOKEN_TOTALS["prompt"] += prompt_tokens or 0
    _TOKEN_TOTALS["completion"] += completion_tokens or 0

    text = resp.choices[0].message.content.strip()

    def _field(label: str, others: list) -> Optional[str]:
        stop = "|".join([o + ":" for o in others] + [r"```"])
        m = re.search(rf"^{label}:\s*(.+?)(?=\n(?:{stop})|\Z)", text, re.DOTALL | re.MULTILINE)
        return m.group(1).strip() if m else None

    direction = _field("DIRECTION", ["SOURCE", "HYPOTHESIS", "NEXT_NOTES"])
    source = _field("SOURCE", ["DIRECTION", "HYPOTHESIS", "NEXT_NOTES"])
    hypothesis = _field("HYPOTHESIS", ["DIRECTION", "SOURCE", "NEXT_NOTES"])
    next_notes = _field("NEXT_NOTES", ["DIRECTION", "SOURCE", "HYPOTHESIS"])
    code_match = re.search(r"```(?:python)?\s*\n(.*?)```", text, re.DOTALL)

    # DIRECTION + HYPOTHESIS + code are required (DIRECTION drives the
    # anti-repetition guardrail); SOURCE / NEXT_NOTES are best-effort.
    if not hypothesis or not direction or not code_match:
        raise ValueError(
            "Could not parse LLM response into direction + hypothesis + code "
            f"(direction={bool(direction)}, hypothesis={bool(hypothesis)}, "
            f"code={bool(code_match)}). First 500 chars of raw response:\n{text[:500]}"
        )

    proposal = {
        "hypothesis": hypothesis,
        "direction": direction,
        "source": source or "(not provided)",
        "next_notes": next_notes or "(not provided)",
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
    }
    return proposal, code_match.group(1)


# ---- Convergence ----------------------------------------------------------

def has_converged(history: list[Iteration]) -> bool:
    # A candidate rejected for suspected leakage carries an inflated validation
    # primary and must not anchor the convergence test. (`evaluation` is now
    # populated for every iteration with a deterministic verdict, so test for
    # the REJECTED prefix specifically rather than for None -- same effect as
    # before for the leakage guard.)
    scored = [h for h in history
              if h.error is None and not str(h.evaluation or "").startswith("REJECTED")]
    if len(scored) < CONVERGENCE_N + 1:
        return False
    recent = scored[-(CONVERGENCE_N + 1):]
    deltas = [
        recent[i + 1].metrics["valid"]["primary"] - recent[i].metrics["valid"]["primary"]
        for i in range(len(recent) - 1)
    ]
    return all(abs(d) <= CONVERGENCE_EPS for d in deltas)


def summarize_evaluation(iteration: Iteration, best_before: float) -> str:
    """Deterministic one-line verdict from this iteration's own metrics/error
    plus the running best it was compared against -- no extra LLM call. NOT
    called when iteration.evaluation is already set (e.g. a leakage REJECTION),
    so that guard's string is never clobbered. 'Improved ...' is emitted iff the
    iteration became the new best, which is exactly what the anti-repetition
    guardrail keys on."""
    if iteration.error:
        return f"Failed: {iteration.error.strip().splitlines()[-1][:200]}"
    vp = iteration.metrics.get("valid", {}).get("primary")
    if vp is None:
        return "Failed: result had no valid primary"
    delta = vp - best_before
    if delta > 0:
        within = " (within noise)" if delta <= CONVERGENCE_EPS else ""
        return f"Improved valid primary by {delta:+.4f}{within} -- new best {vp:.4f}"
    if abs(delta) <= CONVERGENCE_EPS:
        return f"No improvement, within noise ({delta:+.4f} vs best {best_before:.4f})"
    return f"No improvement, regression ({delta:+.4f} vs best {best_before:.4f})"


def append_log(iteration: Iteration):
    """Single writer for the run log. Each line is appended to BOTH run_log.jsonl
    (the current run only, overwritten every run so the summary tooling keeps
    working unchanged) and the permanent per-run archive under runs/. Every line
    is stamped with the run identifier so an archived file stands alone."""
    line = json.dumps({
        "run_number": _RUN_META.get("run_number"),
        "run_id": _RUN_META.get("run_id"),
        "run_started": _RUN_META.get("run_started"),
        "index": iteration.index,
        "hypothesis": iteration.hypothesis,
        "direction": iteration.direction,
        "source": iteration.source,
        "metrics": iteration.metrics,
        "error": iteration.error,
        "evaluation": iteration.evaluation,
        "code_diff": iteration.code_diff,
        "next_notes": iteration.next_notes,
        "prompt_tokens": iteration.prompt_tokens,
        "completion_tokens": iteration.completion_tokens,
        "duration_sec": iteration.duration_sec,
    }) + "\n"
    with LOG_PATH.open("a") as f:
        f.write(line)
    archive = _RUN_META.get("archive_path")
    if archive is not None:
        with archive.open("a") as f:
            f.write(line)


# ---- Main loop --------------------------------------------------------------

def main():
    WORKDIR.mkdir(exist_ok=True)
    RUNS_DIR.mkdir(exist_ok=True)
    bootstrap_pipeline()

    # Identify this run. run_log.jsonl holds just this run (so print_run_summary
    # etc. keep working unchanged); a timestamped copy is also kept under runs/
    # so no past run is ever lost. run_number is "files already in runs/ + 1" --
    # best-effort ordering, the timestamp in the name is the real identifier.
    run_started = time.strftime("%Y-%m-%dT%H:%M:%S")
    run_id = time.strftime("%Y%m%dT%H%M%S")
    run_number = 1 + len(list(RUNS_DIR.glob("run_log_*.jsonl")))
    archive_path = RUNS_DIR / f"run_log_{run_number:03d}_{run_id}.jsonl"
    _RUN_META.clear()
    _RUN_META.update(run_number=run_number, run_id=run_id,
                     run_started=run_started, archive_path=archive_path)
    _TOKEN_TOTALS.update(prompt=0, completion=0, calls=0, calls_without_usage=0)
    LOG_PATH.write_text("")
    archive_path.write_text("")
    print(f"Run {run_number} (id {run_id}) -- archiving to {archive_path.relative_to(ROOT)}")

    history: list[Iteration] = []
    start_time = time.time()

    print("Profiling dataset (one-time, grounds the LLM in real facts)...")
    data_profile = profile_dataset()
    print(data_profile)

    # Score iteration 0 (the deterministic baseline) first, before any LLM call.
    print("Scoring bootstrap baseline (iteration 0)...")
    t0 = time.time()
    baseline_res = run_pipeline(load_current_pipeline_code())
    best_valid_primary = baseline_res["valid"]["primary"]
    iteration0 = Iteration(
        index=0, hypothesis="Bootstrap: official FM baseline, unmodified.",
        code_path=BEST_CODE_PATH, metrics=baseline_res, duration_sec=time.time() - t0,
        direction="baseline", source="bootstrap (no LLM call)",
        next_notes="Reference score; every later iteration is measured against this.",
        evaluation=f"Baseline established (valid primary {best_valid_primary:.4f})",
    )
    history.append(iteration0)
    # Iteration 0 is the incumbent best until something beats it, so its test
    # scores are the submission fallback if every later candidate fails.
    capture_best_test_scores(0)
    print(f"  baseline valid.primary={best_valid_primary:.4f} test.primary={baseline_res['test']['primary']:.4f}")
    append_log(iteration0)

    for i in range(1, MAX_ITERATIONS):
        if time.time() - start_time > WALL_CLOCK_LIMIT_SEC:
            print(f"Wall-clock limit hit at iteration {i}. Stopping.")
            break

        current_code = load_current_pipeline_code()

        t0 = time.time()
        try:
            proposal, candidate_code = propose_next_iteration(current_code, history, data_profile)
        except Exception as e:  # noqa: BLE001 -- a bad/malformed LLM response must not kill the run
            iteration = Iteration(
                index=i, hypothesis="(LLM proposal step failed)",
                code_path=WORKDIR / "candidate_pipeline.py",
                error=f"propose_next_iteration failed: {e}",
                duration_sec=time.time() - t0,
            )
            iteration.evaluation = summarize_evaluation(iteration, best_valid_primary)
            print(f"[iter {i}] PROPOSAL ERROR: {e}")
            history.append(iteration)
            append_log(iteration)
            continue

        hypothesis = proposal["hypothesis"]
        code_diff = "".join(difflib.unified_diff(
            current_code.splitlines(keepends=True),
            candidate_code.splitlines(keepends=True),
            fromfile="best_pipeline.py", tofile=f"candidate_iter_{i}.py",
        ))
        iteration = Iteration(
            index=i, hypothesis=hypothesis, code_path=WORKDIR / "candidate_pipeline.py",
            direction=proposal["direction"], source=proposal["source"],
            next_notes=proposal["next_notes"], code_diff=code_diff,
            prompt_tokens=proposal["prompt_tokens"],
            completion_tokens=proposal["completion_tokens"],
        )
        prev_best = best_valid_primary
        try:
            res = run_pipeline(candidate_code)
            iteration.metrics = res
            iteration.duration_sec = time.time() - t0
            vp = res["valid"]["primary"]
            tp = res.get("test", {}).get("primary")
            tp_str = f"{tp:.4f}" if tp is not None else "n/a"
            gap = (vp - tp) if tp is not None else None

            if gap is not None and gap > LEAKAGE_GAP_THRESHOLD:
                # Suspected train/valid/test leakage: validation primary sits far
                # above test primary (honest gap ~0.006). Do NOT accept as the new
                # best even though vp is higher -- keep the previous best. This is
                # a tunable heuristic (LEAKAGE_GAP_THRESHOLD), not a hard cutoff.
                iteration.evaluation = (
                    f"REJECTED: suspected leakage, valid-test gap {gap:.4f} "
                    f"exceeds {LEAKAGE_GAP_THRESHOLD}"
                )
                print(f"[iter {i}] REJECTED (suspected leakage): valid.primary={vp:.4f} "
                      f"test.primary={tp_str} gap={gap:.4f} > {LEAKAGE_GAP_THRESHOLD}; "
                      f"best stays {best_valid_primary:.4f} -- {hypothesis}")
            elif vp > best_valid_primary:
                best_valid_primary = vp
                BEST_CODE_PATH.write_text(candidate_code)
                capture_best_test_scores(i)
                print(f"[iter {i}] NEW BEST valid.primary={vp:.4f} test.primary={tp_str} -- {hypothesis}")
            else:
                print(f"[iter {i}] valid.primary={vp:.4f} (best={best_valid_primary:.4f}) -- {hypothesis}")

        except Exception as e:  # noqa: BLE001 -- log & continue, this IS the robustness story
            iteration.error = str(e)
            iteration.duration_sec = time.time() - t0
            print(f"[iter {i}] ERROR: {e}")

        # Deterministic verdict for the log. A leakage REJECTION is already set
        # above and must not be overwritten.
        if not iteration.evaluation:
            iteration.evaluation = summarize_evaluation(iteration, prev_best)

        history.append(iteration)
        append_log(iteration)

        if has_converged(history):
            print(f"Converged after {i + 1} iterations (incl. bootstrap).")
            break

    wall_sec = time.time() - start_time
    llm_iters = len([h for h in history if h.index >= 1])  # excludes bootstrap iteration 0
    in_tok, out_tok = _TOKEN_TOTALS["prompt"], _TOKEN_TOTALS["completion"]

    print(f"\nDone. Best valid.primary: {best_valid_primary:.4f}")
    print(f"Best pipeline saved at: {BEST_CODE_PATH}")
    print(f"Current run log at: {LOG_PATH}")
    print(f"This run archived at: {archive_path}  (run {run_number}, id {run_id})")
    print()
    print("=== Results Summary (resource usage -- for the deliverable) ===")
    print(f"  Best valid.primary  : {best_valid_primary:.4f}")
    print(f"  Wall-clock          : {wall_sec:.0f}s  ({wall_sec / 60:.1f} min)")
    print(f"  Iterations used     : {llm_iters} LLM iterations "
          f"(cap {MAX_ITERATIONS - 1}); {len(history)} logged incl. bootstrap")
    print(f"  Token consumption   : {in_tok + out_tok:,} total  "
          f"({in_tok:,} input + {out_tok:,} output)  over {_TOKEN_TOTALS['calls']} LLM calls")
    if _TOKEN_TOTALS["calls_without_usage"]:
        print(f"  [!] {_TOKEN_TOTALS['calls_without_usage']} of {_TOKEN_TOTALS['calls']} "
              f"LLM calls returned no usage field -- token totals are a LOWER BOUND")
    print()
    print_run_summary()
    print()
    try:
        write_submission_csv()
    except Exception as e:  # noqa: BLE001 -- a failed submission write must not
        # discard a completed run; best_pipeline.py and best_test_scores.npy are
        # already on disk, so this can be retried standalone afterward.
        print(f"Submission generation failed: {e}")
        print("The run itself is intact -- retry with:\n"
              '  python3 -c "from agent_loop import write_submission_csv; write_submission_csv()"')


def write_submission_csv():
    """Build submission.csv from the winning iteration's saved test-split
    scores. The CSV format is NOT reimplemented here -- submit.py's own
    write_submission() is imported and called, so the file we produce can
    never drift from the format its read_submission() checker enforces."""
    if not BEST_TEST_SCORES_PATH.exists():
        print("No best_test_scores.npy found -- no scored iteration ever saved "
              "test scores. No submission.csv was generated. To submit the "
              "official baseline instead, run the kit's own generator:\n"
              f"  python3 submit.py --make --split test {SUBMISSION_PATH.name}")
        return

    import importlib
    import numpy as np

    if str(WORKDIR) not in sys.path:
        sys.path.insert(0, str(WORKDIR))
    if not (WORKDIR / "submit.py").exists() and not sync_submit_module():
        print("Cannot generate submission.csv: submit.py not found at "
              f"{ROOT / 'submit.py'} -- copy it in from the official starter kit.")
        return

    data_mod = importlib.import_module("data")
    submit_mod = importlib.import_module("submit")

    splits = data_mod.load(str(DATA_DIR.resolve()))
    rows = splits["test"]
    scores = np.load(BEST_TEST_SCORES_PATH)

    if len(scores) != len(rows):
        print(f"WARNING: best_test_scores.npy has {len(scores)} rows but the test "
              f"split has {len(rows)} rows -- these must match exactly. NOT writing "
              f"submission.csv; investigate the winning iteration before submitting.")
        return

    submit_mod.write_submission(str(SUBMISSION_PATH), rows, scores)

    print(f"Wrote {SUBMISSION_PATH} ({len(rows):,d} rows) from the best "
          f"iteration's saved test scores, via submit.py's write_submission().")
    print("Verify independently with the official checker before submitting "
          f"(run from {ROOT}):")
    print(f"  python3 submit.py --check --split test {SUBMISSION_PATH.name}   # format + row alignment")
    expect = best_logged_test_primary()
    note = (f"# should reprint test primary {expect:.4f}" if expect is not None
            else "# rescores the file end-to-end")
    print(f"  python3 submit.py --score --split test {SUBMISSION_PATH.name}   {note}")


def best_logged_test_primary():
    """Test primary of the run's best-by-validation iteration, per run_log.jsonl.
    Used only to print an expected value alongside the --score command, so a
    mismatch between the written CSV and the reported metric is obvious."""
    if not LOG_PATH.exists():
        return None
    best_valid, best_test = -1.0, None
    with LOG_PATH.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                entry = json.loads(line)
            except ValueError:
                continue
            # A rejected (suspected-leakage) iteration never produced
            # best_test_scores.npy, so it can't be the submission source.
            if str(entry.get("evaluation") or "").startswith("REJECTED"):
                continue
            m = entry.get("metrics") or {}
            vp, tp = m.get("valid", {}).get("primary"), m.get("test", {}).get("primary")
            if vp is not None and vp > best_valid:
                best_valid, best_test = vp, tp
    return best_test


def print_run_summary(logfile: Optional[Path] = None):
    """Pretty-print a run log as a readable table. Defaults to run_log.jsonl (the
    latest run); pass a path from runs/ to re-view any past run, e.g.
    python3 -c "from agent_loop import print_run_summary; print_run_summary('runs/run_log_003_20260831T0145.jsonl')"
    """
    path = Path(logfile) if logfile is not None else LOG_PATH
    if not path.exists():
        print(f"No log found at {path}")
        return

    iters = []
    with path.open() as f:
        for line in f:
            line = line.strip()
            if line:
                iters.append(json.loads(line))

    meta = next((it for it in iters if it.get("run_id")), None)
    if meta:
        print(f"Run {meta.get('run_number')} (id {meta.get('run_id')}, "
              f"started {meta.get('run_started')})")

    def fmt_primary(entry, split):
        v = (entry.get("metrics") or {}).get(split, {}).get("primary")
        return f"{v:.4f}" if v is not None else "  --  "

    baseline_valid = None
    best_idx, best_valid = None, -1.0

    print(f"{'#':>3} {'status':<7} {'valid':>8} {'test':>8} {'Δvalid':>8} {'time':>7} "
          f"{'direction':<14} hypothesis")
    print("-" * 108)

    for it in iters:
        idx = it["index"]
        err = it.get("error")
        evaluation = it.get("evaluation")
        rejected = bool(evaluation) and evaluation.startswith("REJECTED")
        status = "ERROR" if err else ("REJECT" if rejected else "OK")
        valid_p = (it.get("metrics") or {}).get("valid", {}).get("primary")
        dur = it.get("duration_sec", 0)
        direction = (it.get("direction") or "-")[:14]

        if idx == 0 and valid_p is not None:
            baseline_valid = valid_p
        # A rejected (suspected-leakage) iteration must never count as "best".
        if valid_p is not None and not rejected and valid_p > best_valid:
            best_valid, best_idx = valid_p, idx

        delta = ""
        if valid_p is not None and baseline_valid is not None:
            delta = f"{valid_p - baseline_valid:+.4f}"

        hyp = it.get("hypothesis", "")
        hyp_short = (hyp[:72] + "...") if len(hyp) > 75 else hyp
        # One compact continuation line: error / leakage verdict / source.
        cont = None
        if err:
            cont = err.strip().splitlines()[-1][:78]
        elif rejected:
            cont = evaluation[:78]
        elif it.get("source") and it.get("source") != "(not provided)":
            cont = f"src: {it['source'][:74]}"
        if cont:
            hyp_short = f"{hyp_short}\n{'':>49}⤷ {cont}"

        print(f"{idx:>3} {status:<7} {fmt_primary(it,'valid'):>8} {fmt_primary(it,'test'):>8} "
              f"{delta:>8} {dur:>6.1f}s {direction:<14} {hyp_short}")

    print("-" * 108)
    n_rej = sum(1 for it in iters
                if str(it.get("evaluation") or "").startswith("REJECTED"))
    n_ok = sum(1 for it in iters if not it.get("error")
               and not str(it.get("evaluation") or "").startswith("REJECTED"))
    tail = f", {n_rej} rejected (suspected leakage)" if n_rej else ""
    print(f"{len(iters)} iterations: {n_ok} succeeded, "
          f"{len(iters) - n_ok - n_rej} errored{tail}.")
    if best_idx is not None:
        print(f"Best so far: iteration {best_idx} (valid primary {best_valid:.4f}, "
              f"baseline was {baseline_valid:.4f})")


if __name__ == "__main__":
    main()