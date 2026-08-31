import difflib
import hashlib
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
LOG_PATH = ROOT / "run_log.jsonl"   # the LATEST run only (overwritten each run)
RUNS_DIR = ROOT / "runs"            # committed archive of every run
RUNS_INDEX_PATH = RUNS_DIR / "runs_index.json"     # accumulating machine-readable
RUN_SUMMARY_MD_PATH = RUNS_DIR / "run_summary.md"  # accumulating human-readable
BEST_CODE_PATH = ROOT / "best_pipeline.py"
BEST_TEST_SCORES_PATH = ROOT / "best_test_scores.npy"
SUBMISSION_PATH = ROOT / "submission.csv"

# Passed to candidate scripts via an env var.
DATA_DIR = ROOT / "KuaiRand-Pure" / "data"

MAX_ITERATIONS = 50
WALL_CLOCK_LIMIT_SEC = 6 * 60 * 60
CONVERGENCE_EPS = 0.002
CONVERGENCE_N = 3
PER_ITERATION_TIMEOUT_SEC = 5 * 60  # FM baseline is ~40s; kill slow/hung candidates fast to save budget

# Heuristic threshold, not a hard scientific cutoff. The honest FM baseline's
# validation-minus-test primary gap is ~0.006. A candidate whose gap is several
# times that has almost certainly computed a statistic / bucket edge / vocabulary
# using valid or test rows (leakage), inflating validation while test stays flat.
# Candidates exceeding this gap are not accepted as the new best (see main()).
LEAKAGE_GAP_THRESHOLD = 0.02

# The valid-test GAP guard above only catches ASYMMETRIC leaks (valid inflated,
# test not). A feature built from same-split labels for BOTH valid and test
# inflates both equally and slips past it. Backstop: no legitimate single change
# in this task has ever moved validation primary more than ~0.0025 at once, so a
# one-iteration jump bigger than this is auto-rejected for manual review.
LARGE_JUMP_THRESHOLD = 0.015

# When True, main() snapshots the submission-critical files (submission.csv,
# best_pipeline.py, run_log.jsonl, best_test_scores.npy) before the run touches
# them and auto-restores them at the end if this run's best test primary did not
# beat the incumbent, so a run that does not improve cannot lose a good committed
# submission. The run's own log/summary are still kept under runs/. When False,
# the run always overwrites those files.
PROTECT_BEST_SUBMISSION = True

MODEL = "qwen3-coder-next"
API_BASE_URL = os.environ.get("API_BASE_URL")
API_KEY = os.environ.get("API_KEY")

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

# Best-effort manual-intervention detection (see _intervention_verdict()). Reset
# by main(). best_pipeline_hash is refreshed after every write the loop itself
# performs; file_integrity_violations collects any on-disk change the loop did
# NOT make; source_hash_at_start pins agent_loop.py's own source at run start.
_INTEGRITY: dict = {
    "best_pipeline_hash": None,
    "file_integrity_violations": [],
    "source_hash_at_start": None,
}


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


# ---- Leakage & integrity checks ----------------------------------------------

# A loop over splits[<'train'|'valid'|'test'|variable>], capturing which index.
_SPLIT_ITER_RE = re.compile(
    r"for\s+[\w ,]+\s+in\s+(?:enumerate\s*\(\s*)?splits\s*\[\s*"
    r"(?:['\"](?P<lit>train|valid|test)['\"]|(?P<var>\w+))\s*\]"
)
# A per-row label read from a raw KuaiRand tuple: index 6 is `long_view`, [-1] too.
_LABEL_READ_RE = re.compile(r"\[\s*6\s*\]|\[\s*-\s*1\s*\]")


def scan_candidate_for_leakage(code: str):
    """Best-effort STATIC scan for the leak class the valid-test gap guard can't
    catch: a per-row feature built from valid/test *labels* (raw-tuple index 6 =
    long_view, or [-1]), computed per-split so BOTH valid and test inflate
    equally. Flags a loop over splits[valid|test] (or a variable split name --
    the leaky pattern parameterises it) that reads a label index within ~15
    lines. Returns a reason string, or None if clean. Heuristic: a legitimate
    candidate can trip it; widen the window / tighten the regex if so."""
    lines = code.splitlines()
    for m in _SPLIT_ITER_RE.finditer(code):
        if m.group("lit") == "train":
            continue  # aggregating over train only is the CORRECT pattern
        ln = code[:m.start()].count("\n")
        window = "\n".join(lines[ln:ln + 15])
        lab = _LABEL_READ_RE.search(window)
        if lab:
            which = m.group("lit") or f"variable '{m.group('var')}'"
            return (f"line ~{ln + 1}: loop over splits[{which}] reads a per-row label "
                    f"({lab.group(0)!r}) within ~15 lines -- a feature built from "
                    f"valid/test labels leaks symmetrically across both splits and "
                    f"evades the valid-test primary gap guard")
    return None


def _sha256_file(path):
    try:
        return hashlib.sha256(Path(path).read_bytes()).hexdigest()
    except OSError:
        return None


def _record_best_pipeline_hash():
    """Call immediately after the loop itself writes best_pipeline.py."""
    _INTEGRITY["best_pipeline_hash"] = _sha256_file(BEST_CODE_PATH)


def _verify_best_pipeline_hash(iteration_index):
    """Call immediately before load_current_pipeline_code(). Any mismatch means
    something other than this loop edited best_pipeline.py on disk."""
    expected = _INTEGRITY["best_pipeline_hash"]
    if expected is None:
        return
    actual = _sha256_file(BEST_CODE_PATH)
    if actual != expected:
        _INTEGRITY["file_integrity_violations"].append({
            "iteration": iteration_index,
            "detail": (f"best_pipeline.py content hash changed outside the loop "
                       f"(expected {expected[:12]}..., found "
                       f"{actual[:12] + '...' if actual else 'unreadable'})"),
        })
        _INTEGRITY["best_pipeline_hash"] = actual  # re-baseline so each edit flags once


def _intervention_verdict(wall_sec: float, history, prof_sec: float) -> dict:
    """Combine the three checkable intervention signals into a verdict. Not proof
    of anything -- vectors outside this process are invisible from here."""
    accounted = prof_sec + sum(h.duration_sec for h in history)
    gap = wall_sec - accounted
    gap_exceeds = gap > 30.0 and gap > 0.05 * wall_sec

    src_end = _sha256_file(__file__)
    src_start = _INTEGRITY["source_hash_at_start"]
    source_modified = bool(src_start and src_end and src_end != src_start)

    violations = list(_INTEGRITY["file_integrity_violations"])
    if not violations and not source_modified and not gap_exceeds:
        summary = (
            "No evidence of manual intervention detected (best_pipeline.py integrity "
            "intact, script source unmodified, no unexplained wall-clock gap). This is "
            "evidence of an unattended run, not proof -- intervention vectors outside "
            "this process (e.g. editing unrelated files, OS-level pause/resume) are not "
            "detectable from here."
        )
    else:
        parts = []
        if violations:
            parts.append(
                f"{len(violations)} best_pipeline.py integrity violation(s) [" +
                "; ".join(f"iter {v['iteration']}: {v['detail']}" for v in violations) + "]")
        if source_modified:
            parts.append("agent_loop.py's own source file changed on disk during the run")
        if gap_exceeds:
            parts.append(
                f"unexplained wall-clock gap of {gap:.0f}s "
                f"({gap / wall_sec * 100:.1f}% of {wall_sec:.0f}s total) -- soft heuristic, not proof")
        summary = "POSSIBLE MANUAL INTERVENTION DETECTED: " + "; ".join(parts)

    return {
        "intervention_signals_detected": violations,
        "own_source_modified_during_run": source_modified,
        "unaccounted_wall_clock_gap_seconds": round(gap, 1),
        "unaccounted_wall_clock_gap_exceeds_threshold": gap_exceeds,
        "summary": summary,
    }


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
  almost always leaking. The loop AUTO-REJECTS any candidate whose
  (valid primary - test primary) gap exceeds ~0.02 -- the honest baseline
  gap is ~0.006 -- keeping the previous best instead.
  ⚠️ SYMMETRIC LEAKAGE -- the trap the gap check CANNOT see. If you build a
  per-row feature from the LABELS of other rows in the SAME split -- e.g.
  "this user's mean long_view over their previous VALID impressions" for
  valid rows, and over their previous TEST impressions for test rows -- then
  BOTH valid and test primary inflate by the same amount, the gap stays
  normal, and the check above passes a result that is still cheating. The
  hidden-test re-score would just be measuring the peek. This is explicitly
  forbidden: a history / rate / count / sequence feature for a valid or test
  row MUST be computed from that user's TRAIN rows only (a frozen per-user
  train statistic, looked up by user_id), NEVER from other valid/test rows.
  Two extra guards enforce this: (a) candidates are STATICALLY SCANNED for a
  loop over splits['valid'/'test'] (or a variable split name) that reads a
  per-row label (raw-tuple index 6 = long_view, or [-1]) -- a match is
  auto-rejected before it even runs; (b) any single iteration that gains
  more than ~0.015 validation primary over the current best is auto-rejected
  for manual review, because no legitimate change in this task has ever
  moved the score that much in one step. Build the feature train-only the
  first time.
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
    # Pin our own source hash at the very start (Part 2 check #2).
    _INTEGRITY.update(best_pipeline_hash=None, file_integrity_violations=[],
                      source_hash_at_start=_sha256_file(__file__))

    WORKDIR.mkdir(exist_ok=True)
    RUNS_DIR.mkdir(exist_ok=True)

    run_started = time.strftime("%Y-%m-%dT%H:%M:%S")
    run_id = time.strftime("%Y%m%dT%H%M%S")

    # Snapshot the submission-critical files BEFORE bootstrap overwrites them, so
    # finalize_submission() can roll back a run that doesn't beat the incumbent.
    restore_dir = RUNS_DIR / f"_incumbent_{run_id}"
    incumbent_test_primary = (snapshot_incumbent(restore_dir)
                              if PROTECT_BEST_SUBMISSION else None)

    bootstrap_pipeline()
    _record_best_pipeline_hash()  # the loop just wrote best_pipeline.py

    # Identify this run. run_log.jsonl holds just this run (so print_run_summary
    # etc. keep working unchanged); a timestamped copy is also kept under runs/
    # so no past run is ever lost. run_number is "files already in runs/ + 1" --
    # best-effort ordering, the timestamp in the name is the real identifier.
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
    _t_prof = time.time()
    data_profile = profile_dataset()
    prof_sec = time.time() - _t_prof
    print(data_profile)

    # Score iteration 0 (the deterministic baseline) first, before any LLM call.
    print("Scoring bootstrap baseline (iteration 0)...")
    t0 = time.time()
    _verify_best_pipeline_hash(0)
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

        _verify_best_pipeline_hash(i)
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

        # Static leakage scan -- reject BEFORE running (saves the compute) if the
        # candidate reads valid/test labels to build a feature (the symmetric
        # leak the valid-test gap guard can't catch).
        leak_reason = scan_candidate_for_leakage(candidate_code)
        if leak_reason is not None:
            iteration.evaluation = f"REJECTED: static leakage scan -- {leak_reason}"
            iteration.duration_sec = time.time() - t0
            print(f"[iter {i}] REJECTED (static leakage scan): {leak_reason} -- {hypothesis}")
            history.append(iteration)
            append_log(iteration)
            continue

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
                # above test primary (honest gap ~0.006). Not accepted as the new
                # best even though vp is higher; the previous best is kept.
                iteration.evaluation = (
                    f"REJECTED: suspected leakage, valid-test gap {gap:.4f} "
                    f"exceeds {LEAKAGE_GAP_THRESHOLD}"
                )
                print(f"[iter {i}] REJECTED (suspected leakage): valid.primary={vp:.4f} "
                      f"test.primary={tp_str} gap={gap:.4f} > {LEAKAGE_GAP_THRESHOLD}; "
                      f"best stays {best_valid_primary:.4f} -- {hypothesis}")
            elif (vp - prev_best) > LARGE_JUMP_THRESHOLD:
                # Backstop for a symmetric leak the gap check misses: an
                # implausibly large one-step gain. Not accepted -- flagged for
                # manual review.
                iteration.evaluation = (
                    f"REJECTED: implausibly large single-iteration gain "
                    f"(+{vp - prev_best:.4f} vs best {prev_best:.4f}, threshold "
                    f"{LARGE_JUMP_THRESHOLD}) -- likely undetected leakage, needs manual review"
                )
                print(f"[iter {i}] REJECTED (implausible +{vp - prev_best:.4f} jump): "
                      f"valid.primary={vp:.4f} test.primary={tp_str}; best stays "
                      f"{best_valid_primary:.4f} -- {hypothesis}")
            elif vp > best_valid_primary:
                best_valid_primary = vp
                BEST_CODE_PATH.write_text(candidate_code)
                _record_best_pipeline_hash()  # the loop just wrote best_pipeline.py
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
    verdict = _intervention_verdict(wall_sec, history, prof_sec)

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
    print(f"  Manual-intervention check: {verdict['summary']}")
    print()
    summary = print_run_summary()
    print()
    outcome = finalize_submission(summary, incumbent_test_primary, restore_dir)
    print()
    try:
        for label, p in _append_run_summary(
                run_id, run_number, run_started, summary, verdict,
                wall_sec, llm_iters, len(history), in_tok, out_tok,
                outcome, incumbent_test_primary):
            print(f"{label}: {p}")
    except Exception as e:  # noqa: BLE001 -- a summary-file failure must not
        # discard an otherwise-complete run.
        print(f"Run-summary file not updated (run is otherwise complete): {e}")


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
    """Pretty-print a run log as a readable table AND return its computed data.
    Defaults to run_log.jsonl (the latest run); pass a path from runs/ to
    re-view any past run, e.g.
    python3 -c "from agent_loop import print_run_summary; print_run_summary('runs/run_log_003_20260831T0145.jsonl')"

    Returns None if the log file is missing, otherwise a dict:
      {"text", "iterations", "best_index", "best_valid_primary",
       "best_test_primary", "baseline_valid_primary", "n_ok", "n_errored",
       "n_rejected"}
    Terminal output is byte-for-byte identical to printing alone."""
    path = Path(logfile) if logfile is not None else LOG_PATH
    if not path.exists():
        print(f"No log found at {path}")
        return None

    iters = []
    with path.open() as f:
        for line in f:
            line = line.strip()
            if line:
                iters.append(json.loads(line))

    lines: list = []

    def _emit(s=""):
        print(s)
        lines.append(s)

    meta = next((it for it in iters if it.get("run_id")), None)
    if meta:
        _emit(f"Run {meta.get('run_number')} (id {meta.get('run_id')}, "
              f"started {meta.get('run_started')})")

    def fmt_primary(entry, split):
        v = (entry.get("metrics") or {}).get(split, {}).get("primary")
        return f"{v:.4f}" if v is not None else "  --  "

    baseline_valid = None
    best_idx, best_valid, best_test = None, -1.0, None
    table_rows: list = []

    _emit(f"{'#':>3} {'status':<7} {'valid':>8} {'test':>8} {'Δvalid':>8} {'time':>7} "
          f"{'direction':<14} hypothesis")
    _emit("-" * 108)

    for it in iters:
        idx = it["index"]
        err = it.get("error")
        evaluation = it.get("evaluation")
        rejected = bool(evaluation) and evaluation.startswith("REJECTED")
        status = "ERROR" if err else ("REJECT" if rejected else "OK")
        valid_p = (it.get("metrics") or {}).get("valid", {}).get("primary")
        test_p = (it.get("metrics") or {}).get("test", {}).get("primary")
        dur = it.get("duration_sec", 0)
        direction = (it.get("direction") or "-")[:14]

        if idx == 0 and valid_p is not None:
            baseline_valid = valid_p
        # A rejected (suspected-leakage) iteration must never count as "best".
        if valid_p is not None and not rejected and valid_p > best_valid:
            best_valid, best_idx, best_test = valid_p, idx, test_p

        delta_valid = (valid_p - baseline_valid) if (
            valid_p is not None and baseline_valid is not None) else None
        delta = f"{delta_valid:+.4f}" if delta_valid is not None else ""

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

        _emit(f"{idx:>3} {status:<7} {fmt_primary(it,'valid'):>8} {fmt_primary(it,'test'):>8} "
              f"{delta:>8} {dur:>6.1f}s {direction:<14} {hyp_short}")

        table_rows.append({
            "index": idx,
            "status": status,
            "valid_primary": valid_p,
            "test_primary": test_p,
            "delta_valid": delta_valid,
            "direction": it.get("direction"),
            "duration_sec": dur,
            "hypothesis": it.get("hypothesis", ""),
            "source": it.get("source"),
        })

    _emit("-" * 108)
    n_rej = sum(1 for it in iters
                if str(it.get("evaluation") or "").startswith("REJECTED"))
    n_ok = sum(1 for it in iters if not it.get("error")
               and not str(it.get("evaluation") or "").startswith("REJECTED"))
    tail = f", {n_rej} rejected (suspected leakage)" if n_rej else ""
    _emit(f"{len(iters)} iterations: {n_ok} succeeded, "
          f"{len(iters) - n_ok - n_rej} errored{tail}.")
    if best_idx is not None:
        _emit(f"Best so far: iteration {best_idx} (valid primary {best_valid:.4f}, "
              f"baseline was {baseline_valid:.4f})")

    return {
        "text": "\n".join(lines),
        "iterations": table_rows,
        "best_index": best_idx,
        "best_valid_primary": best_valid if best_idx is not None else None,
        "best_test_primary": best_test if best_idx is not None else None,
        "baseline_valid_primary": baseline_valid,
        "n_ok": n_ok,
        "n_errored": len(iters) - n_ok - n_rej,
        "n_rejected": n_rej,
    }


# ---- Submission protection --------------------------------------------------

# All four are snapshotted before a run. On a rollback only the SUBMISSION files
# are restored -- run_log.jsonl deliberately stays as the LATEST run's (it is the
# tracked "current representative run", distinct from the best-kept submission).
_PROTECTED_FILES = ("submission.csv", "best_pipeline.py", "run_log.jsonl",
                    "best_test_scores.npy")
_SUBMISSION_FILES = ("submission.csv", "best_pipeline.py", "best_test_scores.npy")


def _score_submission_test(path: Path) -> Optional[float]:
    """Independently score an existing submission.csv on the test split (loads
    data fresh, uses the untouched evaluate()). None if it can't be scored."""
    if not path.exists():
        return None
    try:
        import importlib
        if str(WORKDIR) not in sys.path:
            sys.path.insert(0, str(WORKDIR))
        if not (WORKDIR / "submit.py").exists():
            sync_submit_module()
        data_mod = importlib.import_module("data")
        submit_mod = importlib.import_module("submit")
        ev_mod = importlib.import_module("evaluate")
        rows = data_mod.load(str(DATA_DIR.resolve()))["test"]
        scores = submit_mod.read_submission(str(path), rows)
        r = ev_mod.evaluate([x[1] for x in rows], [x[6] for x in rows], scores)
        return float(r["primary"])
    except Exception as e:  # noqa: BLE001
        print(f"  (could not score existing submission.csv: {e})")
        return None


def snapshot_incumbent(restore_dir: Path) -> Optional[float]:
    """Copy the submission-critical files aside (byte copies) and return the
    incumbent submission.csv's test primary, scored directly -- run_log.jsonl is
    the LATEST run and may not be the run behind the submission."""
    incumbent = _score_submission_test(SUBMISSION_PATH)
    restore_dir.mkdir(parents=True, exist_ok=True)
    saved = []
    for name in _PROTECTED_FILES:
        p = ROOT / name
        if p.exists():
            (restore_dir / name).write_bytes(p.read_bytes())
            saved.append(name)
    where = restore_dir.relative_to(ROOT)
    if incumbent is not None:
        print(f"Protection ON -- snapshotted {', '.join(saved) or 'nothing'} to {where}; "
              f"incumbent submission test primary = {incumbent:.4f}")
    else:
        print(f"Protection ON -- snapshotted {', '.join(saved) or 'nothing'} to {where}; "
              f"no scorable incumbent submission (any result will be kept)")
    return incumbent


def finalize_submission(summary, incumbent_test_primary: Optional[float],
                        restore_dir: Path) -> str:
    """Keep this run's submission if it STRICTLY beat the incumbent, else roll
    submission.csv / best_pipeline.py / best_test_scores.npy back to the pre-run
    snapshot. run_log.jsonl is NOT rolled back -- it stays as the latest run's.
    Returns one of: "kept", "kept_no_incumbent", "kept_unprotected", "rolled_back"."""
    new_best_test = (summary or {}).get("best_test_primary")
    guarding = (PROTECT_BEST_SUBMISSION and incumbent_test_primary is not None
                and restore_dir.exists())

    if guarding and (new_best_test is None
                     or new_best_test <= incumbent_test_primary + 1e-9):
        got = f"{new_best_test:.4f}" if new_best_test is not None else "none"
        print(f"\nThis run's best test primary ({got}) did NOT beat the incumbent "
              f"submission ({incumbent_test_primary:.4f}). Restoring the submission "
              f"files from {restore_dir.relative_to(ROOT)} (run_log.jsonl stays as "
              f"this run's):")
        for name in _SUBMISSION_FILES:
            src, dst = restore_dir / name, ROOT / name
            if src.exists():
                dst.write_bytes(src.read_bytes())
                print(f"  restored {name}")
            elif dst.exists():
                dst.unlink()  # incumbent had none of this file -- drop this run's
                print(f"  removed {name} (incumbent had none)")
        print("This run's full log is in the runs/ archive and runs/run_summary.md -- nothing lost.")
        return "rolled_back"

    try:
        write_submission_csv()
    except Exception as e:  # noqa: BLE001 -- a failed submission write must not
        # discard a completed run; best_pipeline.py and best_test_scores.npy are
        # already on disk, so this can be retried standalone afterward.
        print(f"Submission generation failed: {e}")
        print("The run itself is intact -- retry with:\n"
              '  python3 -c "from agent_loop import write_submission_csv; write_submission_csv()"')

    if guarding:
        print(f"\nThis run beat the incumbent ({new_best_test:.4f} > "
              f"{incumbent_test_primary:.4f}). Submission files updated; previous "
              f"versions saved at {restore_dir.relative_to(ROOT)}.")
        return "kept"
    if PROTECT_BEST_SUBMISSION and incumbent_test_primary is None:
        print("\n(No scorable prior submission to protect -- kept this run's output.)")
        return "kept_no_incumbent"
    return "kept_unprotected"


def preflight():
    """`python3 agent_loop.py --preflight`: report what a run would overwrite and
    whether it's safe to run, WITHOUT running anything."""
    print("Preflight -- a run of agent_loop.py overwrites these files:")
    for name in _PROTECTED_FILES:
        p = ROOT / name
        print(f"  {name:<22} {'present' if p.exists() else 'absent'}")

    sc = _score_submission_test(SUBMISSION_PATH)
    logged = best_logged_test_primary()
    print(f"  submission.csv test primary (scored)      : "
          f"{sc:.4f}" if sc is not None else "  submission.csv test primary (scored)      : n/a")
    print(f"  run_log.jsonl best test primary (recorded): "
          f"{logged:.4f}" if logged is not None else "  run_log.jsonl best test primary (recorded): n/a")
    if sc is not None and logged is not None and abs(sc - logged) > 5e-4:
        print("  note: submission.csv is the best-KEPT result, run_log.jsonl is the "
              "LATEST run -- they can legitimately differ under this layout")

    try:
        out = subprocess.run(
            ["git", "status", "--porcelain", "--", *_PROTECTED_FILES],
            cwd=str(ROOT), capture_output=True, text=True, timeout=15)
        dirty = [l for l in out.stdout.splitlines() if l.strip()]
        if dirty:
            print("  ⚠️ uncommitted changes in protected files:")
            for l in dirty:
                print(f"      {l}")
            print("  Commit or stash them first so a bad run can be rolled back to a known state.")
        else:
            print("  protected files are clean vs git HEAD")
    except Exception:  # noqa: BLE001
        print("  (git status check skipped -- not a git repo or git unavailable)")

    mode = "ON" if PROTECT_BEST_SUBMISSION else "OFF"
    ref = sc if sc is not None else logged
    if PROTECT_BEST_SUBMISSION and ref is not None:
        print(f"\nProtection mode: {mode} -- a run whose best test primary is <= "
              f"{ref:.4f} will auto-restore submission.csv / best_pipeline.py / "
              f"best_test_scores.npy (run_log.jsonl keeps the latest run).")
    else:
        print(f"\nProtection mode: {mode}.")


def _append_run_summary(run_id, run_number, run_started, summary, verdict,
                        wall_sec, llm_iters, iterations_logged, in_tok, out_tok,
                        outcome, incumbent_test_primary):
    """Append this run to runs/runs_index.json and regenerate runs/run_summary.md
    -- ONE accumulating human-readable file, newest run first, with a
    'Best result so far' line on top. Returns [(label, Path), ...]."""
    summary = summary or {}
    RUNS_DIR.mkdir(exist_ok=True)

    n_signals = (len(verdict["intervention_signals_detected"])
                 + (1 if verdict["own_source_modified_during_run"] else 0)
                 + (1 if verdict["unaccounted_wall_clock_gap_exceeds_threshold"] else 0))

    record = {
        "run_number": run_number,
        "run_id": run_id,
        "run_started": run_started,
        "submission_outcome": outcome,
        "best_iteration_index": summary.get("best_index"),
        "best_valid_primary": summary.get("best_valid_primary"),
        "best_test_primary": summary.get("best_test_primary"),
        "baseline_valid_primary": summary.get("baseline_valid_primary"),
        "iterations_used": llm_iters,
        "iterations_logged": iterations_logged,
        "iterations_cap": MAX_ITERATIONS - 1,
        "tokens": {
            "prompt": in_tok, "completion": out_tok, "total": in_tok + out_tok,
            "calls": _TOKEN_TOTALS["calls"],
            "calls_without_usage": _TOKEN_TOTALS["calls_without_usage"],
        },
        "wall_clock_seconds": round(wall_sec, 1),
        "manual_intervention_signals": n_signals,
        "manual_intervention_summary": verdict["summary"],
        "own_source_modified_during_run": verdict["own_source_modified_during_run"],
        "intervention_signals_detected": verdict["intervention_signals_detected"],
        "unaccounted_wall_clock_gap_seconds": verdict["unaccounted_wall_clock_gap_seconds"],
        "unaccounted_wall_clock_gap_exceeds_threshold": verdict["unaccounted_wall_clock_gap_exceeds_threshold"],
        "summary_text": summary.get("text") or "",
        "iterations": summary.get("iterations", []),
    }

    index = []
    if RUNS_INDEX_PATH.exists():
        try:
            loaded = json.loads(RUNS_INDEX_PATH.read_text())
            index = loaded if isinstance(loaded, list) else []
        except ValueError:
            index = []
    index = [r for r in index if r.get("run_id") != run_id]  # replace on a re-run
    index.append(record)
    RUNS_INDEX_PATH.write_text(json.dumps(index, indent=2))

    kept = [r for r in index
            if str(r.get("submission_outcome", "")).startswith("kept")
            and r.get("best_test_primary") is not None]
    if kept:
        b = max(kept, key=lambda r: r["best_test_primary"])
        best_line = (f"**Best result so far:** test primary {b['best_test_primary']:.4f} "
                     f"(valid {b['best_valid_primary']:.4f}) -- run {b['run_number']} "
                     f"(`{b['run_id']}`)")
    elif incumbent_test_primary is not None:
        best_line = (f"**Best result so far:** test primary {incumbent_test_primary:.4f} "
                     f"-- committed submission.csv, from a run predating this tracking")
    else:
        best_line = "**Best result so far:** none recorded yet"

    md = ["# Agent run summaries", "",
          f"_KuaiRand-Pure autonomous agent -- {len(index)} run(s) recorded, newest first._",
          "", best_line, ""]
    for r in sorted(index, key=lambda r: r.get("run_id", ""), reverse=True):
        t = r["tokens"]
        lb = "  [some calls lacked usage -- LOWER BOUND]" if t["calls_without_usage"] else ""
        md += [
            "---", "",
            f"## Run {r['run_number']} -- `{r['run_id']}` -- {r['submission_outcome']}",
            f"_started {r['run_started']}_", "",
            f"- Best: valid {r['best_valid_primary']} / test {r['best_test_primary']} "
            f"(iteration {r['best_iteration_index']}); baseline valid {r['baseline_valid_primary']}",
            f"- Iterations: {r['iterations_used']} used (cap {r['iterations_cap']}), "
            f"{r['iterations_logged']} logged incl. bootstrap",
            f"- Wall-clock: {r['wall_clock_seconds']}s ({r['wall_clock_seconds'] / 60:.1f} min)",
            f"- Tokens: {t['total']:,} total ({t['prompt']:,} in + {t['completion']:,} out) "
            f"over {t['calls']} LLM calls{lb}",
            f"- Manual interventions detected: {r['manual_intervention_signals']} "
            f"-- {r['manual_intervention_summary']}",
            f"- Full per-iteration log with code diffs: "
            f"`runs/run_log_{r['run_number']:03d}_{r['run_id']}.jsonl`",
            "", "```",
            r.get("summary_text") or "(no summary text)",
            "```", "",
        ]
    RUN_SUMMARY_MD_PATH.write_text("\n".join(md))
    return [("Run summary (md)", RUN_SUMMARY_MD_PATH), ("Runs index (json)", RUNS_INDEX_PATH)]


if __name__ == "__main__":
    if "--preflight" in sys.argv:
        preflight()
    else:
        main()