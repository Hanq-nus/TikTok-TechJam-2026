# Autonomous ML Research Agent — KuaiRand-Pure

Solo entry, TikTok TechJam 2026, Track 2 (Autonomous Machine Learning Research
Agent for Recommender Systems).

## Project overview

`agent_loop.py` runs an autonomous propose → execute → evaluate → reflect loop
against the official KuaiRand-Pure starter kit. Each iteration, an LLM
(`qwen3-coder-next`, via an OpenAI-compatible endpoint) is shown the current
best pipeline's real source code, a ground-truth profile of the dataset, and
recent iteration history, then proposes one focused code change. The
candidate is executed against the organizers' own fixed evaluator; the loop
accepts it as the new best, rejects it (including two dedicated guards
against data leakage — see Limitations), or logs the failure and continues —
entirely without human intervention during the run.

Final result: validation primary **0.6043**, hidden-test primary **0.5970**,
vs. the official baseline's 0.5946 (delta **+0.0024**), independently
verified via the organizers' own `submit.py --score`. The winning change was
a contrastive margin-scaled gradient (reweighting the pointwise loss by the
signed margin `y·z`) combined with recency-weighted training samples —
targeting the loss/metric mismatch between pointwise training and the
rank-based GAUC/nDCG@5 scoring.

## Setup and installation

```bash
python3 -m venv venv
source venv/bin/activate
pip install openai python-dotenv
```

Create a `.env` file in the project root:
```
API_KEY=your-key-here
API_BASE_URL=your-endpoint-base-url-here
```

Place the official starter kit files directly in `workdir/`: `data.py`,
`evaluate.py`, `baseline.py`, `submit.py` — unmodified, exactly as provided.
Download and unpack the KuaiRand-Pure dataset per the kit's own instructions
so it ends up at `KuaiRand-Pure/data/` in the project root.

## Running

```bash
python3 agent_loop.py
```

Bootstraps iteration 0 (the verified official FM baseline), then iterates
until convergence (validation primary hasn't improved by more than
ε = 0.002 over 3 consecutive iterations, and never before iteration 20), the
50-iteration cap, or the 6-hour wall-clock limit. A submission-protection
check compares the run's result against the last known-good one and won't
overwrite a better committed result with a worse one.

### Run artifacts

| File | What it is |
| --- | --- |
| `run_log.jsonl` | The **latest run**, one JSON object per iteration: the hypothesis and its DIRECTION label, the unified code diff applied, valid + test GAUC / nDCG@5 / primary, any error traceback, and a deterministic one-line verdict (`Improved … / No improvement … / Failed: … / REJECTED: …`). |
| `run_summary.md` | **Accumulating, human-readable**, newest run first. A "Best result so far" line on top, then one section per run: resources used (iterations vs. the 49 cap, wall-clock, input/output tokens), the number of manual-intervention signals detected, the rollback outcome, and the run's full per-iteration table. |
| `run_summary.json` | The same, machine-readable — one record per run with the resource totals, intervention verdict, `submission_outcome`, and the per-iteration array. |
| `runs/run_log_<n>_<timestamp>.jsonl` | A permanent copy of every run's `run_log.jsonl` (with code diffs), so no run's iteration log is ever lost. |

On completion the agent writes `submission.csv`, updates `run_summary.md` /
`run_summary.json`, and archives the run under `runs/`.

## Reproducing the result

`best_pipeline.py` is a static file once selected — training uses a fixed
numpy seed, so re-running it reproduces the exact submitted numbers:

```bash
cd workdir && python3 best_pipeline.py
```

Or verify the committed submission file directly:

```bash
python3 submit.py --check --split test submission.csv   # format + row alignment
python3 submit.py --score --split test submission.csv   # should reprint test primary 0.5970
```

(Note: this reproduces the *result*. The *search* that found it isn't
reproducible — the LLM proposal step is intentionally unseeded so the agent
can explore freely; see Limitations.)

## Limitations and what I'd improve with more time

- **Run-to-run search variance.** The LLM proposal step is unseeded on
  purpose, so each full run is a genuinely different search and isn't
  guaranteed to rediscover the same result. Multiple runs after the first
  did not beat this one's peak.
- **Two rounds of data leakage were caught and fixed, not just theorized
  about.** A validation-vs-test primary gap guard was added after one run's
  best-looking candidate turned out to be leaking (a same-split expanding
  accumulator inflating validation while test stayed flat). A later run
  found a subtler *symmetric* variant — a per-user history feature that
  leaked into both validation and test equally, evading the gap check
  entirely, and briefly produced a striking-looking (but invalid) test
  primary of 0.6144. That was caught, reverted, and closed with two further
  guards: a static scan for any loop over validation/test rows reading a
  label, and a behavioral backstop rejecting any single-iteration gain
  implausibly larger than any legitimate gain observed across the project.
  A submission-protection layer now also prevents a future run from
  silently overwriting a better, already-committed result.
- **User history/sequence modeling and multi-task learning — the two
  directions the kit's own analysis flagged as most promising — never
  produced a legitimately verified working result.** The one time a
  history-based feature appeared to work, it was the leak described above.
  This is the direction I'd push hardest with more time, built correctly
  from the start (a frozen per-user statistic computed from train rows only,
  looked up by user_id — never recomputed from validation/test rows).
- **No live web search for external methods.** The agent draws on the LLM's
  own training knowledge of published techniques (BPR, listwise ranking,
  etc.) rather than searching for papers at run time — a deliberate scope
  decision given the kit's own analysis already distills a prioritized
  reading list, not an oversight.

## Team member contributions

Solo entry — not applicable.