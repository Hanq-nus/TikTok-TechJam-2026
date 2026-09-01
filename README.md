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

Final result: validation primary **0.6040**, hidden-test primary **0.5975**,
vs. the official baseline's 0.5946 (delta **+0.0029**), independently
verified via the organizers' own `submit.py --score`. The winning pipeline
adds a **train-only per-user long-view rate**, bucketed into 6 bins, as an
extra FM field — so it interacts with the item-side embeddings (the kit's
own note that user features only help ranking *through* such interactions),
computed strictly from each user's training rows and looked up by
`user_id`, with unseen users falling into a shared UNK bucket. On top of
that sit a trained per-bucket scalar offset and per-user bias/temperature
terms (rank-neutral calibration). Earlier the agent kept abandoning this
direction after one or two flat iterations; letting it keep refining a
direction that already holds the running best is what carried it past the
prior calibration-only plateau at 0.5970.

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
numpy seed, so re-running it reproduces the exact submitted numbers
(`valid 0.6040 / test 0.5975`). It imports `data.py` / `evaluate.py` from
`workdir/`, so run it with that on the path:

```bash
cd workdir && KUAIRAND_DATA_DIR=../KuaiRand-Pure/data python3 ../best_pipeline.py
```

Or verify the committed submission file directly:

```bash
python3 submit.py --check --split test submission.csv   # format + row alignment
python3 submit.py --score --split test submission.csv   # should reprint test primary 0.5975
```

(Note: this reproduces the *result*. The *search* that found it isn't
reproducible — the LLM proposal step is intentionally unseeded so the agent
can explore freely; see Limitations.)

## Limitations and what I'd improve with more time

- **Run-to-run search variance.** The LLM proposal step is unseeded on
  purpose, so each full run is a genuinely different search and isn't
  guaranteed to rediscover the same result. Across ~13 full runs the
  hidden-test primary ranged roughly 0.594–0.5975; the submitted pipeline
  is the best of those, kept automatically by the submission-protection
  layer, and independently re-verified.
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
- **User-history modeling only started paying off once the loop was allowed
  to *refine* it.** The first working, non-leaky version of a train-only
  per-user rate feature (now in the winning pipeline) still only adds
  ~+0.0005 test primary. Two things held it back for most of the project:
  the leakage-prone obvious implementations (reading valid/test labels),
  which the guards now reject; and an anti-repetition rule that forced the
  agent off a direction after two flat iterations, before it could tune the
  bins / decay / field encoding. With more time I'd push this further —
  proper sequence features (per-impression EMA rate, recency gaps) rather
  than a single static bucket — and revisit multi-task learning with the
  auxiliary signals (`is_click`, `is_like`, …), which never produced a
  verified gain.
- **No live web search for external methods.** The agent draws on the LLM's
  own training knowledge of published techniques (BPR, listwise ranking,
  etc.) rather than searching for papers at run time — a deliberate scope
  decision given the kit's own analysis already distills a prioritized
  reading list, not an oversight.

## Team member contributions

Solo entry — not applicable.