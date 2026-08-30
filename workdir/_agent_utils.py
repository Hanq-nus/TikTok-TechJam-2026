"""Fixed helper module, not LLM-generated. Available to every candidate
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
