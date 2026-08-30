"""Fixed helper module, not LLM-generated. Available to every candidate
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
