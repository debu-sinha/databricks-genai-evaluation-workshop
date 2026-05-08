"""Scorer definitions for the workshop.

Lives in a regular Python module file (not a notebook) so the scheduled-scorer
subsystem can deserialize the function objects on remote workers. Notebook-defined
scorers fail to deserialize because their __module__ resolves to __main__, which
remote workers can't reconstruct.
"""

from mlflow.genai.scorers import scorer
from mlflow.genai.judges import make_judge


@scorer
def answer_non_empty(outputs):
    """Cheap safety scorer. Run on 100% of traces.

    Self-contained: only uses builtins, no external imports needed at function-call time.
    """
    if outputs is None:
        return 0.0
    answer = ""
    if isinstance(outputs, dict):
        answer = outputs.get("answer") or ""
    elif isinstance(outputs, str):
        answer = outputs
    return 1.0 if str(answer).strip() else 0.0


def make_relevance_judge():
    """Factory for the relevance judge. Returns a fresh judge per call.

    Defined as a factory rather than a module-level constant so that the
    `make_judge` object isn't created at import time (which can have side effects).
    """
    return make_judge(
        name="relevance",
        instructions=(
            "You are evaluating whether an answer is relevant to a user's question.\n\n"
            "Question: {{ inputs }}\n"
            "Answer: {{ outputs }}\n\n"
            "Reply with exactly one word from this set: yes, partial, no."
        ),
        model="databricks:/databricks-claude-sonnet-4-6",
    )
