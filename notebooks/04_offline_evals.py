# Databricks notebook source
# MAGIC %md
# MAGIC # Lesson 4 - Run an offline evaluation
# MAGIC
# MAGIC `mlflow.genai.evaluate` over a small dataset with a custom code-based scorer and an LLM judge.

# COMMAND ----------

# MAGIC %md
# MAGIC ## What is happening?
# MAGIC
# MAGIC `mlflow.genai.evaluate(data, predict_fn, scorers)` calls your agent on each row of the dataset,
# MAGIC runs every scorer against the result, and produces an eval run with per-row assessments and
# MAGIC aggregate metrics.
# MAGIC
# MAGIC Two scorer flavors live side by side here:
# MAGIC
# MAGIC - **Code-based scorer** (`@scorer` decorator). Cheap, deterministic, no LLM cost. Good for
# MAGIC   structural checks like format validation, keyword overlap, response length.
# MAGIC - **LLM-as-a-judge** (`make_judge`). Prompt-based scoring. Higher signal, higher cost. The
# MAGIC   prompt must constrain the output to a fixed value set so the judge response parses cleanly
# MAGIC   into a categorical or numeric value.
# MAGIC
# MAGIC Both kinds of scorer are first-class on the same eval run. You can mix as many as you want.
# MAGIC
# MAGIC ## Choosing a scorer
# MAGIC
# MAGIC | Approach | When to use | Cost | Output |
# MAGIC |---|---|---|---|
# MAGIC | Built-in scorers | Standard quality checks (latency, format) | Free | Numeric or boolean |
# MAGIC | Guidelines judge | Quick guideline-based check | LLM call per row | Pass/fail |
# MAGIC | Custom code-based scorer (`@scorer`) | Custom logic, statistical metrics | Free | Numeric, boolean, or string |
# MAGIC | Custom LLM-as-a-judge (`make_judge`) | Subjective quality, faithfulness, helpfulness | LLM call per row | Whatever your prompt constrains |
# MAGIC
# MAGIC References: [custom scorers](https://docs.databricks.com/aws/en/mlflow3/genai/eval-monitor/custom-scorers),
# MAGIC [make_judge](https://docs.databricks.com/aws/en/mlflow3/genai/eval-monitor/custom-judge/create-custom-judge).

# COMMAND ----------

# MAGIC %pip install -U -qqqq mlflow databricks-sdk databricks-agents
# MAGIC dbutils.library.restartPython()

# COMMAND ----------

# MAGIC %run ./_resources/setup

# COMMAND ----------

# MAGIC %md
# MAGIC ### Eval dataset
# MAGIC
# MAGIC Six rows: four queries the agent should be able to answer from the corpus, two it should refuse
# MAGIC because they're out of scope. The mix gives the eval real signal instead of a row of zeros.

# COMMAND ----------

import pandas as pd

eval_dataset = pd.DataFrame(
    [
        {
            "inputs": {"query": "How are MLflow traces structured?"},
            "expected_response": "MLflow traces use hierarchical spans for chain, retrieve, and generate steps.",
            "category": "in-corpus",
        },
        {
            "inputs": {"query": "Can I store OpenTelemetry traces in Unity Catalog?"},
            "expected_response": "Yes, OTel traces can be ingested into Unity Catalog Delta tables.",
            "category": "in-corpus",
        },
        {
            "inputs": {"query": "What is make_judge?"},
            "expected_response": "make_judge defines an LLM judge in MLflow.",
            "category": "in-corpus",
        },
        {
            "inputs": {
                "query": "What drift metrics does Lakehouse monitoring compute?"
            },
            "expected_response": "Lakehouse monitoring computes PSI, JS, KS, and chi-squared drift metrics.",
            "category": "in-corpus",
        },
        {
            "inputs": {"query": "What is the airspeed velocity of an unladen swallow?"},
            "expected_response": "I don't know.",
            "category": "out-of-corpus",
        },
        {
            "inputs": {"query": "What's the capital of France?"},
            "expected_response": "I don't know.",
            "category": "out-of-corpus",
        },
    ]
)
eval_dataset

# COMMAND ----------

# MAGIC %md
# MAGIC ### Custom code-based scorer
# MAGIC
# MAGIC `@scorer` turns a Python function into a pluggable scorer. The function receives the agent's
# MAGIC output and the row's expectations as keyword arguments. Return a number, boolean, or string.
# MAGIC
# MAGIC This one checks whether any non-trivial keyword from the expected response appears in the
# MAGIC agent's actual answer. Crude, but it catches blatantly wrong answers.

# COMMAND ----------

from mlflow.genai.scorers import scorer


@scorer
def answer_contains_expected_keyword(outputs: dict, expectations: dict) -> float:
    answer = (outputs.get("answer") or "").lower()
    expected = (expectations.get("expected_response") or "").lower()
    if not expected:
        return 0.0
    keywords = [w for w in expected.split() if len(w) > 4]
    if not keywords:
        return 0.0 if not answer else 1.0
    matches = sum(1 for w in keywords if w in answer)
    return matches / len(keywords)


# COMMAND ----------

# MAGIC %md
# MAGIC ### LLM-as-a-judge
# MAGIC
# MAGIC `make_judge` builds a prompt-driven scorer. The instructions template uses Jinja-style placeholders
# MAGIC (`{{ inputs }}`, `{{ outputs }}`) that MLflow fills in per row.
# MAGIC
# MAGIC Constrain the output to a fixed set of words so MLflow can parse the response as a categorical
# MAGIC value. A prompt like "rate relevance 1-5" can sometimes return prose, which gets recorded as
# MAGIC `None` and silently breaks the metric. The yes/partial/no pattern below avoids that.

# COMMAND ----------

from typing import Literal
from mlflow.genai.judges import make_judge

relevance_judge = make_judge(
    name="relevance",
    instructions=(
        "You are evaluating whether an answer is relevant to a user's question.\n\n"
        "Question: {{ inputs }}\n"
        "Answer: {{ outputs }}\n\n"
        "Reply with exactly one word from this set: yes, partial, no.\n"
        "- yes: the answer directly addresses the question.\n"
        "- partial: the answer addresses some aspect but misses key parts.\n"
        "- no: the answer does not address the question or hallucinates.\n"
        "Reply with only the single word, no punctuation, no explanation."
    ),
    model="databricks:/databricks-claude-sonnet-4-6",
    feedback_value_type=Literal["yes", "partial", "no"],
)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Run the evaluation
# MAGIC
# MAGIC `predict_fn` receives the keys of `inputs` as keyword arguments. We pass them straight to
# MAGIC `answer_question` from the shared setup module, so the eval runs the real agent code, not a stub.

# COMMAND ----------


def run_agent(query: str) -> dict:
    return answer_question(
        query=query, session_id="eval-session", user_id="eval@example.com"
    )


results = mlflow.genai.evaluate(
    data=eval_dataset,
    predict_fn=run_agent,
    scorers=[answer_contains_expected_keyword, relevance_judge],
)

# COMMAND ----------

# MAGIC %md
# MAGIC ## What to verify
# MAGIC
# MAGIC 1. Open the **Experiments** left nav, find your experiment, click **Runs**
# MAGIC 2. Click the most recent run (auto-named like `tasteful-cub-576`)
# MAGIC 3. Look at **Metrics** - both `answer_contains_expected_keyword/mean` and judge metrics should
# MAGIC    appear (judge metrics may be a value distribution rather than a mean depending on output type)
# MAGIC 4. Click into individual traces. The `relevance` value should be `yes`, `partial`, or `no` -
# MAGIC    not `None`. If you see `None`, the judge prompt isn't constraining output enough.
# MAGIC
# MAGIC Continue to [`05_judge_alignment`]($./05_judge_alignment) to calibrate this judge against the SME labels from lesson 3.
