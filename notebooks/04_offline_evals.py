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
# MAGIC ![Offline evaluation concept](../images/hd_offline_evals_concept.png)
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
# MAGIC
# MAGIC ![Two scorer shapes](../images/hd_scorer_pattern.png)

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

# Print a direct link to the eval run. The Evaluation runs tab in some
# workspaces has a sticky user-level filter (e.g. `params.model = "tree"`)
# inherited from the AutoML / forecasting onboarding flow that hides freshly
# created runs. Clicking this URL bypasses the filter and lands you on the run.
import mlflow as _mlflow

_exp = _mlflow.get_experiment_by_name(EXPERIMENT_PATH)
_recent = _mlflow.search_runs(
    experiment_ids=[_exp.experiment_id],
    max_results=1,
    order_by=["start_time DESC"],
)
if not _recent.empty:
    _run_id = _recent.iloc[0]["run_id"]
    # Host detection works in both interactive and job contexts. browserHostName()
    # returns None in job runs (no browser), so fall back to WorkspaceClient.
    _host = None
    try:
        from databricks.sdk import WorkspaceClient as _WC

        _host = _WC().config.host.replace("https://", "").replace("http://", "")
    except Exception:
        pass
    if _host:
        print("Eval run created. Open directly:")
        print(f"  https://{_host}/ml/experiments/{_exp.experiment_id}/runs/{_run_id}")
    else:
        print(f"Eval run created. run_id={_run_id}, experiment_id={_exp.experiment_id}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## What to verify
# MAGIC
# MAGIC 1. Click the direct eval run URL printed above. (If you go via the **Experiments** -> **Evaluation runs** tab and don't see your run, check the filter row at the top - some workspaces have a sticky `params.model = "tree"` filter from the AutoML onboarding that hides new runs. Click the X on the filter chip to clear it.)
# MAGIC 2. Look at **Metrics** - both `answer_contains_expected_keyword/mean` and judge metrics should
# MAGIC    appear (judge metrics may be a value distribution rather than a mean depending on output type)
# MAGIC 3. Click into individual traces. The `relevance` value should be `yes`, `partial`, or `no` -
# MAGIC    not `None`. If you see `None`, the judge prompt isn't constraining output enough.

# COMMAND ----------

# MAGIC %md
# MAGIC ## Extending the pattern: evaluating an AI Functions batch pipeline
# MAGIC
# MAGIC The same eval surface works for batch pipelines that use Databricks AI Functions
# MAGIC (`ai_classify`, `ai_extract`, `ai_query`, `ai_summarize`, etc.). An AI Function call returns
# MAGIC a DataFrame you can use directly, or persist to a Unity Catalog Delta table via
# MAGIC `CREATE TABLE AS SELECT` or `.write.saveAsTable(...)`. Either way, the result is the eval
# MAGIC dataset: pass it to `mlflow.genai.evaluate(data=df, scorers=[...])` and judges plus code
# MAGIC scorers run against it the same way they ran against `eval_dataset` above.
# MAGIC
# MAGIC No agent code in this section. The pipeline is pure SQL plus the same scorer pattern.
# MAGIC
# MAGIC References:
# MAGIC [ai_classify](https://docs.databricks.com/aws/en/sql/language-manual/functions/ai_classify),
# MAGIC [AI Functions overview](https://docs.databricks.com/aws/en/large-language-models/ai-functions).

# COMMAND ----------

# Build a small input DataFrame to classify. In a real batch pipeline this is an existing UC table.
import pandas as pd

classify_inputs = pd.DataFrame(
    [
        {"id": "t1", "text": "Customer cannot complete checkout due to payment error."},
        {"id": "t2", "text": "The mobile app crashes when I open the settings screen."},
        {"id": "t3", "text": "I never received the package I ordered last week."},
        {"id": "t4", "text": "How do I reset my password?"},
    ]
)
spark.createDataFrame(classify_inputs).createOrReplaceTempView("classify_inputs")

# Run ai_classify via SQL. v2 takes a JSON string of label name -> description and returns
# VARIANT with `response` (array of labels) and `error_message`. The function is callable
# from any SQL surface on Databricks (notebook %sql, SQL warehouses, Workflows, Lakeflow).
classified = spark.sql(
    """
    SELECT
      id,
      text,
      ai_classify(
        text,
        '{"billing_error":"Payment or checkout failures","app_bug":"Software defects in the app","logistics":"Shipping or delivery issues","account_help":"Account, login, or password support"}'
      ) AS prediction
    FROM classify_inputs
    """
)
display(classified)

# COMMAND ----------

# Reshape the AI Functions output into the format mlflow.genai.evaluate expects:
#  - `inputs`  dict, what went into the classifier
#  - `outputs` dict, what the classifier produced
#  - `expectations` dict, optional ground truth for scoring
#
# In a real workflow the ground-truth labels come from a labeled UC table. Here they're inline.

import json

GROUND_TRUTH = {
    "t1": "billing_error",
    "t2": "app_bug",
    "t3": "logistics",
    "t4": "account_help",
}


def _extract_label(prediction):
    """v2 ai_classify returns {"response": ["billing_error"], "error_message": null}."""
    if prediction is None:
        return None
    if isinstance(prediction, str):
        prediction = json.loads(prediction)
    resp = prediction.get("response") if isinstance(prediction, dict) else None
    if isinstance(resp, list) and resp:
        return resp[0]
    return None


ai_eval_dataset = pd.DataFrame(
    [
        {
            "inputs": {"text": row["text"]},
            "outputs": {"label": _extract_label(row["prediction"])},
            "expectations": {"expected_label": GROUND_TRUTH[row["id"]]},
        }
        for _, row in classified.toPandas().iterrows()
    ]
)

# COMMAND ----------

# A simple code-based scorer for label correctness. Same @scorer decorator pattern as the
# `answer_contains_expected_keyword` scorer earlier in this notebook.
from mlflow.genai.scorers import scorer as ai_scorer


@ai_scorer
def label_exact_match(outputs: dict, expectations: dict) -> float:
    predicted = (outputs.get("label") or "").strip()
    expected = (expectations.get("expected_label") or "").strip()
    return 1.0 if predicted == expected else 0.0


# Run the evaluation against the pre-computed AI Functions output. No predict_fn here -
# the outputs are already populated from the ai_classify call. mlflow.genai.evaluate
# accepts pre-computed inputs+outputs DataFrames and runs scorers against them.
ai_results = mlflow.genai.evaluate(
    data=ai_eval_dataset,
    scorers=[label_exact_match],
)

# COMMAND ----------

# MAGIC %md
# MAGIC ### What just happened
# MAGIC
# MAGIC The eval surface did not care that the outputs came from `ai_classify` instead of from a
# MAGIC traced RAG agent. Same `mlflow.genai.evaluate` call, same Evaluation UI, same scorer pattern.
# MAGIC A prompt-based `make_judge` scorer would slot into the `scorers=[...]` list alongside
# MAGIC `label_exact_match` the same way it slotted in earlier in this notebook.
# MAGIC
# MAGIC Production shape:
# MAGIC
# MAGIC 1. Schedule the `ai_classify` SQL as a Lakeflow Spark Declarative Pipeline or Workflows job
# MAGIC    that writes to a Delta table.
# MAGIC 2. Schedule a follow-up notebook that reads the table and runs `mlflow.genai.evaluate(...)`.
# MAGIC 3. The eval surface ends up in the same Evaluation UI as the rest of the workshop's evals.
# MAGIC
# MAGIC The same pattern applies to `ai_extract` (returns structured fields), `ai_query` (any LLM
# MAGIC call), `ai_summarize`, `ai_translate`, etc. The output table is the eval dataset.
# MAGIC
# MAGIC Continue to [`05_judge_alignment`]($./05_judge_alignment) to calibrate this judge against the SME labels from lesson 3.
