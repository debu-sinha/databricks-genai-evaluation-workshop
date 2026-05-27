# Databricks notebook source
# MAGIC %md
# MAGIC # Lesson 6 - Schedule scorers in production
# MAGIC
# MAGIC Run the same scorers from lesson 4 against existing traces synchronously, then register them
# MAGIC for continuous scoring of new traces. The `relevance` scorer registered here picks up the
# MAGIC aligned version from lesson 5 if you ran that lesson.

# COMMAND ----------

# MAGIC %md
# MAGIC ## Challenges Addressed
# MAGIC
# MAGIC 1. How do you score a batch of existing production traces immediately?
# MAGIC 2. How do you keep scoring continuously as new traces arrive?
# MAGIC 3. How do you bound the cost of LLM judges across millions of traces?
# MAGIC
# MAGIC ## What is happening?
# MAGIC
# MAGIC Two complementary modes for the same scorer code:
# MAGIC
# MAGIC - **Synchronous (Part A)**: `mlflow.genai.evaluate(data=traces, scorers=[...])` runs every
# MAGIC   scorer against the supplied traces and produces an immediate eval run. Use this to backfill
# MAGIC   scores on existing data or to score a one-off batch.
# MAGIC - **Scheduled (Part B)**: `scorer.register(name).start(sampling_config=...)` registers the
# MAGIC   scorer with the experiment so it runs continuously against new traces. Sampling rate
# MAGIC   controls cost. The scorer fires asynchronously, with ~15-20 minute initial latency before
# MAGIC   results show up in the Monitoring tab.
# MAGIC
# MAGIC Scheduled monitoring is currently **Beta**. Hard cap: 20 scorers per experiment. Recommended
# MAGIC sampling: 1.0 for cheap safety scorers, 0.05-0.20 for expensive LLM judges.
# MAGIC
# MAGIC References: [run scorers in production](https://docs.databricks.com/aws/en/mlflow3/genai/eval-monitor/run-scorer-in-prod).
# MAGIC
# MAGIC ![Production monitoring: scheduled scorers](../images/hd_production_monitoring_concept.png)

# COMMAND ----------

# MAGIC %pip install -U -qqqq mlflow databricks-sdk databricks-agents
# MAGIC dbutils.library.restartPython()

# COMMAND ----------

# MAGIC %run ./_resources/setup

# COMMAND ----------

# MAGIC %md
# MAGIC ### Load the scorers from `workshop_scorers.py`
# MAGIC
# MAGIC The cheap safety scorer runs at 100% sampling, the LLM judge at 10% to keep cost bounded.
# MAGIC
# MAGIC Scorers used by the scheduled-scorer subsystem must be importable as a real Python module,
# MAGIC not defined inside a notebook. Functions defined inside a notebook get `__module__` set to
# MAGIC `__main__`, which the scheduled-scorer remote workers can't reconstruct, and registration
# MAGIC silently fails to fire. Putting them in `workshop_scorers.py` at the repo root makes the
# MAGIC import path stable.

# COMMAND ----------

import sys
import os
from mlflow.genai.scorers import ScorerSamplingConfig, get_scorer

# Databricks Repos puts the repo root on sys.path automatically when the notebook
# runs inside a Repo. For Workspace-imported notebooks we add it explicitly.
_repo_root = os.path.abspath(os.path.join(os.getcwd(), ".."))
if _repo_root not in sys.path:
    sys.path.insert(0, _repo_root)

from workshop_scorers import answer_non_empty, make_relevance_judge

# COMMAND ----------

# MAGIC %md
# MAGIC ### Pick up the aligned `relevance` judge from lesson 5 if it's there
# MAGIC
# MAGIC `get_scorer(name="relevance")` retrieves the version `aligned_judge.register(...)` saved at
# MAGIC the end of lesson 5. If lesson 5 has not yet run on this experiment, fall back to a fresh
# MAGIC unaligned judge from the workshop_scorers module so the rest of this lesson still works.

# COMMAND ----------

try:
    relevance_judge = get_scorer(name="relevance")
    print(
        f"Picked up registered '{relevance_judge.name}' judge (aligned version from lesson 5)."
    )
except Exception as e:
    if "not found" in str(e).lower() or "does not exist" in str(e).lower():
        relevance_judge = make_relevance_judge()
        print(
            "No registered 'relevance' judge found. Using fresh judge from workshop_scorers."
        )
        print(
            "Run lesson 5 to align the judge against SME labels before re-running this notebook."
        )
    else:
        raise

# COMMAND ----------

# MAGIC %md
# MAGIC ## Part A - Synchronous score over existing traces
# MAGIC
# MAGIC Run scorers directly via `mlflow.genai.evaluate` on the existing `answer_question` traces from
# MAGIC lesson 2. Same scorer code, immediate output, scores land in the eval UI in seconds.

# COMMAND ----------

from mlflow.client import MlflowClient

client = MlflowClient()
exp = mlflow.get_experiment_by_name(EXPERIMENT_PATH)

production_traces = client.search_traces(
    experiment_ids=[exp.experiment_id],
    filter_string="trace.name = 'answer_question'",
    max_results=10,
)
print(f"Scoring {len(production_traces)} traces")

scoring_results = mlflow.genai.evaluate(
    data=production_traces,
    scorers=[answer_non_empty, relevance_judge],
)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Part B - Register scheduled scorers
# MAGIC
# MAGIC `register(name)` records the scorer with the experiment. `start(sampling_config=...)` activates
# MAGIC continuous scoring. The scheduled-scorer subsystem picks up new traces as they arrive and
# MAGIC writes assessments back to them at the configured sampling rate.
# MAGIC
# MAGIC The try/except wrapper makes registration idempotent across re-runs. If the scorer is already
# MAGIC registered, leave it running rather than treating "already registered" as an error.

# COMMAND ----------


def _register_or_skip(scorer_obj, name, sampling_rate):
    try:
        registered = scorer_obj.register(name=name)
        registered.start(
            sampling_config=ScorerSamplingConfig(sample_rate=sampling_rate)
        )
        print(f"  registered + started '{name}' at sampling={sampling_rate}")
    except Exception as e:
        if "already been registered" in str(e):
            print(f"  '{name}' already registered, leaving running")
        else:
            raise


_register_or_skip(answer_non_empty, "answer_non_empty", 1.0)
_register_or_skip(relevance_judge, "relevance", 0.10)

# COMMAND ----------

# MAGIC %md
# MAGIC ## What to verify
# MAGIC
# MAGIC 1. Part A: Open the **Experiments** left nav -> **Runs** tab. The newest run has scores from
# MAGIC    `answer_non_empty` and `relevance` on every trace it processed.
# MAGIC 2. Part B: Open the **Monitoring** tab on the experiment. Both scorers appear with the correct
# MAGIC    sample rates (`answer_non_empty` 100%, `relevance` 10%).
# MAGIC 3. Scheduled scorer execution has 15-20 minute initial processing latency. The first results
# MAGIC    show up after that window; you don't need to watch for them now.
# MAGIC
# MAGIC Continue to [`07_otel_uc_integration`]($./07_otel_uc_integration).
