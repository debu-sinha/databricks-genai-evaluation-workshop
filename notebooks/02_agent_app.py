# Databricks notebook source
# MAGIC %md
# MAGIC # Lesson 2 - Build a traced RAG agent
# MAGIC
# MAGIC A small retrieval-augmented agent that emits hierarchical MLflow traces. The point is the
# MAGIC platform around the agent, not the agent itself.

# COMMAND ----------

# MAGIC %md
# MAGIC ## What is happening?
# MAGIC
# MAGIC The agent has three steps - top-level chain, retriever, generator - each decorated with
# MAGIC `@mlflow.trace`. MLflow auto-builds the span tree from the call graph. Inside the chain,
# MAGIC `mlflow.update_current_trace(metadata=...)` writes canonical session and user fields that the
# MAGIC MLflow UI uses to drive its filter chips.
# MAGIC
# MAGIC We then call the agent on a query bank to populate the experiment with ~23 traces under three
# MAGIC simulated users. Lessons 3 to 7 read from this trace surface, and lesson 8 deploys an
# MAGIC endpoint that writes more traces back into it.
# MAGIC
# MAGIC Reference: [MLflow tracing](https://docs.databricks.com/aws/en/mlflow3/genai/tracing/) and
# MAGIC [user/session metadata](https://docs.databricks.com/aws/en/mlflow3/genai/tracing/track-users-sessions).

# COMMAND ----------

# MAGIC %pip install -U -qqqq mlflow databricks-sdk databricks-agents
# MAGIC dbutils.library.restartPython()

# COMMAND ----------

# MAGIC %run ./_resources/setup

# COMMAND ----------

# MAGIC %md
# MAGIC ### Run a single query
# MAGIC
# MAGIC One invocation produces one trace. Open the experiment Traces tab afterward and click into the
# MAGIC trace to see the span tree (chain -> retrieve -> generate) and the metadata block on the right.

# COMMAND ----------

result = answer_question(
    query="How are MLflow traces structured?",
    session_id="demo-session-001",
    user_id="alice@example.com",
)
print(result["answer"])

# COMMAND ----------

# MAGIC %md
# MAGIC ### Populate the experiment
# MAGIC
# MAGIC The remaining lessons read from a populated experiment. We run the agent across three simulated
# MAGIC users on shuffled queries from the demo bank. Sleep 0.3s between calls to be polite to the
# MAGIC Foundation Model endpoint.
# MAGIC
# MAGIC **Re-run note.** If you have already populated this experiment in a previous notebook
# MAGIC session, the cell below may report "skipped N duplicate trace IDs". That is safe but
# MAGIC produces fewer fresh traces, which can starve lesson 5 (needs >= 10 paired traces). The
# MAGIC diagnostic cell directly below this one prints the current trace count. If it is already
# MAGIC well above 20 you can skip the populate cell. If you want a clean slate, wipe traces from
# MAGIC the experiment first (Experiments -> Traces tab -> select all -> Delete) and then run.

# COMMAND ----------

from mlflow.client import MlflowClient

_client = MlflowClient()
_exp = mlflow.get_experiment_by_name(EXPERIMENT_PATH)
_existing = _client.search_traces(
    experiment_ids=[_exp.experiment_id],
    max_results=100,
)
print(f"Existing traces in experiment: {len(_existing)}")
if len(_existing) >= 20:
    print(
        "  >= 20 already. You can skip the populate cell below and continue to lesson 3."
    )
else:
    print("  < 20. Run the populate cell below to add ~23 more.")

# COMMAND ----------

import random
import time

USERS = [
    ("alice@example.com", "session-alice-001"),
    ("alice@example.com", "session-alice-002"),
    ("bob@example.com", "session-bob-001"),
    ("batch@example.com", "session-batch-001"),
]

# No random.seed here on purpose. Seeding makes the call ordering deterministic
# across runs, and on serverless tracing we have seen that determinism cause
# trace ID collisions on re-runs ("a trace with ID ... already exists"). Each
# run now shuffles freshly so the trace surface is unique across re-runs.
shuffled = DEMO_QUERIES.copy()
random.shuffle(shuffled)

ok = 0
duplicates = 0
errors = 0
for query in shuffled:
    user_id, session_id = random.choice(USERS)
    try:
        answer_question(query=query, session_id=session_id, user_id=user_id)
        ok += 1
    except Exception as e:
        msg = str(e)
        if "already exists" in msg.lower():
            duplicates += 1
        else:
            errors += 1
            print(f"  iter failed: {type(e).__name__}: {msg[:200]}")
    time.sleep(0.3)

print(f"populated {ok}/{len(shuffled)} traces")
if duplicates:
    print(
        f"skipped {duplicates} duplicate trace IDs (safe on re-run, expected on second invocation of this cell)"
    )
if errors:
    print(f"saw {errors} non-duplicate errors above")

# COMMAND ----------

# MAGIC %md
# MAGIC ## What to verify
# MAGIC
# MAGIC 1. Open the **Experiments** left nav, find `agent_traces`
# MAGIC 2. Click the **Traces** tab. You should see ~23 traces named `answer_question`
# MAGIC 3. Click any trace. The span tree shows `answer_question` (CHAIN) -> `retrieve` (RETRIEVER) -> `generate` (LLM)
# MAGIC 4. In the Metadata panel, find `mlflow.trace.user` and `mlflow.trace.session`
# MAGIC 5. Use the **Filter** chip above the trace list to filter by user (`alice@example.com`). The list narrows.
# MAGIC
# MAGIC Continue to [`03_capture_assessments`]($./03_capture_assessments).
