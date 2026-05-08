# Databricks notebook source
# MAGIC %md
# MAGIC # Lesson 3 - Capture human assessments
# MAGIC
# MAGIC Two annotation surfaces: in-app feedback and the Review App. Both flow into the same trace
# MAGIC assessment surface, queryable later by SQL or the eval API.

# COMMAND ----------

# MAGIC %md
# MAGIC ## Challenges Addressed
# MAGIC
# MAGIC 1. How do you log an end-user thumbs up/down on a specific trace?
# MAGIC 2. How do you queue traces for batch SME labeling against a typed schema?
# MAGIC 3. How do you make this idempotent so re-runs don't duplicate assessments?
# MAGIC
# MAGIC ## What is happening?
# MAGIC
# MAGIC `mlflow.log_feedback(trace_id, name, value, source, rationale)` writes a typed assessment on a
# MAGIC trace. The `source` field carries provenance - HUMAN, LLM_JUDGE, or CODE - so downstream queries
# MAGIC can filter end-user feedback from automated judge scores.
# MAGIC
# MAGIC `mlflow.genai.label_schemas.create_label_schema` defines reusable label dimensions
# MAGIC (groundedness, relevance, etc.) with typed inputs. `mlflow.genai.labeling.create_labeling_session`
# MAGIC binds a list of traces to those schemas and assigns reviewers. Reviewers see the queued traces
# MAGIC in the Review App and submit labels through the UI.
# MAGIC
# MAGIC References: [feedback assessments](https://docs.databricks.com/aws/en/mlflow3/genai/getting-started/),
# MAGIC [labeling sessions](https://docs.databricks.com/aws/en/mlflow3/genai/human-feedback/concepts/review-app).

# COMMAND ----------

# MAGIC %pip install -U -qqqq mlflow databricks-sdk databricks-agents
# MAGIC dbutils.library.restartPython()

# COMMAND ----------

# MAGIC %run ./_resources/setup

# COMMAND ----------

# MAGIC %md
# MAGIC ### Find traces from lesson 2
# MAGIC
# MAGIC We filter to `trace.name = 'answer_question'` so we only label real agent traces, not the
# MAGIC eval-side traces lessons 4 and 5 will produce.

# COMMAND ----------

from mlflow.client import MlflowClient

client = MlflowClient()
experiment = mlflow.get_experiment_by_name(EXPERIMENT_PATH)
if experiment is None:
    raise RuntimeError(f"Experiment not found: {EXPERIMENT_PATH}. Run lesson 2 first.")

candidate_traces = client.search_traces(
    experiment_ids=[experiment.experiment_id],
    filter_string="trace.name = 'answer_question'",
    max_results=50,
)
print(f"Found {len(candidate_traces)} answer_question traces")

# COMMAND ----------

# MAGIC %md
# MAGIC ### Log in-app feedback
# MAGIC
# MAGIC In a real app, your feedback widget calls `mlflow.log_feedback(...)` whenever an end user clicks
# MAGIC thumbs up/down. Here we simulate that by deriving a stable rating from the trace_id hash so the
# MAGIC distribution is roughly 70/30 positive across re-runs. Real feedback would come from real users.
# MAGIC
# MAGIC The idempotency guard skips traces that already have an `end_user_rating` so re-running the
# MAGIC notebook doesn't pile on duplicate assessments.

# COMMAND ----------

import hashlib
from mlflow.entities import AssessmentSource


def has_end_user_rating(trace) -> bool:
    for a in trace.info.assessments or []:
        if a.name == "end_user_rating":
            return True
    return False


to_label = [t for t in candidate_traces if not has_end_user_rating(t)]
print(f"  {len(to_label)} need feedback (rest already labeled)")

POS_COMMENTS = [
    "Answer was clear and matched what I was looking for.",
    "Helpful, included the key term I needed.",
    "Concise and accurate.",
    "Good - linked the right concepts.",
]
NEG_COMMENTS = [
    "Missed the main point of the question.",
    "Too vague, did not answer specifically.",
    "Confused two concepts.",
    "Hallucinated content not in the source docs.",
]


def deterministic_thumbs(trace_id: str) -> tuple[bool, str]:
    h = int(hashlib.md5(trace_id.encode()).hexdigest()[:8], 16)
    is_positive = (h % 10) < 7
    pool = POS_COMMENTS if is_positive else NEG_COMMENTS
    return is_positive, pool[h % len(pool)]


for t in to_label:
    trace_id = t.info.trace_id
    is_positive, comment = deterministic_thumbs(trace_id)
    mlflow.log_feedback(
        trace_id=trace_id,
        name="end_user_rating",
        value=is_positive,
        source=AssessmentSource(
            source_type="HUMAN",
            source_id="end_user@example.com",
        ),
        rationale=comment,
    )

print(f"Logged feedback on {len(to_label)} traces")

# COMMAND ----------

# MAGIC %md
# MAGIC ### Define the label schema
# MAGIC
# MAGIC Three dimensions for the SME review pass:
# MAGIC
# MAGIC - `groundedness` - is the answer supported by the retrieved context (yes/no/partial)
# MAGIC - `relevance` - does the answer address the question (yes/no/partial)
# MAGIC - `rationale` - free-text reasoning, optional
# MAGIC
# MAGIC The try/except wrapper makes schema creation idempotent. Once a schema is referenced by an
# MAGIC active labeling session, it can't be removed, so we reuse the existing definition on re-run.

# COMMAND ----------

from mlflow.genai.label_schemas import (
    create_label_schema,
    InputCategorical,
    InputText,
)
from mlflow.genai.labeling import create_labeling_session


def _get_or_create_schema(name, type, title, input):
    try:
        return create_label_schema(name=name, type=type, title=title, input=input)
    except Exception as e:
        if "must be unique" in str(e) or "Duplicate" in str(e):
            return None
        raise


_get_or_create_schema(
    name="groundedness",
    type="feedback",
    title="Is the answer grounded in the retrieved documents?",
    input=InputCategorical(options=["yes", "no", "partial"]),
)

_get_or_create_schema(
    name="relevance",
    type="feedback",
    title="Is the answer relevant to the question?",
    input=InputCategorical(options=["yes", "no", "partial"]),
)

_get_or_create_schema(
    name="rationale",
    type="feedback",
    title="Optional reasoning",
    input=InputText(max_length=500),
)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Create a labeling session
# MAGIC
# MAGIC Session name is timestamped so re-runs don't collide. We assign the session to the current user
# MAGIC for this workshop; in a real workflow you'd assign one or more SME email addresses.

# COMMAND ----------

import time

session_name = f"mlflow_evals_workshop_labeling_session_{int(time.time())}"
session = create_labeling_session(
    name=session_name,
    label_schemas=["groundedness", "relevance", "rationale"],
    assigned_users=[USER],
)
session.add_traces(candidate_traces[:20])
print(f"Labeling session URL: {session.url}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## What to verify
# MAGIC
# MAGIC 1. Click the labeling session URL above. The Review App opens.
# MAGIC 2. You see 20 traces queued, with three label dimensions on the right.
# MAGIC 3. Submit a label or two on the first trace. Click Save.
# MAGIC 4. Switch back to the experiment Traces tab. The trace you just labeled now has additional
# MAGIC    assessments visible.
# MAGIC
# MAGIC Continue to [`04_offline_evals`]($./04_offline_evals).
