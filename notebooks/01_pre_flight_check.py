# Databricks notebook source
# MAGIC %md
# MAGIC # Lesson 1 - Pre-flight check
# MAGIC
# MAGIC Confirms every API surface the workshop depends on. Eight checks, ~1 minute.

# COMMAND ----------

# MAGIC %md
# MAGIC ## Challenges Addressed
# MAGIC
# MAGIC 1. Does the workspace have managed MLflow 3 with the GenAI surface?
# MAGIC 2. Does the Foundation Model endpoint resolve from this notebook?
# MAGIC 3. Are the Review App and scheduled-scorer Python APIs importable?
# MAGIC
# MAGIC ## What is happening?
# MAGIC
# MAGIC Each cell exercises one capability the later lessons depend on. We catch the errors here so the
# MAGIC actual lessons run cleanly. If any check fails, fix the issue (usually a missing permission, an
# MAGIC old MLflow version, or a wrong endpoint name) before moving on.

# COMMAND ----------

# MAGIC %pip install -U -qqqq mlflow databricks-sdk databricks-agents
# MAGIC dbutils.library.restartPython()

# COMMAND ----------

import sys
import traceback
from typing import Callable

CHECKS_PASSED = 0
CHECKS_FAILED = 0
FAILURES: list[tuple[str, str]] = []


def check(name: str, fn: Callable[[], None]) -> None:
    global CHECKS_PASSED, CHECKS_FAILED
    try:
        fn()
        print(f"PASS {name}")
        CHECKS_PASSED += 1
    except Exception as e:
        msg = f"{type(e).__name__}: {e}"
        print(f"FAIL {name}")
        print(f"     {msg}")
        traceback.print_exc(limit=2, file=sys.stdout)
        FAILURES.append((name, msg))
        CHECKS_FAILED += 1


# COMMAND ----------

# MAGIC %md
# MAGIC ### 1. Workspace identity
# MAGIC
# MAGIC `WorkspaceClient` resolves your current user from the notebook's runtime credentials. We use
# MAGIC this in lesson 2 to build a per-user MLflow experiment path.

# COMMAND ----------


def check_workspace_client():
    from databricks.sdk import WorkspaceClient

    me = WorkspaceClient().current_user.me()
    assert me.user_name, "current_user.me returned no user_name"
    print(f"     user: {me.user_name}")


check("workspace_client_resolves", check_workspace_client)

# COMMAND ----------

# MAGIC %md
# MAGIC ### 2. MLflow experiment is reachable
# MAGIC
# MAGIC `mlflow.set_experiment(path)` creates the experiment under your user folder if missing. All
# MAGIC traces and eval runs in this workshop land here.

# COMMAND ----------


def check_mlflow_experiment():
    import mlflow
    from databricks.sdk import WorkspaceClient

    user = WorkspaceClient().current_user.me().user_name
    path = f"/Workspace/Users/{user}/mlflow_evals_workshop/agent_traces"
    mlflow.set_experiment(path)
    exp = mlflow.get_experiment_by_name(path)
    assert exp is not None, f"experiment not found at {path}"
    print(f"     experiment_id: {exp.experiment_id}")


check("mlflow_experiment_reachable", check_mlflow_experiment)

# COMMAND ----------

# MAGIC %md
# MAGIC ### 3. Foundation Model endpoint responds
# MAGIC
# MAGIC The agent in lesson 2 and the judge in lesson 4 both call this endpoint. If your workspace
# MAGIC doesn't have `databricks-claude-sonnet-4-6`, edit `ENDPOINT_NAME` below to one you do have
# MAGIC (the **Serving** left nav lists what's available).

# COMMAND ----------

ENDPOINT_NAME = "databricks-claude-sonnet-4-6"


def check_fm_endpoint():
    from databricks.sdk import WorkspaceClient
    from databricks.sdk.service.serving import ChatMessage, ChatMessageRole

    response = WorkspaceClient().serving_endpoints.query(
        name=ENDPOINT_NAME,
        messages=[
            ChatMessage(role=ChatMessageRole.USER, content="reply with the word OK")
        ],
        max_tokens=10,
    )
    choices = response.choices or []
    assert choices, "endpoint returned no choices"
    content = choices[0].message.content if choices[0].message else None
    assert content, "endpoint returned empty content"
    print(f"     endpoint reply: {content[:60]}")


check(f"fm_endpoint_{ENDPOINT_NAME}_responds", check_fm_endpoint)

# COMMAND ----------

# MAGIC %md
# MAGIC ### 4. Tracing decorator works
# MAGIC
# MAGIC `@mlflow.trace` produces a span on every call. Lesson 2 uses it on the agent's chain, retriever,
# MAGIC and generator functions to build the trace tree.

# COMMAND ----------


def check_trace_decorator():
    import mlflow
    from mlflow.entities import SpanType

    @mlflow.trace(span_type=SpanType.CHAIN)
    def smoke_test_chain(x: str) -> str:
        return f"echo: {x}"

    smoke_test_chain("preflight")


check("trace_decorator_works", check_trace_decorator)

# COMMAND ----------

# MAGIC %md
# MAGIC ### 5. GenAI judge and scorer modules import
# MAGIC
# MAGIC `make_judge` is the LLM-as-a-judge constructor. `@scorer` turns any Python function into a
# MAGIC pluggable scorer. Both ship with managed MLflow 3.

# COMMAND ----------


def check_make_judge():
    from mlflow.genai.judges import make_judge  # noqa: F401


def check_scorer_decorator():
    from mlflow.genai.scorers import scorer  # noqa: F401


check("make_judge_importable", check_make_judge)
check("scorer_decorator_importable", check_scorer_decorator)

# COMMAND ----------

# MAGIC %md
# MAGIC ### 6. Review App and scheduled-scorer APIs
# MAGIC
# MAGIC Review App labeling APIs live in `mlflow.genai.label_schemas` and `mlflow.genai.labeling`. The
# MAGIC scheduled-scorer surface lives in `mlflow.genai.scorers`. Both require the `databricks-agents`
# MAGIC package the pip cell above installs.

# COMMAND ----------


def check_review_app_imports():
    from mlflow.genai.label_schemas import (  # noqa: F401
        create_label_schema,
        InputCategorical,
        InputText,
    )
    from mlflow.genai.labeling import create_labeling_session  # noqa: F401


def check_production_monitoring_imports():
    from mlflow.genai.scorers import ScorerSamplingConfig  # noqa: F401


check("review_app_imports", check_review_app_imports)
check("production_monitoring_imports", check_production_monitoring_imports)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Result

# COMMAND ----------

print(f"\nPASSED: {CHECKS_PASSED}")
print(f"FAILED: {CHECKS_FAILED}")

if CHECKS_FAILED > 0:
    failure_summary = "; ".join(f"{n}: {m}" for n, m in FAILURES)
    raise RuntimeError(
        f"{CHECKS_FAILED} pre-flight check(s) failed -> {failure_summary}"
    )

# COMMAND ----------

# MAGIC %md
# MAGIC All eight checks passed. Continue to [`02_agent_app`]($./02_agent_app).
