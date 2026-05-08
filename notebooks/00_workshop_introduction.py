# Databricks notebook source
# MAGIC %md-sandbox
# MAGIC # Build and evaluate GenAI agents on Databricks
# MAGIC
# MAGIC This workshop walks through the full MLflow GenAI evaluation surface on Databricks: tracing,
# MAGIC custom scorers, LLM-as-a-judge, human review labeling, scheduled production monitoring, and
# MAGIC OpenTelemetry traces in Unity Catalog.
# MAGIC
# MAGIC By the end you will have:
# MAGIC
# MAGIC - A small RAG agent emitting structured MLflow traces with session and user metadata
# MAGIC - In-app feedback and SME review labels logged as typed assessments on those traces
# MAGIC - An offline evaluation run with a custom code-based scorer and an LLM-as-a-judge
# MAGIC - Two scheduled scorers monitoring production traces with sampling
# MAGIC - A clear picture of how OpenTelemetry traces flow into Unity Catalog Delta tables and how to query them with SQL
# MAGIC
# MAGIC Plan on about 75 minutes if you run all seven lessons in order.

# COMMAND ----------

# MAGIC %md
# MAGIC ## Prerequisites
# MAGIC
# MAGIC - A Databricks workspace with **managed MLflow 3** (default on current runtimes)
# MAGIC - Network access to install `mlflow`, `databricks-sdk`, and `databricks-agents` from PyPI
# MAGIC - A Foundation Model serving endpoint that resolves. The lessons default to `databricks-claude-sonnet-4-6` - swap to whatever your workspace has if needed
# MAGIC - **Optional**: OTel + Traces in Unity Catalog enabled for lesson 7. The other lessons work without it.
# MAGIC - Either serverless compute or an interactive cluster on DBR 14.x or newer

# COMMAND ----------

# MAGIC %md
# MAGIC ## The flow
# MAGIC
# MAGIC The lessons build on each other. Run them in order the first time.
# MAGIC
# MAGIC ![Workshop architecture](./images/architecture.svg)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1/ Pre-flight check
# MAGIC
# MAGIC Confirms your workspace has every API surface the workshop needs. Eight checks, fast feedback.
# MAGIC Run this first; do not proceed if any check fails.
# MAGIC
# MAGIC Open [`01_pre_flight_check`]($./01_pre_flight_check)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2/ Build a traced RAG agent
# MAGIC
# MAGIC A small retrieval-augmented agent that emits hierarchical MLflow traces. Demonstrates `@mlflow.trace`,
# MAGIC `SpanType.RETRIEVER` / `SpanType.LLM`, and the `mlflow.update_current_trace` API for session and user
# MAGIC metadata that powers UI filter chips.
# MAGIC
# MAGIC Open [`02_agent_app`]($./02_agent_app)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3/ Capture human assessments
# MAGIC
# MAGIC Two annotation surfaces: in-app feedback (thumbs up/down logged via `mlflow.log_feedback`) and
# MAGIC the Review App for batch SME labeling. Both flow into the same trace assessment surface.
# MAGIC
# MAGIC Open [`03_capture_assessments`]($./03_capture_assessments)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 4/ Run an offline evaluation
# MAGIC
# MAGIC `mlflow.genai.evaluate` over a small dataset with a custom `@scorer` function and an LLM judge built
# MAGIC from `mlflow.genai.judges.make_judge`. Results show side-by-side trace comparisons in the eval UI.
# MAGIC
# MAGIC Open [`04_offline_evals`]($./04_offline_evals)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 5/ Calibrate the LLM judge against SME labels
# MAGIC
# MAGIC `judge.align(traces)` rewrites the judge's instructions to maximize agreement with paired
# MAGIC `HUMAN` + `LLM_JUDGE` assessments under the same name. Default optimizer is SIMBA;
# MAGIC GEPA and MemAlign also available. Returns a new judge object you register so production
# MAGIC monitoring uses the calibrated version. This is the closed loop: SMEs in lesson 3 produce
# MAGIC ground truth, lesson 4 runs the unaligned judge, lesson 5 aligns and reports the lift.
# MAGIC
# MAGIC Open [`05_judge_alignment`]($./05_judge_alignment)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 6/ Schedule scorers in production
# MAGIC
# MAGIC Same scorers run two ways: synchronously over a batch of existing traces (immediate scores in the UI),
# MAGIC and continuously over new traces via `scorer.register(...).start(sampling_config=...)`. Beta surface;
# MAGIC sampling controls cost. Picks up the aligned judge from lesson 5 automatically.
# MAGIC
# MAGIC Open [`06_production_monitoring`]($./06_production_monitoring)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 7/ Query traces in Unity Catalog
# MAGIC
# MAGIC OTel + Traces in UC stores the same trace data in a Delta table that you can query with plain SQL,
# MAGIC point Genie or AI/BI dashboards at, and govern with Unity Catalog. The lesson also walks through the
# MAGIC dual-export pattern for keeping an existing observability tool (e.g. Datadog) in the loop.
# MAGIC
# MAGIC Open [`07_otel_uc_integration`]($./07_otel_uc_integration)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Feature reference
# MAGIC
# MAGIC Each Databricks/MLflow capability used in the workshop, with a one-line role and a docs link.
# MAGIC
# MAGIC | Feature | Role in the workshop | Docs |
# MAGIC |---|---|---|
# MAGIC | MLflow tracing (`@mlflow.trace`) | Captures the agent's chain, retrieve, and generate steps as a span tree | [docs](https://docs.databricks.com/aws/en/mlflow3/genai/tracing/) |
# MAGIC | Trace metadata (`mlflow.update_current_trace`) | Sets `mlflow.trace.user` / `mlflow.trace.session` so the UI filter chips work | [docs](https://docs.databricks.com/aws/en/mlflow3/genai/tracing/track-users-sessions) |
# MAGIC | Feedback assessments (`mlflow.log_feedback`) | Logs typed end-user thumbs up/down on a trace | [docs](https://docs.databricks.com/aws/en/mlflow3/genai/getting-started/) |
# MAGIC | Review App labeling sessions | Lets SMEs batch-label existing traces against a typed schema | [docs](https://docs.databricks.com/aws/en/mlflow3/genai/human-feedback/concepts/review-app) |
# MAGIC | Custom code-based scorers (`@scorer`) | Any Python function becomes a scorer | [docs](https://docs.databricks.com/aws/en/mlflow3/genai/eval-monitor/custom-scorers) |
# MAGIC | LLM-as-a-judge (`make_judge`) | Prompt-based scorer that returns a categorical or numeric value | [docs](https://docs.databricks.com/aws/en/mlflow3/genai/eval-monitor/custom-judge/create-custom-judge) |
# MAGIC | Judge alignment (`judge.align`) | Optimizes the judge's instructions against paired SME labels | [docs](https://docs.databricks.com/aws/en/mlflow3/genai/eval-monitor/align-judges) |
# MAGIC | Offline evaluation (`mlflow.genai.evaluate`) | Runs scorers over a dataset and produces an eval run | [docs](https://docs.databricks.com/aws/en/mlflow3/genai/eval-monitor/) |
# MAGIC | Scheduled scorers (Beta) | Continuous scoring of production traces with sampling | [docs](https://docs.databricks.com/aws/en/mlflow3/genai/eval-monitor/run-scorer-in-prod) |
# MAGIC | OTel + Traces in Unity Catalog (Public Preview) | Trace data as a Delta table, queryable with SQL | [docs](https://docs.databricks.com/aws/en/mlflow3/genai/tracing/trace-unity-catalog) |
# MAGIC | Foundation Model APIs | Hosts the LLM the agent calls and the judge uses | [docs](https://docs.databricks.com/aws/en/machine-learning/foundation-model-apis/) |
# MAGIC | Databricks Asset Bundles | Versioned, deployable wrapper for the workshop notebooks | [docs](https://docs.databricks.com/aws/en/dev-tools/bundles/) |

# COMMAND ----------

# MAGIC %md
# MAGIC ## What's next
# MAGIC
# MAGIC When you finish:
# MAGIC
# MAGIC - Replace the toy corpus in `_resources/setup` with your own retrieval source (Vector Search index, your Delta table, etc.)
# MAGIC - Replace the eval dataset in lesson 4 with a labeled trace set from your domain
# MAGIC - Run lesson 5 with real SME labels from lesson 3 to actually calibrate the judge
# MAGIC - Tune the sampling rates in lesson 6 to your judge cost budget
# MAGIC - Configure OTel + Traces in UC in your workspace and wire the SQL queries in lesson 7 to your real catalog
# MAGIC
# MAGIC Open [`01_pre_flight_check`]($./01_pre_flight_check) when you're ready.
