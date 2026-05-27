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
# MAGIC - A serving endpoint hosting the agent with the canonical three-env-var trace-routing pattern, so AI Playground calls land in the same MLflow experiment as your offline evals
# MAGIC
# MAGIC Plan on about 90 minutes if you run all eight lessons in order. Lesson 8 adds endpoint
# MAGIC provisioning wait time on top of the others.

# COMMAND ----------

# MAGIC %md
# MAGIC ## What this workshop addresses
# MAGIC
# MAGIC ![The problem](./images/hd_problem.png)

# COMMAND ----------

# MAGIC %md
# MAGIC ## The three pillars
# MAGIC
# MAGIC Three principles tie every lesson together. Each capability you build in the workshop maps back to one of them.
# MAGIC
# MAGIC ![Three pillars](./images/hd_three_pillars.png)

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
# MAGIC `judge.align(traces, optimizer=...)` rewrites the judge's instructions to maximize agreement
# MAGIC with paired `HUMAN` + `LLM_JUDGE` assessments under the same name. Three optimizers ship in
# MAGIC `mlflow.genai.judges.optimizers`: **SIMBA** (the no-arg default, DSPy-based), **GEPA**
# MAGIC (LLM-driven reflection, stronger when SME rationales are rich), and **MemAlign**
# MAGIC (memory-augmented, experimental, MLflow team flagged as the planned future default on
# MAGIC 2026-05-13). Lesson 5 explicitly uses MemAlign to demonstrate the parameter pattern and
# MAGIC put the workshop on the same algorithm the MLflow team is moving the default toward.
# MAGIC Returns a new judge object you register so production monitoring uses the calibrated
# MAGIC version. This is the closed loop: SMEs in lesson 3 produce ground truth, lesson 4 runs
# MAGIC the unaligned judge, lesson 5 aligns and reports the lift.
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
# MAGIC ## 8/ Deploy the agent and route Playground traces back
# MAGIC
# MAGIC Wraps the agent as an `mlflow.pyfunc.ChatModel`, registers it in Unity Catalog, and creates a
# MAGIC Mosaic AI Model Serving endpoint. Every AI Playground or curl call to the endpoint writes a
# MAGIC trace into the same experiment lessons 2 to 6 read from. Closes the production loop: live
# MAGIC traffic feeds the same eval surface as your offline runs.
# MAGIC
# MAGIC The non-obvious part is three env vars on the served entity (`ENABLE_MLFLOW_TRACING`,
# MAGIC `MLFLOW_TRACKING_URI`, `MLFLOW_EXPERIMENT_ID`). Missing `MLFLOW_TRACKING_URI=databricks` is
# MAGIC the most common gotcha - the serving runtime falls back to a container-local file store and
# MAGIC traces never reach the experiment. The public docs page lists only the first and third;
# MAGIC the second is an empirically-observed requirement we verified on a fresh deploy 2026-05-18.
# MAGIC
# MAGIC Open [`08_deploy_agent`]($./08_deploy_agent)

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
# MAGIC | Mosaic AI Model Serving + `ChatModel` | Hosts the deployed agent endpoint that AI Playground talks to | [docs](https://docs.databricks.com/aws/en/machine-learning/model-serving/) |
# MAGIC | Serving-endpoint tracing env vars | `ENABLE_MLFLOW_TRACING` + `MLFLOW_TRACKING_URI` + `MLFLOW_EXPERIMENT_ID` route Playground / curl call traces back to the named experiment | [docs](https://docs.databricks.com/aws/en/mlflow3/genai/tracing/prod-tracing) |
# MAGIC | Canonical retriever-span schema | `page_content` + `metadata.doc_uri` renders retriever output as document cards in the MLflow trace UI | [docs](https://mlflow.org/docs/latest/genai/concepts/span/#retriever-spans) |
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
# MAGIC - Promote the lesson 8 endpoint to production-tier workload size, disable scale-to-zero, and wire UC Models aliases (`production` / `candidate`) for atomic blue/green deploys
# MAGIC
# MAGIC Open [`01_pre_flight_check`]($./01_pre_flight_check) when you're ready.
