# Databricks notebook source
# MAGIC %md
# MAGIC # Lesson 6 - Query traces in Unity Catalog
# MAGIC
# MAGIC OTel + Traces in Unity Catalog stores the same trace data in a Delta table that you can query
# MAGIC with plain SQL, point Genie or AI/BI dashboards at, and govern with Unity Catalog ACLs.

# COMMAND ----------

# MAGIC %md
# MAGIC ## Challenges Addressed
# MAGIC
# MAGIC 1. How do you query trace data with SQL instead of the MLflow API?
# MAGIC 2. How do existing observability pipelines (Datadog, Honeycomb, etc.) coexist with Databricks tracing?
# MAGIC 3. What can your data team do once trace data is just another Delta table?
# MAGIC
# MAGIC ## What is happening?
# MAGIC
# MAGIC When OTel + Traces in Unity Catalog is enabled for the workspace, MLflow writes every trace
# MAGIC into a Delta table inside Unity Catalog. The MLflow API still works the same way, but the
# MAGIC same data is also readable via:
# MAGIC
# MAGIC - SQL warehouses (any SQL editor, AI/BI dashboards, Genie spaces)
# MAGIC - Lakehouse Federation
# MAGIC - Spark from notebooks (`spark.table(...)`)
# MAGIC
# MAGIC The Delta table sits inside whatever catalog and schema your account team configured for trace
# MAGIC storage. Permissions follow Unity Catalog as for any other table.
# MAGIC
# MAGIC OTel + Traces in UC is currently **Public Preview**. Operational caveats: 200 traces/sec per
# MAGIC workspace, 100 MB/sec per table out of the box, no per-trace delete API. See the docs for
# MAGIC current limits and region availability.
# MAGIC
# MAGIC References: [Traces in Unity Catalog](https://docs.databricks.com/aws/en/mlflow3/genai/tracing/trace-unity-catalog),
# MAGIC [OTel span attributes](https://docs.databricks.com/aws/en/mlflow3/genai/tracing/third-party/otel-span-attributes).

# COMMAND ----------

# MAGIC %pip install -U -qqqq mlflow databricks-sdk databricks-agents
# MAGIC dbutils.library.restartPython()

# COMMAND ----------

# MAGIC %run ./_resources/setup

# COMMAND ----------

# MAGIC %md
# MAGIC ### List recent traces via the MLflow client
# MAGIC
# MAGIC This works in any workspace, OTel-in-UC enabled or not. Use it as a sanity check that the
# MAGIC experiment has data.

# COMMAND ----------

from mlflow.client import MlflowClient

client = MlflowClient()
experiment = mlflow.get_experiment_by_name(EXPERIMENT_PATH)

if experiment:
    traces = client.search_traces(
        experiment_ids=[experiment.experiment_id], max_results=10
    )
    print(f"Recent traces: {len(traces)}")
    for t in traces[:5]:
        print(f"  {t.info.trace_id} | {t.info.request_time}")

# COMMAND ----------

# MAGIC %md
# MAGIC ### Same data, SQL access
# MAGIC
# MAGIC Once OTel + Traces in UC is provisioned, the trace surface lives at
# MAGIC `<catalog>.<schema>.traces`. The schema is documented in the [trace storage reference](https://docs.databricks.com/aws/en/mlflow3/genai/tracing/trace-unity-catalog).
# MAGIC
# MAGIC The cell below is a SQL example you can run in the **SQL Editor** or in a notebook with a
# MAGIC `%sql` magic. Replace `<catalog>` and `<schema>` with whatever your workspace is configured for.
# MAGIC
# MAGIC ```sql
# MAGIC SELECT
# MAGIC   trace_id,
# MAGIC   experiment_id,
# MAGIC   timestamp_ms,
# MAGIC   tags['mlflow.traceName'] AS trace_name,
# MAGIC   request_metadata['mlflow.trace.user'] AS mlflow_user
# MAGIC FROM <catalog>.<schema>.traces
# MAGIC WHERE timestamp_ms > date_sub(current_timestamp(), 1)
# MAGIC LIMIT 20
# MAGIC ```

# COMMAND ----------

# MAGIC %md
# MAGIC ### What this unlocks
# MAGIC
# MAGIC Once trace data is in a Delta table, every Databricks SQL surface works against it:
# MAGIC
# MAGIC - **Genie spaces** answer natural-language questions like "which users had the most thumbs-down responses last week?"
# MAGIC - **AI/BI Dashboards** chart trace volume, score distributions, and latency over time
# MAGIC - **Lakehouse Federation** lets your data team join traces with order tables, support tickets, etc.
# MAGIC - **Unity Catalog audit and lineage** apply with no extra setup

# COMMAND ----------

# MAGIC %md
# MAGIC ### Coexisting with another observability tool
# MAGIC
# MAGIC The supported pattern for keeping a separate observability stack (Datadog, Honeycomb, Grafana
# MAGIC Cloud) in the loop is **dual-export from your application**. Configure your app's OpenTelemetry
# MAGIC SDK with two exporters - one pointing at Databricks, one at your existing pipeline.
# MAGIC
# MAGIC ![Dual-export architecture](./images/architecture.svg)
# MAGIC
# MAGIC Routing through Datadog (or any other APM) as a forwarder into Databricks is **not** a
# MAGIC documented integration today. The third-party tracing integrations index lists OpenTelemetry,
# MAGIC LangChain, LangGraph, and a small set of inference SDKs - no APM forwarders.

# COMMAND ----------

# MAGIC %md
# MAGIC ## What to verify
# MAGIC
# MAGIC 1. The MLflow client query above prints recent traces.
# MAGIC 2. If your workspace has OTel + Traces in UC, run the SQL example in a SQL editor against
# MAGIC    your catalog/schema and confirm the same trace IDs come back.
# MAGIC 3. Open the architecture diagram in `./images/architecture.svg` for a visual of the full data flow.
# MAGIC
# MAGIC You've finished the workshop. Next steps in the [introduction]($./00_workshop_introduction).
