# Databricks notebook source
# MAGIC %md
# MAGIC # Lesson 8 - Deploy the agent and route Playground traces back to MLflow
# MAGIC
# MAGIC Wraps the `answer_question` chain from lesson 2 as an MLflow `ChatModel`, registers it in
# MAGIC Unity Catalog, and creates a Mosaic AI Model Serving endpoint. Once the endpoint is `READY`
# MAGIC the agent is callable from AI Playground, from `curl`, or from any downstream service. Every
# MAGIC such call writes a trace into the same MLflow experiment lessons 2 to 7 read from, so the
# MAGIC loop closes: production traffic feeds the same eval and monitoring surface as your offline
# MAGIC traces.

# COMMAND ----------

# MAGIC %md
# MAGIC ## Challenges addressed
# MAGIC
# MAGIC 1. How does a real agent get from a notebook to a callable endpoint?
# MAGIC 2. How do you make AI Playground calls land in your eval pipeline?
# MAGIC 3. What auth + tracing wiring does the serving runtime need to write back to the workspace
# MAGIC    MLflow experiment?
# MAGIC
# MAGIC ## What is happening?
# MAGIC
# MAGIC `mlflow.pyfunc.ChatModel` is the Playground-compatible serving contract. Wrapping the agent
# MAGIC as a ChatModel lets Mosaic AI Model Serving host it like any other LLM endpoint. The
# MAGIC `@mlflow.trace` decorators on `answer_question` / `retrieve` / `generate` still fire inside
# MAGIC the serving container.
# MAGIC
# MAGIC Three environment variables on the served entity wire the trace stream back to the named
# MAGIC workspace experiment. Without these the serving runtime's tracing path is either off
# MAGIC (`ENABLE_MLFLOW_TRACING` defaults to false) or pointed at a container-local file store
# MAGIC (the default when `MLFLOW_TRACKING_URI` is unset), and traces never reach the experiment.
# MAGIC
# MAGIC Note on the third env var: the [public docs page](https://docs.databricks.com/aws/en/mlflow3/genai/tracing/prod-tracing)
# MAGIC lists only `ENABLE_MLFLOW_TRACING` and `MLFLOW_EXPERIMENT_ID`. `MLFLOW_TRACKING_URI=databricks`
# MAGIC is an empirically-observed requirement on serving runtimes where the default tracking URI
# MAGIC falls back to a container-local SQLite store. We verified this on a fresh deploy on
# MAGIC 2026-05-18: omitting the third var produced `RESOURCE_DOES_NOT_EXIST: Node ID 1 does not
# MAGIC exist.` in the serving logs and traces never appeared in the experiment.
# MAGIC
# MAGIC Reference: [Production tracing for MLflow GenAI](https://docs.databricks.com/aws/en/mlflow3/genai/tracing/prod-tracing).

# COMMAND ----------

# MAGIC %pip install -U -qqqq mlflow databricks-sdk databricks-agents
# MAGIC dbutils.library.restartPython()

# COMMAND ----------

# MAGIC %run ./_resources/setup

# COMMAND ----------

# MAGIC %md
# MAGIC ## Configuration
# MAGIC
# MAGIC Override `UC_CATALOG`, `UC_SCHEMA`, and `ENDPOINT_NAME` before running this on your own
# MAGIC workspace. The defaults assume you have `CREATE TABLE` on `main.default`.

# COMMAND ----------

import mlflow

UC_CATALOG = "main"
UC_SCHEMA = "default"
MODEL_NAME = f"{UC_CATALOG}.{UC_SCHEMA}.mlflow_evals_workshop_agent"
ENDPOINT_NAME = "mlflow-evals-workshop-agent"

# COMMAND ----------

# MAGIC %md
# MAGIC ### Wrap the agent as a ChatModel
# MAGIC
# MAGIC `ChatModel.predict` receives `ChatCompletionRequest` messages, pulls out the last user
# MAGIC message, calls our existing `answer_question` function, and returns a `ChatCompletionResponse`.
# MAGIC The hierarchical trace (`answer_question` -> `retrieve` -> `generate`) is recorded by the
# MAGIC decorators on the underlying functions.

# COMMAND ----------

from mlflow.pyfunc import ChatModel
from mlflow.types.llm import (
    ChatCompletionRequest,
    ChatCompletionResponse,
    ChatChoice,
    ChatMessage as MlflowChatMessage,
)


class WorkshopAgent(ChatModel):
    # Captured at log-time so the served container can re-establish the MLflow
    # experiment context inside predict() as a defensive fallback. The primary
    # mechanism is the MLFLOW_EXPERIMENT_ID env var set on the served entity
    # below; this in-predict() call is a safety net.
    _experiment_path = EXPERIMENT_PATH

    def predict(self, context, messages, params=None):
        mlflow.set_experiment(self._experiment_path)

        if isinstance(messages, ChatCompletionRequest):
            msg_list = messages.messages
        else:
            msg_list = messages

        user_message = ""
        for m in reversed(msg_list):
            role = m.role if hasattr(m, "role") else m.get("role")
            if role == "user":
                user_message = m.content if hasattr(m, "content") else m.get("content")
                break

        session_id = "playground"
        user_id = "playground-user"
        if params and isinstance(params, dict):
            session_id = params.get("session_id", session_id)
            user_id = params.get("user_id", user_id)

        result = answer_question(
            query=user_message,
            session_id=session_id,
            user_id=user_id,
        )

        return ChatCompletionResponse(
            choices=[
                ChatChoice(
                    index=0,
                    message=MlflowChatMessage(
                        role="assistant", content=result["answer"]
                    ),
                )
            ],
            model=ENDPOINT_NAME,
        )


# COMMAND ----------

# MAGIC %md
# MAGIC ### Log + register the model in Unity Catalog
# MAGIC
# MAGIC `resources=[DatabricksServingEndpoint(...)]` tells Model Serving to inject the credentials
# MAGIC the agent needs to call its Foundation Model endpoint from inside the serving container.
# MAGIC Without this declaration, the `WorkspaceClient()` call inside `generate()` fails with
# MAGIC "default auth: cannot configure default credentials" when the deployed model is queried.

# COMMAND ----------

from mlflow.models.resources import DatabricksServingEndpoint

mlflow.set_registry_uri("databricks-uc")

with mlflow.start_run(run_name="mlflow_evals_workshop_agent_v1") as run:
    logged = mlflow.pyfunc.log_model(
        name="agent",
        python_model=WorkshopAgent(),
        registered_model_name=MODEL_NAME,
        pip_requirements=[
            "mlflow",
            "databricks-sdk",
        ],
        resources=[
            DatabricksServingEndpoint(endpoint_name=FM_ENDPOINT),
        ],
    )
    print(f"Logged: {logged.model_uri}")
    print(f"Registered as: {MODEL_NAME}")

# COMMAND ----------

# MAGIC %md
# MAGIC ### Create the serving endpoint with the canonical trace-routing env vars
# MAGIC
# MAGIC Three environment variables together route AI Playground / curl / SDK calls into the named
# MAGIC workspace experiment:
# MAGIC
# MAGIC | Variable | Purpose |
# MAGIC |---|---|
# MAGIC | `ENABLE_MLFLOW_TRACING` | Flips the `@mlflow.trace` decorators ON inside the serving container. Default OFF. |
# MAGIC | `MLFLOW_TRACKING_URI` | Points the runtime at the Databricks-hosted MLflow store. Without `databricks` here the runtime creates a local SQLite MLflow DB inside the container and trace export silently fails with `RESOURCE_DOES_NOT_EXIST`. |
# MAGIC | `MLFLOW_EXPERIMENT_ID` | Routes captured traces to the named workspace experiment. |
# MAGIC
# MAGIC Missing `MLFLOW_TRACKING_URI=databricks` is the most common gotcha. Endpoint provisioning
# MAGIC takes 5 to 15 minutes for the first deployment. Once `READY` the endpoint appears in the AI
# MAGIC Playground dropdown.

# COMMAND ----------

from databricks.sdk import WorkspaceClient
from databricks.sdk.service.serving import (
    EndpointCoreConfigInput,
    ServedEntityInput,
)
from mlflow.tracking import MlflowClient

w = WorkspaceClient()

client_uc = MlflowClient(registry_uri="databricks-uc")
versions = client_uc.search_model_versions(f"name='{MODEL_NAME}'")
latest_version = max(int(v.version) for v in versions)
print(f"Latest version: {latest_version}")

exp_obj = mlflow.get_experiment_by_name(EXPERIMENT_PATH)
exp_id = exp_obj.experiment_id if exp_obj else ""
print(f"Routing serving-endpoint traces to experiment_id={exp_id}")

config = EndpointCoreConfigInput(
    name=ENDPOINT_NAME,
    served_entities=[
        ServedEntityInput(
            entity_name=MODEL_NAME,
            entity_version=str(latest_version),
            workload_size="Small",
            scale_to_zero_enabled=True,
            environment_vars={
                "ENABLE_MLFLOW_TRACING": "true",
                "MLFLOW_TRACKING_URI": "databricks",
                "MLFLOW_EXPERIMENT_ID": exp_id,
            },
        )
    ],
)

try:
    w.serving_endpoints.get(name=ENDPOINT_NAME)
    print(f"Endpoint exists. Updating to version {latest_version}.")
    w.serving_endpoints.update_config(
        name=ENDPOINT_NAME, served_entities=config.served_entities
    )
except Exception:
    print(f"Creating new endpoint: {ENDPOINT_NAME}.")
    w.serving_endpoints.create(name=ENDPOINT_NAME, config=config)

print()
print("Endpoint provisioning. State moves NOT_READY -> IN_PROGRESS -> READY.")
print(
    f"Once READY: open AI Playground -> select '{ENDPOINT_NAME}' -> chat with the agent."
)
print(
    "Every Playground call writes a trace into the experiment this lesson reads from."
)

# COMMAND ----------

# MAGIC %md
# MAGIC ## What to verify after the endpoint is READY
# MAGIC
# MAGIC 1. **Serving** tab in the workspace nav. Find your endpoint. State = READY.
# MAGIC 2. **AI Playground** in the left nav. Endpoint dropdown shows your endpoint.
# MAGIC 3. Ask the agent a question. Confirm a corpus-grounded answer comes back.
# MAGIC 4. Go back to **Experiments -> your experiment -> Traces**. The Playground call's trace
# MAGIC    should appear within ~30 seconds, with `session_id=playground` / `user_id=playground-user`
# MAGIC    if you used the default params.
# MAGIC 5. Click into the trace. The span tree shows `answer_question -> retrieve, generate`. The
# MAGIC    `retrieve` span renders its output as document cards (canonical retriever schema), not
# MAGIC    raw JSON.
# MAGIC
# MAGIC If the trace does not appear in the experiment, the most likely cause is a missing env var.
# MAGIC Verify all three are set:
# MAGIC ```
# MAGIC databricks serving-endpoints get <ENDPOINT_NAME> | grep -A 5 environment_vars
# MAGIC ```
# MAGIC
# MAGIC ## Production hardening checklist
# MAGIC
# MAGIC Before pointing traffic at this endpoint in your own workspace:
# MAGIC
# MAGIC - Replace the `WorkshopAgent` corpus with your real retrieval source (Vector Search index,
# MAGIC   external API, etc.).
# MAGIC - Set `workload_size` higher than `Small` and disable `scale_to_zero_enabled` for sustained
# MAGIC   traffic.
# MAGIC - Add an inference-table-backed capture for raw request/response logging (separate from
# MAGIC   MLflow tracing).
# MAGIC - Wire `judge.align`-calibrated scorers from lesson 5 as scheduled scorers (lesson 6) on the
# MAGIC   production experiment so quality regression is visible.
# MAGIC - Plan endpoint version cutovers via UC Models aliases (`production`, `candidate`) for atomic
# MAGIC   blue/green deploys.
# MAGIC - If your agent needs streaming responses, swap `mlflow.pyfunc.ChatModel` for
# MAGIC   `mlflow.pyfunc.ResponsesAgent`. Same Playground compatibility, additional streaming-event
# MAGIC   capture in the trace.
