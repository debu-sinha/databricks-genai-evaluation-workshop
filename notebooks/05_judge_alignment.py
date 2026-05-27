# Databricks notebook source
# MAGIC %md
# MAGIC # Lesson 5 - Calibrate the LLM judge against SME labels
# MAGIC
# MAGIC The `relevance` judge from lesson 4 captures one engineer's intuition about what relevance means.
# MAGIC SMEs often have a stricter standard, or care about edge cases the prompt missed. This lesson closes the loop: the SME labels
# MAGIC collected via the Review App in lesson 3 become the ground truth, and `judge.align()` rewrites
# MAGIC the judge's instructions to match SME judgment.
# MAGIC
# MAGIC This notebook depends on Databricks runtime built-ins (`display`, the `%run` magic, the `dbutils` global). It will
# MAGIC `NameError` if run outside a Databricks workspace.

# COMMAND ----------

# MAGIC %md
# MAGIC ## Two questions this lesson answers
# MAGIC
# MAGIC - Whether your LLM judge actually agrees with your SMEs (vs. you assuming it does because the prompt sounds reasonable)
# MAGIC - Once you measure disagreement, how to systematically rewrite the judge prompt instead of guessing
# MAGIC
# MAGIC ## What is happening?
# MAGIC
# MAGIC Three steps:
# MAGIC
# MAGIC 1. **Pair assessments by trace.** Every trace that has both an `LLM_JUDGE` assessment named
# MAGIC    `relevance` (from lesson 4) and a `HUMAN` assessment named `relevance` (from lesson 3) is
# MAGIC    a paired example. The names must match exactly.
# MAGIC 2. **Run `judge.align(paired, optimizer=MemAlignOptimizer(...))`.** MLflow runs the
# MAGIC    optimizer over the judge's instruction string to maximize agreement with the human
# MAGIC    labels. This lesson explicitly uses MemAlign (memory-augmented, currently
# MAGIC    experimental, MLflow team flagged as the planned future default on 2026-05-13).
# MAGIC    The no-arg `judge.align(paired)` form would route to SIMBA (the current documented
# MAGIC    default); GEPA is available for stronger LLM-reflection alignment. Returns a new
# MAGIC    judge object; the original is untouched.
# MAGIC 3. **Register the aligned judge.** Save it as a named scorer so lesson 6's production
# MAGIC    monitoring uses the calibrated version instead of the original.
# MAGIC
# MAGIC ### What this needs that the prior lessons may not have produced
# MAGIC
# MAGIC `align()` requires at least 10 traces with paired HUMAN + LLM_JUDGE assessments under the
# MAGIC same name. In a real workflow the human side comes from SMEs filling out the labeling session
# MAGIC in the Review App. For workshop reproducibility this notebook will fall back to logging
# MAGIC synthetic HUMAN labels if no real ones exist yet, with the rule clearly stated below.
# MAGIC
# MAGIC ![Closed loop: SME labels improve the judge](../images/hd_loop_diagram.png)

# COMMAND ----------

# MAGIC %pip install -U -qqqq mlflow databricks-sdk databricks-agents dspy
# MAGIC dbutils.library.restartPython()

# COMMAND ----------

# MAGIC %md
# MAGIC ### Why `dspy` is in the install line above
# MAGIC
# MAGIC All three `judge.align` optimizers (SIMBA, GEPA, MemAlign) depend on DSPy. The
# MAGIC [MemAlign docs](https://mlflow.org/docs/latest/genai/eval-monitor/scorers/llm-judge/memalign/)
# MAGIC list `pip install mlflow dspy jinja2 tqdm` as the prerequisites; jinja2 and tqdm are
# MAGIC pre-installed on Databricks runtimes, so we add only `dspy` here. Other lessons don't
# MAGIC need it.

# COMMAND ----------

# MAGIC %run ./_resources/setup

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 1 - Make sure the judge has run on existing traces
# MAGIC
# MAGIC Re-create the relevance judge from lesson 4 and apply it to every trace in the experiment.
# MAGIC If lesson 4 already populated judge assessments, this is a no-op for already-scored traces.

# COMMAND ----------

from typing import Literal
from mlflow.entities import AssessmentSource, AssessmentSourceType
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

print(f"Judge '{relevance_judge.name}' rebuilt. Running on existing traces.")

# COMMAND ----------

traces = mlflow.search_traces(
    experiment_ids=[mlflow.get_experiment_by_name(EXPERIMENT_PATH).experiment_id],
    return_type="list",
    max_results=200,
)
print(f"Found {len(traces)} traces in the experiment.")

scored = 0
for t in traces:
    existing = [
        a
        for a in (t.info.assessments or [])
        if a.name == "relevance"
        and a.source.source_type == AssessmentSourceType.LLM_JUDGE
    ]
    if existing:
        continue
    inputs = t.data.request if t.data else None
    outputs = t.data.response if t.data else None
    if not inputs or not outputs:
        continue
    assessment = relevance_judge(inputs={"question": str(inputs)}, outputs=str(outputs))
    mlflow.log_assessment(trace_id=t.info.trace_id, assessment=assessment)
    scored += 1

print(f"Added judge assessments to {scored} previously unjudged traces.")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 2 - Make sure SME labels exist
# MAGIC
# MAGIC In production, the human labels come from SMEs working through the labeling session created
# MAGIC in lesson 3. For the workshop, if no HUMAN `relevance` labels exist yet, we synthesize them
# MAGIC using a deterministic rule so the rest of this notebook is runnable. Replace this cell with
# MAGIC real SME labels before drawing conclusions about agreement numbers.

# COMMAND ----------

traces = mlflow.search_traces(
    experiment_ids=[mlflow.get_experiment_by_name(EXPERIMENT_PATH).experiment_id],
    return_type="list",
    max_results=200,
)

human_count = sum(
    1
    for t in traces
    for a in (t.info.assessments or [])
    if a.name == "relevance" and a.source.source_type == AssessmentSourceType.HUMAN
)

if human_count >= 10:
    print(f"{human_count} HUMAN relevance labels already exist. Using real SME data.")
else:
    print(
        f"Only {human_count} HUMAN labels found. Logging synthetic ones for workshop runnability."
    )
    # Synthetic rule - intentionally stricter than the unaligned judge so alignment has signal to learn from.
    # Counts an answer as relevant only when the response is non-empty AND mentions a token from the question.
    synthetic = 0
    for t in traces:
        if any(
            a.name == "relevance" and a.source.source_type == AssessmentSourceType.HUMAN
            for a in (t.info.assessments or [])
        ):
            continue
        question = str(t.data.request or "").lower()
        answer = str(t.data.response or "").lower()
        if not answer.strip() or "don't have" in answer or "i don't know" in answer:
            value = "no"
        else:
            q_tokens = {tok for tok in question.split() if len(tok) > 4}
            overlap = sum(1 for tok in q_tokens if tok in answer)
            if overlap >= 2:
                value = "yes"
            elif overlap == 1:
                value = "partial"
            else:
                value = "no"
        mlflow.log_feedback(
            trace_id=t.info.trace_id,
            name="relevance",
            value=value,
            source=AssessmentSource(
                source_type=AssessmentSourceType.HUMAN,
                source_id="workshop_synthetic_sme",
            ),
            rationale="Workshop simulation - replace with real SME labels in production.",
        )
        synthetic += 1
    print(f"Logged {synthetic} synthetic HUMAN labels.")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 3 - Filter to paired traces and measure baseline agreement

# COMMAND ----------

import pandas as pd

traces = mlflow.search_traces(
    experiment_ids=[mlflow.get_experiment_by_name(EXPERIMENT_PATH).experiment_id],
    return_type="list",
    max_results=200,
)

paired = []
rows = []
for t in traces:
    judge_vals = [
        a.feedback.value
        for a in (t.info.assessments or [])
        if a.name == "relevance"
        and a.source.source_type == AssessmentSourceType.LLM_JUDGE
    ]
    human_vals = [
        a.feedback.value
        for a in (t.info.assessments or [])
        if a.name == "relevance" and a.source.source_type == AssessmentSourceType.HUMAN
    ]
    if judge_vals and human_vals:
        paired.append(t)
        rows.append(
            {
                "trace_id": t.info.trace_id,
                "judge": judge_vals[-1],
                "human": human_vals[-1],
            }
        )

df = pd.DataFrame(rows)
print(f"Paired traces: {len(paired)} (need at least 10 for align())")
display(df)

if len(paired) >= 10:
    baseline_agreement = (df.judge == df.human).mean()
    print(f"Baseline agreement (judge vs human): {baseline_agreement:.0%}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 4 - Run alignment with MemAlign
# MAGIC
# MAGIC `judge.align(traces)` returns a new judge with rewritten instructions. MLflow ships three
# MAGIC optimizers in `mlflow.genai.judges.optimizers`:
# MAGIC
# MAGIC - **SIMBA** (Simplified Multi-Bootstrap Aggregation, DSPy-based) - the no-arg default
# MAGIC   per the [SIMBA docs](https://mlflow.org/docs/latest/genai/eval-monitor/scorers/llm-judge/simba/).
# MAGIC - **GEPA** (LLM-driven reflection, DSPy-based) - stronger when SMEs leave rich textual
# MAGIC   rationales. See the [GEPA docs](https://mlflow.org/docs/latest/genai/eval-monitor/scorers/llm-judge/gepa/).
# MAGIC - **MemAlign** (memory-augmented, currently experimental) - the MLflow team indicated on
# MAGIC   2026-05-13 it's a likely future default. See the [MemAlign docs](https://mlflow.org/docs/latest/genai/eval-monitor/scorers/llm-judge/memalign/).
# MAGIC
# MAGIC We explicitly pass `MemAlignOptimizer(reflection_lm=...)` below. This is the same call
# MAGIC shape the MemAlign docs use, and it demonstrates the parameter pattern. Swap to SIMBA by
# MAGIC removing the `optimizer=` kwarg or pass `GEPAAlignmentOptimizer(...)` if you want LLM
# MAGIC reflection over rich SME rationales.

# COMMAND ----------

from mlflow.genai.judges.optimizers import MemAlignOptimizer

assert len(paired) >= 10, (
    "Need at least 10 paired traces. Run lesson 2 to generate more traces, "
    "then re-run this notebook."
)

# MemAlign needs both a reflection_lm (for distilling guidelines from feedback)
# and an embedding_model (for retrieving relevant feedback examples). The
# embedding_model defaults to `openai:/text-embedding-3-small` which has no
# credentials in serverless Databricks - we point both at Databricks-hosted
# Foundation Model endpoints to avoid the missing-OPENAI_API_KEY failure mode.
memalign_optimizer = MemAlignOptimizer(
    reflection_lm=f"databricks:/{FM_ENDPOINT}",
    embedding_model="databricks:/databricks-gte-large-en",
)
aligned_judge = relevance_judge.align(paired, optimizer=memalign_optimizer)
print(f"Alignment complete with MemAlign. New judge name: {aligned_judge.name}")
print("\nUnaligned instructions:")
print(relevance_judge.instructions[:300] + "...")
print("\nAligned instructions:")
print(aligned_judge.instructions[:300] + "...")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 5 - Score the same traces with the aligned judge and re-measure

# COMMAND ----------

aligned_rows = []
for t in paired:
    inputs = t.data.request
    outputs = t.data.response
    aligned_assessment = aligned_judge(
        inputs={"question": str(inputs)},
        outputs=str(outputs),
    )
    mlflow.log_assessment(trace_id=t.info.trace_id, assessment=aligned_assessment)
    aligned_rows.append(
        {
            "trace_id": t.info.trace_id,
            "aligned_judge": aligned_assessment.feedback.value,
        }
    )

aligned_df = pd.DataFrame(aligned_rows).merge(df, on="trace_id")
new_agreement = (aligned_df.aligned_judge == aligned_df.human).mean()

print(f"Baseline agreement: {baseline_agreement:.0%}")
print(f"Aligned agreement:  {new_agreement:.0%}")
print(
    f"Lift:               {(new_agreement - baseline_agreement) * 100:+.1f} percentage points"
)
display(aligned_df[["trace_id", "human", "judge", "aligned_judge"]])

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 6 - Register the aligned judge for production use
# MAGIC
# MAGIC `aligned_judge.register(experiment_id=...)` saves the calibrated judge as a named scorer in
# MAGIC the experiment. Lesson 6's production monitoring picks it up via `get_scorer(name="relevance")`,
# MAGIC so scheduled scoring uses the aligned version automatically.

# COMMAND ----------

experiment_id = mlflow.get_experiment_by_name(EXPERIMENT_PATH).experiment_id

# Re-run safety: if a scorer named "relevance" is already registered on this
# experiment (e.g. from a previous alignment pass), register() raises ValueError.
# The current MLflow scorer API does not expose a "replace the registered body"
# operation - .update() only changes sampling config, and there is no delete
# verb. So on re-runs we keep the previously-registered version and continue;
# the in-memory `aligned_judge` from THIS run is still available for the rest
# of this notebook (Step 5 below uses it directly).
try:
    aligned_judge.register(experiment_id=experiment_id)
    print(f"Registered aligned 'relevance' judge against experiment {experiment_id}.")
except ValueError as e:
    if "already been registered" in str(e):
        print(
            f"'relevance' scorer already registered on experiment {experiment_id}. "
            "Skipping re-register - the previously aligned version stays as the "
            "scheduled scorer that lesson 6 picks up. This run's aligned_judge "
            "remains available in memory for Step 5 below."
        )
    else:
        raise

# COMMAND ----------

# MAGIC %md
# MAGIC ## What to verify
# MAGIC
# MAGIC 1. The aligned-vs-human agreement is higher than the baseline. If it isn't, you don't have
# MAGIC    enough paired traces, the SME labels are too noisy, or both labels happen to align with the
# MAGIC    original prompt already.
# MAGIC 2. Open one of the paired traces in the MLflow UI. You should see three `relevance`
# MAGIC    assessments: one HUMAN, one LLM_JUDGE (original), one LLM_JUDGE (aligned).
# MAGIC 3. The registered scorer shows up under the experiment's Scorers tab. Lesson 6 will use it.
# MAGIC
# MAGIC ## When to re-align
# MAGIC
# MAGIC - SMEs add a meaningful number of new labels (rule of thumb: 25%+ more than the last align).
# MAGIC - The agent's behavior changes (new tools, different retrieval index, prompt rewrite).
# MAGIC - You see judge-vs-human drift in production monitoring (lesson 6).
# MAGIC
# MAGIC ## Replace synthetic labels before drawing conclusions
# MAGIC
# MAGIC The synthetic-label fallback above exists so the lesson is runnable end-to-end without an SME
# MAGIC labeling pass. The agreement numbers it produces are not meaningful. In a real engagement,
# MAGIC delete the synthesis branch and run alignment only after SMEs have completed enough of the
# MAGIC lesson 3 labeling session.
