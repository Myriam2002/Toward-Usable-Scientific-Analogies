"""
Run Judge Script — Evaluate analogies using a single LLM judge model.

Supports parallel execution by running multiple instances with different
model/mode combinations (launched via PowerShell scripts).

Usage:
    python core/run_judge.py --model gpt-4.1-mini --mode 3scale_fewshot
    python core/run_judge.py --model deepseek-r1  --mode 3scale --test
"""

import argparse
import sys
import os
import math
import random
import time
import pandas as pd
from tqdm import tqdm

# Load environment variables from .env file BEFORE other imports
from dotenv import load_dotenv
env_paths = [
    os.path.join(os.path.dirname(__file__), '..', '..', '..', '.env'),  # root
    os.path.join(os.path.dirname(__file__), '..', '..', '.env'),        # LLM folder
    os.path.join(os.path.dirname(__file__), '.env'),                     # local
]
for env_path in env_paths:
    if os.path.exists(env_path):
        load_dotenv(env_path)
        print(f"Loaded environment from: {env_path}")
        break

# Add path to easy_llm_importer (now in core/ subdirectory)
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', '..', 'stage1_analysis', 'mapping_generation'))

import dspy
from easy_llm_importer import create_client, DSPyAdapter

# =============================================================================
# CONFIGURATION
# =============================================================================

JUDGE_MODELS = ["gpt-4.1-mini", "gemini-2.5-flash-lite", "deepseek-r1", "claude-sonnet-4.6", "mimo-v2-pro"]
JUDGE_MODES  = ["3scale", "3scale_fewshot"]

# BASE_DIR = stage_2_Modular_solution/LLM/ (two levels above core/)
BASE_DIR    = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS_DIR = os.path.join(BASE_DIR, 'results')
OUTPUT_DIR  = os.path.join(RESULTS_DIR, 'upgraded_llm')
os.makedirs(OUTPUT_DIR, exist_ok=True)

STAGE2_TARGETONLY = os.path.join(RESULTS_DIR, 'all_results_targetonly_rerank_edited_threshold.csv')
STAGE2_WITHSUB    = os.path.join(RESULTS_DIR, 'all_results_withsub_rerank_edited_threshold.csv')
STAGE1_JUDGE      = os.path.join(BASE_DIR, '..', '..', 'stage1_analysis', 'llm_judge_results.csv')

TEST_MODE_RECORD_LIMIT = 5

# =============================================================================
# PROMPTS — 1-3 SCALE
# =============================================================================

JUDGE_INSTRUCTIONS_3SCALE = """You are an expert evaluator of scientific analogies.

Given a target concept and a chosen source analogy, evaluate whether this is a good analogy.
A good analogy uses a FAMILIAR source concept to explain an UNFAMILIAR target concept through
meaningful structural or functional parallels.

For each of the three dimensions below, first provide a brief reasoning explaining your
assessment, then give the numeric score (1, 2, or 3).

ANALOGY_COHERENCE: Does the pairing make intuitive sense?
- 3: The connection is immediately clear and natural — most people would see it without explanation.
     The source and target share an obvious structural or functional parallel.
- 2: A meaningful connection exists but requires some explanation to see.
     The link is real but not self-evident; a sentence or two is needed to establish it.
- 1: No meaningful connection exists, or the pairing is random, forced, or misleading.

MAPPING_SOUNDNESS: Could properties/mechanisms of the source map to the target?
- 3: Rich, consistent structural or functional correspondences exist.
     Multiple source properties map precisely onto target properties.
     The mapping holds across the main components of both domains.
- 2: Some valid mappings exist, but coverage is partial.
     Core correspondences work, but important aspects of the target are not represented,
     or some mappings are approximate or strained.
- 1: No valid mappings are possible, or the apparent mappings are entirely superficial
     or misleading. Source and target are fundamentally incompatible.

EXPLANATORY_POWER: Would this analogy help a learner understand the target?
- 3: The analogy clearly illuminates the target concept and supports correct reasoning.
     A learner could use it to predict or explain target behavior.
- 2: The analogy provides partial insight with notable limitations.
     It conveys the general idea but cannot support deeper reasoning,
     or it risks creating minor misconceptions.
- 1: The analogy fails to aid understanding and would likely confuse or mislead a learner.
"""

FEW_SHOT_EXAMPLES_3SCALE = """
CALIBRATION EXAMPLES — use these to anchor your scoring.
Score each dimension independently. The scale is 1-3 only:
  3 = strong / clearly works
  2 = partial / works with caveats
  1 = doesn't work / poor or misleading

Example 1 (scores: 3 / 3 / 3)
  target: "electric circuit"  |  analogy: "water flowing through pipes"

  coherence=3:
    The connection is immediately clear without any explanation needed.
    A driving force pushes a substance through a constrained pathway in both cases —
    this structural parallel is instantly visible.

  mapping=3:
    Multiple precise correspondences hold consistently:
      voltage→pressure | current→flow rate | resistance→pipe restriction
      battery→pump | closed loop→closed pipe system
    Relationships between variables (increase pressure → increase flow) are preserved.
    A 3 requires consistent relational mapping across core components.

  explanatory=3:
    A learner can use the analogy to reason about Ohm's law qualitatively
    and make correct predictions about circuit behavior.
    A 3 requires that the analogy support transferable reasoning, not just description.

Example 2 (scores: 3 / 2 / 3)
  target: "cell"  |  analogy: "factory"

  coherence=3:
    The analogy captures organized division of labor within a bounded system.
    Most people immediately see why a cell resembles a factory.

  mapping=2:
    Clear functional correspondences exist for core components:
      nucleus→control center | ribosomes→production workers
      mitochondria→energy supply | membrane→boundary/security
    However, biochemical signaling, self-repair, and molecular regulation
    have no factory equivalent. Mapping covers the main structure but not the full depth.
    A 2 reflects strong but incomplete structural coverage.

  explanatory=3:
    Helps learners grasp coordination and specialization clearly.
    Supports correct reasoning about organelle roles even if it cannot support
    deeper molecular reasoning.

Example 3 (scores: 2 / 2 / 2)
  target: "mathematical function"  |  analogy: "machine"

  coherence=2:
    The shared idea (input→transformation→output) is real but requires a sentence to establish.
    The connection is not self-evident without explanation.

  mapping=2:
    Some correspondences exist: input→raw material, rule→internal mechanism, output→product.
    Properties like domain, range, composition, and invertibility are not represented.
    Mapping is real but partial.

  explanatory=2:
    Useful for initial intuition about transformation but cannot support reasoning
    about mathematical properties. Partial insight with clear limitations.

Example 4 (scores: 1 / 1 / 1)
  target: "neural network"  |  analogy: "human brain"

  coherence=1:
    The comparison relies on superficial label similarity ("neurons").
    The underlying architectures, learning mechanisms, and operational principles
    differ so significantly that this pairing is misleading rather than illuminating.

  mapping=1:
    Most functional properties (backpropagation, gradient descent, scale, energy use,
    layer structure) have no valid biological counterpart.
    Apparent correspondences are vague and inconsistent.

  explanatory=1:
    Risks creating misconceptions about how neural networks learn.
    Does not support accurate reasoning about the target concept.

Example 5 (scores: 1 / 1 / 1)
  target: "chemical reaction"  |  analogy: "a novel"

  coherence=1:
    No shared causal, structural, or functional system. Pairing is arbitrary.

  mapping=1:
    No systematic correspondences. Any alignment would be forced.

  explanatory=1:
    Provides no instructional value and would confuse learners.

Example 6 (scores: 3 / 1 / 2)
  target: "democracy"  |  analogy: "majority vote in a classroom"

  coherence=3:
    Both involve collective decision-making where the majority determines the outcome.
    The structural rule is identical and immediately visible.

  mapping=1:
    Institutions central to democracy — checks and balances, representation,
    judiciary, constitutional limits, minority protections — have no equivalent
    in the classroom scenario. The mapping captures only a single rule, not the system.
    A 1 reflects the absence of meaningful structural coverage beyond the surface rule.

  explanatory=2:
    Explains the concept of majority rule clearly, but fails to convey institutional
    complexity or why democracies need more than just voting.
    Partial insight with significant limitations.

Example 7 (scores: 2 / 3 / 3)
  target: "atom"  |  analogy: "solar system"

  coherence=2:
    The spatial pattern (central body with orbiting entities) is real but requires
    explanation: the governing forces differ (electrostatics vs. gravity), and
    the quantum nature of electron orbitals breaks the visual analogy.
    A meaningful connection, but not immediately obvious without caveats.

  mapping=3:
    Clear one-to-one correspondences hold for the core structure:
      nucleus→sun | electrons→planets | attraction force→gravity-like pull
    The relative scale and arrangement map consistently across key components.

  explanatory=3:
    Effectively conveys central attraction and relative scale.
    Supports correct initial reasoning about atomic structure even though
    quantum behavior is not captured.

Example 8 (scores: 3 / 2 / 1)
  target: "photosynthesis"  |  analogy: "a solar-powered factory"

  coherence=3:
    Energy input converted to useful product — the high-level systemic similarity
    is immediately clear to most learners.

  mapping=2:
    Sunlight→energy supply | chloroplast→factory unit | glucose→product work as correspondences.
    However, the chemical stages (light reactions, Calvin cycle, electron transport)
    lack precise counterparts. Core mapping works but coverage is partial.

  explanatory=1:
    Oversimplifies multi-step biochemical processes to the point of being unhelpful.
    A learner relying on this analogy cannot reason about photosynthesis mechanisms.

Example 9 (scores: 1 / 3 / 3)
  target: "compound interest"  |  analogy: "a snowball rolling downhill"

  coherence=1:
    The domains (finance vs. physical motion) are unrelated at a systemic level.
    The similarity is metaphorical — growth-via-accumulation — but the causal structures
    are fundamentally different. Does not meet the threshold for a 2.

  mapping=3:
    Despite weak coherence, the growth-through-accumulation structure aligns consistently:
      principal→initial snowball | interest accumulation→snow gathered
      compounding rate→slope steepness
    The relational structure (accumulation accelerates over time) is preserved.

  explanatory=3:
    Powerfully conveys why compound growth accelerates over time.
    A learner can use this analogy to reason correctly about why early investment matters.

Example 10 (scores: 3 / 1 / 2)
  target: "ecosystem"  |  analogy: "a family"

  coherence=3:
    Both involve interdependent members who affect one another.
    Shared systemic interconnection is immediately visible.

  mapping=1:
    Trophic levels, energy flow, nutrient cycles, population dynamics, and
    species interactions have no consistent family equivalents.
    Alignment is purely thematic; no structural correspondences hold.

  explanatory=2:
    Conveys interdependence reasonably well, but cannot support ecological
    reasoning or prediction. Partial intuitive value with significant gaps.
"""

JUDGE_INSTRUCTIONS_FEWSHOT_3SCALE = JUDGE_INSTRUCTIONS_3SCALE + FEW_SHOT_EXAMPLES_3SCALE

# =============================================================================
# DSPY SIGNATURES
# =============================================================================


class AnalogyJudge3Scale(dspy.Signature):
    __doc__ = JUDGE_INSTRUCTIONS_3SCALE

    target_concept: str = dspy.InputField(
        desc="The unfamiliar target concept being explained"
    )
    chosen_analogy: str = dspy.InputField(
        desc="The source concept chosen as the analogy"
    )
    selection_reasoning: str = dspy.InputField(
        desc="The reasoning used to select this analogy (empty if not available)"
    )
    coherence_reasoning: str = dspy.OutputField(
        desc="Brief explanation of why this coherence score was assigned"
    )
    analogy_coherence: int = dspy.OutputField(
        desc="Score 1-3: Does the pairing make intuitive sense?"
    )
    mapping_reasoning: str = dspy.OutputField(
        desc="Brief explanation of why this mapping soundness score was assigned"
    )
    mapping_soundness: int = dspy.OutputField(
        desc="Score 1-3: Could source properties map to target properties?"
    )
    explanatory_reasoning: str = dspy.OutputField(
        desc="Brief explanation of why this explanatory power score was assigned"
    )
    explanatory_power: int = dspy.OutputField(
        desc="Score 1-3: Would this help a learner understand the target?"
    )


class AnalogyJudge3ScaleFewShot(dspy.Signature):
    __doc__ = JUDGE_INSTRUCTIONS_FEWSHOT_3SCALE

    target_concept: str = dspy.InputField(
        desc="The unfamiliar target concept being explained"
    )
    chosen_analogy: str = dspy.InputField(
        desc="The source concept chosen as the analogy"
    )
    selection_reasoning: str = dspy.InputField(
        desc="The reasoning used to select this analogy (empty if not available)"
    )
    coherence_reasoning: str = dspy.OutputField(
        desc="Brief explanation of why this coherence score was assigned"
    )
    analogy_coherence: int = dspy.OutputField(
        desc="Score 1-3: Does the pairing make intuitive sense?"
    )
    mapping_reasoning: str = dspy.OutputField(
        desc="Brief explanation of why this mapping soundness score was assigned"
    )
    mapping_soundness: int = dspy.OutputField(
        desc="Score 1-3: Could source properties map to target properties?"
    )
    explanatory_reasoning: str = dspy.OutputField(
        desc="Brief explanation of why this explanatory power score was assigned"
    )
    explanatory_power: int = dspy.OutputField(
        desc="Score 1-3: Would this help a learner understand the target?"
    )


# =============================================================================
# DATA LOADING
# =============================================================================

def safe_str(val, default=""):
    """Return string value, replacing NaN/None with default."""
    if val is None:
        return default
    if isinstance(val, float) and math.isnan(val):
        return default
    return str(val).strip()


def load_all_records() -> list:
    """
    Load and flatten all evaluation records from Stage 1 and Stage 2 CSVs.
    Returns a deduplicated list of dicts with record_id, target, chosen_analogy,
    selection_reasoning, and provenance metadata.
    """
    all_records = []

    # ── Stage 2: targetonly + withsub ────────────────────────────────────────
    for csv_path, source_dataset in [
        (STAGE2_TARGETONLY, "stage2_targetonly"),
        (STAGE2_WITHSUB,    "stage2_withsub"),
    ]:
        df = pd.read_csv(csv_path)
        df_ok = df[df["status"] == "success"].copy()
        print(f"Loaded {source_dataset}: {len(df_ok):,} successful rows")

        for row_idx, row in df_ok.iterrows():
            orig_id          = row.get("id", row_idx)
            target           = safe_str(row.get("target"))
            model            = safe_str(row.get("model"))
            mode             = safe_str(row.get("mode"))
            gen_reasoning    = safe_str(row.get("reasoning"))
            rerank_reasoning = safe_str(row.get("rerank_reasoning"))

            analogy_map = {
                "baseline":  (safe_str(row.get("top1_baseline")),  gen_reasoning),
                "embedding": (safe_str(row.get("top1_embedding")), gen_reasoning),
                "rerank":    (safe_str(row.get("top1_rerank")),    rerank_reasoning),
            }

            for analogy_type, (analogy, reasoning) in analogy_map.items():
                if not analogy:
                    continue
                record_id = f"{source_dataset}__{model}__{orig_id}__{analogy_type}"
                all_records.append({
                    "record_id":           record_id,
                    "source_dataset":      source_dataset,
                    "original_id":         orig_id,
                    "target":              target,
                    "model":               model,
                    "original_mode":       mode,
                    "analogy_type":        analogy_type,
                    "chosen_analogy":      analogy,
                    "selection_reasoning": reasoning,
                })

    # ── Stage 1: llm_judge_results.csv ───────────────────────────────────────
    df_s1 = pd.read_csv(STAGE1_JUDGE)
    df_s1_ok = df_s1[df_s1["status"] == "success"].copy()
    print(f"Loaded stage1: {len(df_s1_ok):,} successful rows")

    for row_idx, row in df_s1_ok.iterrows():
        orig_id     = row.get("id", row_idx)
        source_file = safe_str(row.get("source_file"))
        target      = safe_str(row.get("target"))
        analogy     = safe_str(row.get("chosen_analogy"))
        if not analogy:
            continue
        record_id = f"stage1__{source_file}__{orig_id}"
        all_records.append({
            "record_id":           record_id,
            "source_dataset":      "stage1",
            "original_id":         orig_id,
            "target":              target,
            "model":               "retrieval-based",
            "original_mode":       source_file,
            "analogy_type":        source_file,
            "chosen_analogy":      analogy,
            "selection_reasoning": "",
        })

    # Verify uniqueness
    record_ids = [r["record_id"] for r in all_records]
    duplicates  = len(record_ids) - len(set(record_ids))
    assert duplicates == 0, f"{duplicates} duplicate record_ids detected!"

    print(f"\nTotal records to evaluate: {len(all_records):,}")
    return all_records


# =============================================================================
# EVALUATION
# =============================================================================

def evaluate_one(
    predictor,
    target: str,
    analogy: str,
    reasoning: str,
    max_retries: int = 3,
) -> dict:
    """Evaluate a single analogy. Returns dict with per-dimension scores and reasoning."""
    last_error = None
    for attempt in range(max_retries):
        try:
            result = predictor(
                target_concept=target,
                chosen_analogy=analogy,
                selection_reasoning=reasoning if reasoning else "No reasoning provided",
            )
            coherence   = max(1, min(3, int(result.analogy_coherence)))
            mapping     = max(1, min(3, int(result.mapping_soundness)))
            explanatory = max(1, min(3, int(result.explanatory_power)))
            return {
                "coherence_reasoning":   result.coherence_reasoning,
                "analogy_coherence":     coherence,
                "mapping_reasoning":     result.mapping_reasoning,
                "mapping_soundness":     mapping,
                "explanatory_reasoning": result.explanatory_reasoning,
                "explanatory_power":     explanatory,
                "average_score":         round((coherence + mapping + explanatory) / 3, 4),
                "status":                "success",
            }
        except Exception as e:
            last_error = str(e)
            if attempt < max_retries - 1:
                time.sleep((attempt + 1) * 2)  # 2s then 4s back-off

    return {
        "coherence_reasoning":   f"ERROR after {max_retries} attempts: {last_error}",
        "analogy_coherence":     None,
        "mapping_reasoning":     None,
        "mapping_soundness":     None,
        "explanatory_reasoning": None,
        "explanatory_power":     None,
        "average_score":         None,
        "status":                "error",
    }


def run_evaluation_mode(
    all_records: list,
    predictor,
    judge_model_name: str,
    judge_mode: str,
    save_interval: int = 25,
    test_n: int = None,
    rerun_errors: bool = False,
) -> pd.DataFrame:
    """
    Evaluate records with the given predictor and judge model.

    Checkpoint/resume: Each run maintains its own checkpoint file. Simply re-run
    the script after an interruption to continue from where it stopped.

    Parameters
    ----------
    all_records      : full list of records from load_all_records()
    predictor        : dspy.Predict(AnalogyJudge3Scale) or dspy.Predict(AnalogyJudge3ScaleFewShot)
    judge_model_name : model name string — stored in 'judge_model' column and used in filenames
    judge_mode       : "3scale" or "3scale_fewshot"
    save_interval    : checkpoint saved every N newly evaluated records
    test_n           : if set, randomly sample and evaluate without writing any files
    rerun_errors     : if True, load the completed output file and re-evaluate only error rows
    """
    # Sanitize model name for filenames (dots→underscores for checkpoint only)
    safe_model      = judge_model_name.replace(".", "_")
    checkpoint_path = os.path.join(OUTPUT_DIR, f"checkpoint_{judge_mode}_{safe_model}.csv")
    output_path     = os.path.join(OUTPUT_DIR, f"upgraded_judge_{judge_mode}_{judge_model_name}.csv")

    # ── Test mode ─────────────────────────────────────────────────────────────
    if test_n is not None:
        sample = random.sample(all_records, min(test_n, len(all_records)))
        print(f"\n[{judge_model_name}/{judge_mode}] TEST MODE — {len(sample)} records (no files written)\n")
        results = []
        errors  = 0
        for record in tqdm(sample, desc=f"[{judge_model_name}] test"):
            scores = evaluate_one(
                predictor,
                record["target"],
                record["chosen_analogy"],
                record["selection_reasoning"],
            )
            if scores["status"] == "error":
                errors += 1
            results.append({**record, **scores,
                            "judge_model": judge_model_name, "judge_mode": judge_mode})

        df = pd.DataFrame(results)
        ok = df[df["status"] == "success"]
        print(f"\nTest complete — {len(ok)}/{len(df)} successful, {errors} errors")
        if not ok.empty:
            print(f"  avg coherence:   {ok['analogy_coherence'].mean():.2f}")
            print(f"  avg mapping:     {ok['mapping_soundness'].mean():.2f}")
            print(f"  avg explanatory: {ok['explanatory_power'].mean():.2f}")
            print(f"  avg overall:     {ok['average_score'].mean():.2f}")
            display_cols = ["target", "chosen_analogy", "analogy_coherence",
                            "mapping_soundness", "explanatory_power", "average_score"]
            print(ok[display_cols].to_string(index=False))
        return df

    # ── Full run with checkpoint ───────────────────────────────────────────────
    if rerun_errors:
        if not os.path.exists(output_path):
            print(f"[{judge_model_name}/{judge_mode}] No output file found: {output_path}")
            print("  Run the full evaluation first, then use --rerun-errors to retry errors.")
            sys.exit(1)
        done_df    = pd.read_csv(output_path)
        error_df   = done_df[done_df["status"] == "error"]
        if error_df.empty:
            print(f"[{judge_model_name}/{judge_mode}] No error records in output file — nothing to rerun.")
            return done_df
        success_df = done_df[done_df["status"] != "error"]
        done_ids   = set(success_df["record_id"].tolist())
        results    = success_df.to_dict("records")
        print(f"[{judge_model_name}/{judge_mode}] Rerunning {len(error_df):,} errors "
              f"({len(done_ids):,} successful records kept)")
    elif os.path.exists(checkpoint_path):
        done_df  = pd.read_csv(checkpoint_path)
        done_ids = set(done_df["record_id"].tolist())
        results  = done_df.to_dict("records")
        print(f"[{judge_model_name}/{judge_mode}] Resuming: {len(done_ids):,} already evaluated")
    else:
        done_ids, results = set(), []

    pending = [r for r in all_records if r["record_id"] not in done_ids]
    print(f"[{judge_model_name}/{judge_mode}] Remaining: {len(pending):,} / {len(all_records):,}")

    errors = 0
    for i, record in enumerate(tqdm(pending, desc=f"[{judge_model_name}/{judge_mode}]")):
        scores = evaluate_one(
            predictor,
            record["target"],
            record["chosen_analogy"],
            record["selection_reasoning"],
        )
        if scores["status"] == "error":
            errors += 1

        results.append({
            **record,
            **scores,
            "judge_model": judge_model_name,
            "judge_mode":  judge_mode,
        })
        done_ids.add(record["record_id"])

        if (i + 1) % save_interval == 0:
            pd.DataFrame(results).to_csv(checkpoint_path, index=False)
            print(f"  Checkpoint: {len(results):,} total ({errors} errors so far)")

    final_df = pd.DataFrame(results)
    final_df.to_csv(output_path, index=False)
    if os.path.exists(checkpoint_path):
        os.remove(checkpoint_path)

    ok_count = (final_df["status"] == "success").sum()
    print(f"\n[{judge_model_name}/{judge_mode}] Complete!")
    print(f"  Records: {len(final_df):,} total | {ok_count:,} successful | {errors} errors")
    print(f"  Output:  {os.path.abspath(output_path)}")
    return final_df


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Evaluate analogies using an LLM-as-a-Judge on a 1-3 scale"
    )
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        choices=JUDGE_MODELS,
        help="Judge model name",
    )
    parser.add_argument(
        "--mode",
        type=str,
        required=True,
        choices=JUDGE_MODES,
        help="Judge mode: 3scale (no few-shot) or 3scale_fewshot (with calibration examples)",
    )
    parser.add_argument(
        "--test",
        action="store_true",
        help=f"Test mode: evaluate {TEST_MODE_RECORD_LIMIT} random records, no files written",
    )
    parser.add_argument(
        "--rerun-errors",
        action="store_true",
        help="Re-evaluate records that previously failed (status='error'). Requires a completed output file.",
    )
    args = parser.parse_args()

    print(f"\n{'='*70}")
    print(f"  LLM-as-a-Judge  |  model: {args.model}  |  mode: {args.mode}")
    if args.test:
        print(f"  [TEST MODE — {TEST_MODE_RECORD_LIMIT} records only]")
    if args.rerun_errors:
        print(f"  [RERUN ERRORS MODE — re-evaluating failed records only]")
    print(f"{'='*70}\n")

    # ── Configure LLM ─────────────────────────────────────────────────────────
    # All three judge models use get_dspy_lm() with no temperature override →
    # temperature=0.1 uniformly (the library default for non-reasoning models).
    client  = create_client()
    adapter = DSPyAdapter(client, args.model)
    lm      = adapter.get_dspy_lm()
    dspy.configure(lm=lm)
    print(f"DSPy configured with {args.model} (temperature=0.1)")

    # ── Select predictor ───────────────────────────────────────────────────────
    if args.mode == "3scale":
        predictor = dspy.Predict(AnalogyJudge3Scale)
    else:
        predictor = dspy.Predict(AnalogyJudge3ScaleFewShot)

    # ── Load data ──────────────────────────────────────────────────────────────
    all_records = load_all_records()

    # ── Run evaluation ─────────────────────────────────────────────────────────
    test_n = TEST_MODE_RECORD_LIMIT if args.test else None
    run_evaluation_mode(
        all_records,
        predictor,
        judge_model_name=args.model,
        judge_mode=args.mode,
        save_interval=25,
        test_n=test_n,
        rerun_errors=args.rerun_errors,
    )


if __name__ == "__main__":
    main()
