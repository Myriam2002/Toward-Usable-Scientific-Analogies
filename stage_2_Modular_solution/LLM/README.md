# LLM Baselines Pipeline

This folder contains the complete pipeline for evaluating LLM-based analogy generation across 12 different language models, including reranking, re-evaluation with multiple judge models, and human annotation analysis.

---

## Methodology Flowchart

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                           LLM BASELINES PIPELINE                                │
└─────────────────────────────────────────────────────────────────────────────────┘

                              ┌─────────────────┐
                              │   SCAR Dataset  │
                              │   (Input Data)  │
                              └────────┬────────┘
                                       │
                                       ▼
                        ┌──────────────────────────────┐
                        │  Step 1: PRECOMPUTATION      │
                        │  (Run Once)                  │
                        ├──────────────────────────────┤
                        │ • Gold Source Embeddings     │
                        │   (all-MiniLM-L6-v2)         │
                        │ • Target Embeddings          │
                        │   (OpenAI text-embedding-3)  │
                        └──────────────┬───────────────┘
                                       │
                    ┌──────────────────┴──────────────────┐
                    │                                     │
                    ▼                                     ▼
     ┌──────────────────────────┐         ┌──────────────────────────┐
     │   MODE: targetonly       │         │    MODE: withsub         │
     ├──────────────────────────┤         ├──────────────────────────┤
     │                          │         │                          │
     │  Target Concept Only     │         │  Target + Sub-concepts   │
     │                          │         │                          │
     │  "cell membrane"         │         │  "cell membrane"         │
     │          │               │         │  + "selective barrier,   │
     │          ▼               │         │     lipid bilayer,       │
     │                          │         │     transport proteins"  │
     └──────────┬───────────────┘         └──────────┬───────────────┘
                │                                     │
                └─────────────┬───────────────────────┘
                              │
                              ▼
              ┌───────────────────────────────────────┐
              │    Step 2: ANALOGY GENERATION         │
              │    (Per Model × Mode)                 │
              ├───────────────────────────────────────┤
              │  • LLM generates 20 source analogies  │
              │  • Output: analogy_1 ... analogy_20   │
              │  • 12 models × 2 modes = 24 runs      │
              └───────────────────┬───────────────────┘
                                  │
                                  ▼
              ┌───────────────────────────────────────┐
              │    Step 3: EVALUATION                 │
              ├───────────────────────────────────────┤
              │                                       │
              │  ┌─────────────────────────────────┐  │
              │  │  3a. Gold Source Matching       │  │
              │  │  • Exact match check            │  │
              │  │  • Semantic similarity check    │  │
              │  │    (threshold: 0.5)             │  │
              │  │  • Compute Hit@K metrics        │  │
              │  └─────────────────────────────────┘  │
              │                                       │
              │  ┌─────────────────────────────────┐  │
              │  │  3b. Top-1 Selection            │  │
              │  │  • top1_baseline: First analogy │  │
              │  │  • top1_embedding: Best match   │  │
              │  │    (by embedding similarity     │  │
              │  │    to target)                   │  │
              │  └─────────────────────────────────┘  │
              │                                       │
              │  ┌─────────────────────────────────┐  │
              │  │  3c. LLM-as-Judge Evaluation    │  │
              │  │  • Evaluates both top1_baseline │  │
              │  │    and top1_embedding           │  │
              │  │  • Scores (1-3):                │  │
              │  │    - Analogy Coherence          │  │
              │  │    - Mapping Soundness          │  │
              │  │    - Explanatory Power          │  │
              │  └─────────────────────────────────┘  │
              └───────────────────┬───────────────────┘
                                  │
                                  ▼
              ┌───────────────────────────────────────┐
              │    Step 4: AGGREGATION & ANALYSIS     │
              ├───────────────────────────────────────┤
              │  • Aggregate results across models    │
              │  • Compute summary metrics            │
              │  • Generate visualizations            │
              │  • Compare targetonly vs withsub      │
              │  • Output: all_results_targetonly.csv │
              │         all_results_withsub.csv        │
              └───────────────────┬───────────────────┘
                                  │
                                  ▼
              ┌───────────────────────────────────────┐
              │    Step 5: RERANKING (Optional)       │
              ├───────────────────────────────────────┤
              │  • Rerank all 20 analogies using      │
              │    LLM reranker                       │
              │  • Re-evaluate top-1 reranked choice  │
              │  • Output: *_rerank.csv files         │
              └───────────────────┬───────────────────┘
                                  │
                                  ▼
              ┌───────────────────────────────────────┐
              │    Step 6: MULTI-JUDGE EVALUATION     │
              │    (Optional — Upgraded Judge)        │
              ├───────────────────────────────────────┤
              │  • Re-evaluate with 5 judge models    │
              │  • Two judge modes: 3scale,           │
              │    3scale_fewshot                     │
              │  • Output: results/upgraded_llm/      │
              └───────────────────────────────────────┘
```

---

## Methodology (Numbered Steps)

### Step 1: Precomputation (Run Once)

1. **Load SCAR Dataset**: Load the Scientific Concept Analogy Resource dataset containing target-source analogy pairs.

2. **Precompute Gold Source Embeddings**:
   - Extract all unique gold sources (correct answers) from the dataset
   - Compute embeddings using **SentenceTransformer all-MiniLM-L6-v2**
   - Save to `data/gold_source_embeddings.pkl`

3. **Precompute Target Embeddings**:
   - Compute target-only embeddings using **OpenAI text-embedding-3-small**
   - Compute target-with-subconcepts embeddings (target + sub-concepts combined)
   - Save to `data/target_embeddings.pkl` and `data/target_with_subconcepts_embeddings.pkl`

---

### Step 2: Analogy Generation (Per Model)

4. **Deduplicate Targets**: Group dataset by target, collecting all gold sources for each unique target.

5. **Two Generation Modes**:
   - **targetonly**: LLM receives only the target concept name
   - **withsub**: LLM receives target + sub-concepts (e.g., key properties extracted via mapping model)

6. **Generate Analogies**:
   - For each target, the LLM generates **20 source analogies**
   - Output format: `analogy_1, analogy_2, ..., analogy_20`

7. **Models Evaluated** (12 models):
   ```
   gpt-oss-20b, gpt-oss-120b, gpt-4.1-mini, gpt-4.1-nano,
   grok-4-fast, gemini-2.5-flash-lite, llama-3.1-405b-instruct,
   meta-llama-3-1-70b-instruct, meta-llama-3-1-8b-instruct,
   deepseek-r1, qwen3-14b, qwen3-32b
   ```

---

### Step 3: Evaluation

8. **Gold Source Matching**:
   - **Exact Match**: Check if any generated analogy exactly matches a gold source
   - **Semantic Match**: Compute cosine similarity (threshold ≥ 0.5) between generated analogies and gold sources
   - Record **gold_ranks** (position of matching analogy) for Hit@K computation

9. **Top-1 Selection Methods**:
   - **top1_baseline**: First generated analogy (position-based)
   - **top1_embedding**: Analogy with highest embedding similarity to target (OpenAI embeddings)

10. **LLM-as-Judge Evaluation**:
    - Default judge model: **gpt-4.1-mini**
    - Evaluates BOTH top1_baseline and top1_embedding
    - Scoring dimensions (1-3 scale):
      - **Analogy Coherence**: Does the pairing make intuitive sense?
      - **Mapping Soundness**: Can source properties map to target properties?
      - **Explanatory Power**: Would this help a learner understand the target?
    - Stores results in `judge_baseline` and `judge_embedding` columns

---

### Step 4: Aggregation & Analysis

11. **Automatic Aggregation**:
    - When all 12 models complete, results are automatically aggregated
    - Creates two files: `all_results_targetonly.csv` and `all_results_withsub.csv`
    - Each file contains all models' results with `model` and `mode` columns

12. **Compute Summary Metrics**:
    - **Hit@K** (K=1,3,5,10,20): Percentage of targets where a gold source was found in top K
    - **Average Judge Scores**: Mean scores across all targets
    - **Top-1 Embedding Score**: Average embedding similarity for selected analogies

13. **Generate Visualizations**:
    - Hit@K comparison across models
    - LLM Judge score distributions
    - Mode comparison (targetonly vs withsub)

14. **Export Results**:
    - Individual CSV files per model/mode: `results/LLM_{model}_{mode}_eval.csv`
    - Aggregated files: `results/all_results_{mode}.csv`
    - Contains all analogies, scores, rankings, and judge evaluations

---

### Step 5: Reranking (Optional)

15. **Rerank Generated Analogies**:
    - Load aggregated results CSV (`all_results_targetonly.csv` or `all_results_withsub.csv`)
    - For `targetonly` mode: Automatically generates `target_subconcepts` and per-analogy subconcepts from SCAR dataset
    - For `withsub` mode: Uses existing subconcepts from CSV
    - Reranks all 20 analogies using LLM reranker (`meta-llama-3-1-70b-instruct`)
    - Evaluates top-1 reranked choice with LLM-as-judge

16. **Output Files**:
    - `all_results_targetonly_rerank.csv`
    - `all_results_withsub_rerank.csv`
    - `all_results_targetonly_rerank_edited_threshold.csv` (reranked with adjusted similarity threshold)
    - `all_results_withsub_rerank_edited_threshold.csv`
    - Contains original columns plus reranking results

**Note**: Reranking must be run in **separate terminals** (one per file) due to DSPy limitations.

---

### Step 6: Multi-Judge Evaluation (Optional)

17. **Upgraded Judge Pipeline**:
    - Re-evaluate analogies using multiple judge models for cross-judge agreement analysis
    - **5 judge models**: `gpt-4.1-mini`, `gemini-2.5-flash-lite`, `deepseek-r1`, `claude-sonnet-4.6`, `mimo-v2-pro`
    - **2 judge modes**: `3scale` (standard), `3scale_fewshot` (with few-shot examples)
    - Output files in `results/upgraded_llm/`

18. **Human Annotation**:
    - A subset of 15 targets was annotated by human annotators
    - Annotation data in `human_annotation/`
    - Comparison with LLM judge scores in `results/human_annotation/`

---

## File Structure

```
LLM/
├── core/                         # Core pipeline scripts
│   ├── config.py                 # Configuration (models, prompts, paths)
│   ├── run_model.py              # Analogy generation script
│   ├── evaluate_model.py         # Evaluation (gold matching + judge)
│   ├── precompute_similarity.py  # Embedding precomputation & semantic matching
│   └── run_judge.py              # Multi-judge evaluation script
│
├── utilities/                    # Post-processing utilities
│   ├── aggregate_results.py      # Aggregates all model results by mode
│   ├── rerank_aggregated_results.py  # Reranks aggregated results
│   ├── rerun_problematic_records.py  # Re-runs failed/problematic records
│   └── rerun_withsub_failed.py   # Re-runs failed withsub records
│
├── scripts/                      # PowerShell launch scripts
│   ├── launch_all_and_aggregate.ps1  # Main: launches all models + auto-aggregates
│   ├── run_single_model.ps1      # Runs one model (generation + evaluation)
│   ├── run_single_judge.ps1      # Runs one judge model
│   ├── run_all_judges.ps1        # Launches all judge models in parallel
│   ├── aggregate_results.ps1     # Manual aggregation script
│   ├── run_withsub_only.ps1      # Runs withsub mode only
│   ├── rerun_model_mode_records.ps1
│   ├── rerun_single_model_targets.ps1
│   ├── rerun_problematic_records.ps1
│   ├── rerun_withsub_failed.ps1
│   ├── rerun_judge_errors.ps1
│   ├── run_models/               # Individual model scripts (12 total)
│   │   ├── run_gpt-4.1-mini.ps1
│   │   ├── run_llama-3.1-405b-instruct.ps1
│   │   └── ... (12 total)
│   └── run_judges/               # Individual judge scripts (5 total)
│       ├── run_judge_gpt-4.1-mini.ps1
│       ├── run_judge_deepseek-r1.ps1
│       ├── run_judge_gemini-2.5-flash-lite.ps1
│       ├── run_judge_claude-sonnet-4.6.ps1
│       └── run_judge_mimo-v2-pro.ps1
│
├── analysis/                     # Analysis notebooks
│   ├── baselines.ipynb           # Main results analysis & visualization
│   ├── results_visualization.ipynb
│   ├── similarity_analysis_v2.ipynb
│   ├── judge_comparison_heatmap.ipynb
│   ├── judge_agreement_analysis.ipynb
│   ├── model_mode_similarity_comparison.ipynb
│   ├── scar_wrong_distribution_comparison.ipynb
│   └── LLM_semi_closed_visualizations.ipynb
│
├── notebooks/                    # Supplementary notebooks
│   ├── human_annotation_analysis.ipynb
│   ├── upgrading_LLM_as_a_judge.ipynb
│   ├── wordnet_distribution_improve.ipynb
│   ├── build_notebook.py
│   └── wordnet_outputs/
│
├── data/                         # Precomputed embeddings & supporting data
│   ├── gold_source_embeddings.pkl        # Gold source embeddings (all-MiniLM-L6-v2)
│   ├── target_embeddings.pkl             # Target-only embeddings (OpenAI)
│   ├── target_with_subconcepts_embeddings.pkl  # Target+subconcepts embeddings
│   └── position_similarity_comparison.csv
│
├── human_annotation/             # Human annotation data
│   ├── 15_targets_full_data.xlsx
│   ├── 15_targets_full_data_with_reasoning.xlsx
│   ├── annotation_form.html
│   ├── annotation_form_updated.html
│   └── human_results/
│
└── results/                      # Output directory for all results
    ├── .markers/                 # Completion markers (auto-created)
    ├── LLM_*_targetonly.csv          # Generation results per model
    ├── LLM_*_targetonly_eval.csv     # Evaluation results per model
    ├── LLM_*_withsub.csv
    ├── LLM_*_withsub_eval.csv
    ├── all_results_targetonly.csv    # Aggregated (all models, targetonly)
    ├── all_results_withsub.csv       # Aggregated (all models, withsub)
    ├── all_results_targetonly_rerank.csv
    ├── all_results_withsub_rerank.csv
    ├── all_results_targetonly_rerank_edited_threshold.csv
    ├── all_results_withsub_rerank_edited_threshold.csv
    ├── final_visualizations/     # Publication-ready plots
    ├── judge_analysis/           # Cross-judge agreement plots
    ├── human_annotation/         # Human vs. LLM comparison plots
    ├── upgraded_llm/             # Multi-judge evaluation outputs
    │   ├── upgraded_judge_3scale_{model}.csv
    │   └── upgraded_judge_3scale_fewshot_{model}.csv
    └── similarity_analysis/      # Similarity distribution plots
```

---

## Quick Start

### 1. Precompute Embeddings (Run Once — REQUIRED)

**IMPORTANT**: Embeddings must be precomputed before running models.

```bash
python core/precompute_similarity.py --mode both
```

This creates in `data/`:
- `gold_source_embeddings.pkl` (for semantic matching)
- `target_embeddings.pkl` (for top-1-embedding selection)
- `target_with_subconcepts_embeddings.pkl` (for withsub mode)

**Note**: If these files already exist, precomputation is skipped automatically.

### 2. Run All Models (Recommended — Auto-Aggregates)

```powershell
# Full run (all 321 records per model)
.\scripts\launch_all_and_aggregate.ps1

# Test mode (5 records per model - for quick testing)
.\scripts\launch_all_and_aggregate.ps1 -Test
```

**What this does**:
1. Launches 12 PowerShell terminals (one per model)
2. Each terminal runs generation (targetonly + withsub) + evaluation (LLM Judge + Semantic Matching)
3. Monitors completion using marker files
4. **Automatically runs aggregation** when all models finish
5. Creates `results/all_results_targetonly.csv` and `results/all_results_withsub.csv`

**Output**: The main terminal shows progress: `[5/12] completed | Elapsed: 00:15:30`

### 3. Run Single Model

```powershell
# Full run
.\scripts\run_single_model.ps1 -Model "gpt-4.1-mini"

# Test mode (5 records)
.\scripts\run_single_model.ps1 -Model "gpt-4.1-mini" -Test
```

Or using Python directly:

```bash
# targetonly mode
python core/run_model.py --model gpt-4.1-mini --mode targetonly

# withsub mode
python core/run_model.py --model gpt-4.1-mini --mode withsub

# Test mode (5 records)
python core/run_model.py --model gpt-4.1-mini --mode targetonly --test
```

### 4. Manual Aggregation (If Needed)

```powershell
.\scripts\aggregate_results.ps1
```

Or:

```bash
python utilities/aggregate_results.py
```

### 5. Rerank Aggregated Results (Optional)

**Important**: Run the two files in **separate terminals** (DSPy does not support in-process parallelism).

**Terminal A (targetonly)**:
```bash
python utilities/rerank_aggregated_results.py --input results/all_results_targetonly.csv --verbose
```

**Terminal B (withsub)**:
```bash
python utilities/rerank_aggregated_results.py --input results/all_results_withsub.csv --verbose
```

**Command-Line Options**:
- `--input PATH` (required): Path to aggregated results CSV
- `--test [N]`: Process only first N records (default: 3 if N not specified)
- `--resume`: Skip rows already present in existing rerank output file (auto-detects existing files)
- `--verbose`: Print detailed output for each record

**Features**:
- **Incremental saving**: Saves after each record to prevent data loss
- **Auto-resume**: Automatically detects existing output files and continues from where it left off
- **Subconcept generation**: For `targetonly`, automatically generates `target_subconcepts` and `sec_generated_subconcepts` from SCAR dataset
- **Subconcept extraction**: For both modes, always extracts `target_subconcepts` from SCAR dataset

### 6. Multi-Judge Evaluation (Optional)

Run all 5 judge models in parallel:

```powershell
.\scripts\run_all_judges.ps1
```

Or run a single judge:

```powershell
.\scripts\run_single_judge.ps1 -Model "deepseek-r1" -Mode "3scale_fewshot"
```

Or using Python directly:

```bash
python core/run_judge.py --model gpt-4.1-mini --mode 3scale_fewshot
python core/run_judge.py --model deepseek-r1  --mode 3scale --test
```

**Judge models**: `gpt-4.1-mini`, `gemini-2.5-flash-lite`, `deepseek-r1`, `claude-sonnet-4.6`, `mimo-v2-pro`
**Judge modes**: `3scale` (standard), `3scale_fewshot` (with few-shot examples)

Output files saved to `results/upgraded_llm/`.

### 7. Analyze Results

Open notebooks in `analysis/` to:
- Compute summary metrics (`baselines.ipynb`)
- Generate visualizations (`results_visualization.ipynb`)
- Compare models and modes
- Analyze judge agreement (`judge_agreement_analysis.ipynb`, `judge_comparison_heatmap.ipynb`)
- Compare with human annotations (`notebooks/human_annotation_analysis.ipynb`)

---

## Key Configuration (`core/config.py`)

| Parameter | Value | Description |
|-----------|-------|-------------|
| `NUM_ANALOGIES` | 20 | Number of analogies generated per target |
| `SIMILARITY_THRESHOLD` | 0.5 | Semantic match threshold |
| `EMBEDDING_MODEL` | all-MiniLM-L6-v2 | For gold source matching |
| `JUDGE_MODEL` | gpt-4.1-mini | Default LLM-as-judge model |
| `MAPPING_MODEL` | meta-llama-3-1-70b-instruct | Sub-concept extraction model |
| `TEST_MODE_RECORD_LIMIT` | 3 | Number of records in test mode |

## Environment Setup

**Required**: Create a `.env` file in the project root with your API keys:

```
OPENAI_API_KEY=your_key_here
OPENROUTER_API_KEY=your_key_here
DEEPINFRA_API_KEY=your_key_here
```

The scripts automatically load environment variables from `.env` file.

---

## Output Columns

### Standard Evaluation CSV

| Column | Description |
|--------|-------------|
| `target` | Target concept |
| `target_subconcepts` | Target sub-concepts (extracted from SCAR for reranked files) |
| `all_gold_sources` | All correct source analogies (JSON list) |
| `generated_analogies` | Generated analogies (JSON list) |
| `top1_baseline` | First generated analogy |
| `top1_embedding` | Best analogy by embedding similarity to target |
| `top1_embedding_score` | Similarity score for top1_embedding |
| `gold_ranks_list` | JSON list of all exact-match ranks — used for Hit@K (exact) |
| `gold_ranks` | JSON dict mapping generated analogy → exact rank |
| `sem_gold_ranks_list` | JSON list of all semantic-match ranks — used for Hit@K (semantic) |
| `sem_gold_ranks` | JSON dict mapping generated analogy → semantic rank |
| `similarity_per_gold` | JSON dict with per-gold similarity stats (debug/analysis) |
| `embedding_all_scores` | JSON dict mapping generated analogy → target similarity score |
| `judge_baseline` | JSON dict with judge scores for `top1_baseline` |
| `judge_embedding` | JSON dict with judge scores for `top1_embedding` |
| `generated_subconcepts` | Sub-concepts used (withsub mode only) |
| `model` | Model name (in aggregated files only) |
| `mode` | Mode: targetonly or withsub (in aggregated files only) |

### Reranked CSV (Additional Columns)

| Column | Description |
|--------|-------------|
| `llm_rerank_order` | JSON list of reranked analogies (best to worst) |
| `llm_rerank_order_indices` | JSON list of original positions (1-20) mapping reranked items back to original list |
| `top1_rerank` | The reranker's top choice |
| `rerank_reasoning` | Reranker's explanation for the ranking |
| `judge_rerank` | LLM-as-judge evaluation of `top1_rerank` |
| `sec_generated_subconcepts` | Generated per-analogy subconcepts (targetonly mode only, JSON list) |
| `target_subconcepts` | Always filled from SCAR dataset (for both modes) |

### Judge Dictionary Structure

Both `judge_baseline`, `judge_embedding`, and `judge_rerank` have this structure:

```json
{
  "analogy": "the evaluated analogy",
  "coherence": 1,
  "mapping": 1,
  "explanatory": 1,
  "average": 1.0,
  "reasoning": "judge's explanation",
  "status": "success"
}
```

---

## Workflow Summary

### Complete Pipeline (Recommended)

1. **Precompute embeddings** (once):
   ```bash
   python core/precompute_similarity.py --mode both
   ```

2. **Run all models with auto-aggregation**:
   ```powershell
   .\scripts\launch_all_and_aggregate.ps1
   ```

3. **Rerank aggregated results** (optional):
   ```bash
   # Terminal A
   python utilities/rerank_aggregated_results.py --input results/all_results_targetonly.csv --verbose

   # Terminal B
   python utilities/rerank_aggregated_results.py --input results/all_results_withsub.csv --verbose
   ```

4. **Run multi-judge evaluation** (optional):
   ```powershell
   .\scripts\run_all_judges.ps1
   ```

5. **Analyze results** in `analysis/baselines.ipynb` and other notebooks

### Test Mode

For quick testing with limited records:

```powershell
# Generation and evaluation
.\scripts\launch_all_and_aggregate.ps1 -Test

# Reranking
python utilities/rerank_aggregated_results.py --input results/all_results_targetonly.csv --test 5 --verbose
```

### Individual Model Run

```powershell
.\scripts\run_single_model.ps1 -Model "gpt-4.1-mini"
```

---

## Notes

- **Embeddings are precomputed once** and stored in `data/`, shared across all models
- **Each model runs in a separate terminal** to avoid DSPy configuration conflicts
- **Completion markers** (`results/.markers/`) track which models have finished
- **Aggregation happens automatically** when all 12 models complete
- **Test mode** uses 3-5 records for quick validation
- **Reranking saves incrementally** after each record to prevent data loss
- **Reranking auto-resumes** if interrupted — just restart the same command
- **Subconcepts are always extracted from SCAR** for both modes in reranked files

---

## Troubleshooting

### Reranking Issues

- **JSON parsing errors**: The reranker output is validated and cleaned automatically. If parsing fails after retries, the record will be marked with an error status.

- **Missing subconcepts**: If a target has no subconcepts in the SCAR dataset, the script will use empty subconcepts and continue processing.

- **Resume mode**: If you interrupt a reranking run, just restart the same command. The script automatically detects existing output files and continues from where it left off.

- **Rate limits**: Reranking makes many LLM calls (reranker + judge per row, plus mapping calls for targetonly). The script includes automatic retries with exponential backoff.

### General Issues

- **DSPy configuration conflicts**: Each model must run in a separate terminal/process
- **Missing embeddings**: Run `core/precompute_similarity.py` first
- **Environment variables**: Ensure `.env` file is in the project root with required API keys
- **Script paths**: All PowerShell scripts are now under `scripts/`; Python scripts are under `core/` or `utilities/`
