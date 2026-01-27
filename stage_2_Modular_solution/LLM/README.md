# LLM Baselines Pipeline

This folder contains the complete pipeline for evaluating LLM-based analogy generation across 12 different language models.

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
              └───────────────────────────────────────┘
```

---

## Methodology (Numbered Steps)

### Step 1: Precomputation (Run Once)

1. **Load SCAR Dataset**: Load the Scientific Concept Analogy Resource dataset containing target-source analogy pairs.

2. **Precompute Gold Source Embeddings**:
   - Extract all unique gold sources (correct answers) from the dataset
   - Compute embeddings using **SentenceTransformer all-MiniLM-L6-v2**
   - Save to `gold_source_embeddings.pkl`

3. **Precompute Target Embeddings**:
   - Compute target-only embeddings using **OpenAI text-embedding-3-small**
   - Compute target-with-subconcepts embeddings (target + sub-concepts combined)
   - Save to `target_embeddings.pkl` and `target_with_subconcepts_embeddings.pkl`

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
    - Judge model: **gpt-4.1-mini**
    - Evaluates BOTH top1_baseline and top1_embedding
    - Scoring dimensions (1-3 scale):
      - **Analogy Coherence**: Does the pairing make intuitive sense?
      - **Mapping Soundness**: Can source properties map to target properties?
      - **Explanatory Power**: Would this help a learner understand the target?
    - Stores results in `judge_baseline` and `judge_embedding` columns

---

### Step 4: Aggregation & Analysis

11. **Compute Summary Metrics**:
    - **Hit@K** (K=1,3,5,10,20): Percentage of targets where a gold source was found in top K
    - **Average Judge Scores**: Mean scores across all targets
    - **Top-1 Embedding Score**: Average embedding similarity for selected analogies

12. **Generate Visualizations**:
    - Hit@K comparison across models
    - LLM Judge score distributions
    - Mode comparison (targetonly vs withsub)

13. **Export Results**:
    - Individual CSV files per model/mode: `results/LLM_{model}_{mode}_eval.csv`
    - Contains all analogies, scores, rankings, and judge evaluations

---

## File Structure

```
LLM/
├── config.py                 # Configuration (models, prompts, paths)
├── run_model.py              # Analogy generation script
├── evaluate_model.py         # Evaluation script (gold matching + judge)
├── precompute_similarity.py  # Embedding precomputation & semantic matching
├── baselines.ipynb           # Main orchestration notebook
├── run_all_models.ps1        # PowerShell script to run all models in parallel
├── gold_source_embeddings.pkl        # Precomputed gold source embeddings
├── target_embeddings.pkl             # Precomputed target-only embeddings
├── target_with_subconcepts_embeddings.pkl  # Precomputed target+subconcepts embeddings
└── results/                  # Output directory for CSV results
```

---

## Quick Start

### 1. Precompute Embeddings (Run Once)

```bash
python precompute_similarity.py --mode both
```

Or run Cell 2 in `baselines.ipynb`.

### 2. Run All Models (Parallel)

```powershell
.\run_all_models.ps1
```

This runs all 12 models with both modes (24 total jobs) using 12 parallel workers.

### 3. Run Single Model

```bash
# targetonly mode
python run_model.py --model gpt-4.1-mini --mode targetonly

# withsub mode
python run_model.py --model gpt-4.1-mini --mode withsub
```

### 4. Analyze Results

Run the analysis cells in `baselines.ipynb` to:
- Aggregate results
- Compute metrics
- Generate visualizations

---

## Key Configuration (config.py)

| Parameter | Value | Description |
|-----------|-------|-------------|
| `NUM_ANALOGIES` | 20 | Number of analogies generated per target |
| `SIMILARITY_THRESHOLD` | 0.5 | Semantic match threshold |
| `EMBEDDING_MODEL` | all-MiniLM-L6-v2 | For gold source matching |
| `JUDGE_MODEL` | gpt-4.1-mini | LLM-as-judge model |
| `MAPPING_MODEL` | llama-3.1-405b-instruct | Sub-concept extraction model |

---

## Output Columns (Evaluation CSV)

| Column | Description |
|--------|-------------|
| `target` | Target concept |
| `all_gold_sources` | All correct source analogies (JSON list) |
| `analogy_1` to `analogy_20` | Generated analogies |
| `top1_baseline` | First generated analogy |
| `top1_embedding` | Best analogy by embedding similarity to target |
| `top1_embedding_score` | Similarity score for top1_embedding |
| `gold_ranks` | Positions where generated analogies match gold (exact) |
| `sem_gold_ranks` | Positions where generated analogies match gold (semantic) |
| `judge_baseline` | LLM judge scores for top1_baseline (dict) |
| `judge_embedding` | LLM judge scores for top1_embedding (dict) |
| `generated_subconcepts` | Sub-concepts used (withsub mode only) |
