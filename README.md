# Teaching Through Analogies: A Modular Pipeline for Educational Analogy Generation

---

## Overview

This project investigates how large language models (LLMs) can support the generation of usable scientific analogies. It proceeds in two stages: first analyzing where LLMs struggle across analogy subtasks, then building and evaluating a full pipeline for LLM-based analogy source selection with multi-model and human annotation evaluation.

**Datasets used:**
- **SCAR** — Scientific Concept Analogy Resource: conceptual analogies across 13 scientific domains
- **Parallel PARC** — Process analogies aligning dynamic events across domains

---

## Repository Structure

```
├── data/                            # SCAR and Parallel PARC datasets
│
├── stage1_analysis/                 # Stage 1: LLM challenge analysis
│   ├── source_finding/              # Closed-corpus source retrieval experiments
│   ├── mapping_generation/          # Property extraction and mapping experiments
│   └── explanation_generation/      # Explanation generation under varying input conditions
│
└── stage_2_Modular_solution/
    └── LLM/                         # Stage 2: LLM baselines pipeline
        ├── core/                    # Generation, evaluation, and judging scripts
        ├── utilities/               # Aggregation, reranking, rerun utilities
        ├── scripts/                 # PowerShell launch scripts
        │   ├── run_models/          # Per-model generation scripts (12 models)
        │   └── run_judges/          # Per-judge evaluation scripts (5 judge models)
        ├── analysis/                # Analysis and visualization notebooks
        ├── notebooks/               # Supplementary notebooks (human annotation, WordNet)
        ├── data/                    # Precomputed embeddings
        ├── human_annotation/        # Human annotation data and forms
        └── results/                 # All output CSVs and figures
```

---

## Stage 1 — Analyzing LLM Challenges

Three analogy subtasks were studied to understand where and how LLMs fail.

### Source Finding (`stage1_analysis/source_finding/`)

Given a target concept and a closed corpus of candidate sources, the task is to retrieve the best analogical source. Two approaches were compared:

- **Embedding-based**: OpenAI embeddings + cosine similarity, across four embedding modes (name only, name + background, name + properties, name + properties + background)
- **Tournament-style elimination**: LLM evaluates sources in configurable batches; winners advance until one remains

### Mapping Generation (`stage1_analysis/mapping_generation/`)

Given a source–target pair, the task is to extract and align shared structural or functional properties. Multiple LLMs were tested under varying property availability conditions. Includes `easy_llm_importer.py`, a unified client that routes calls to OpenAI, OpenRouter, or DeepInfra by model name.

### Explanation Generation (`stage1_analysis/explanation_generation/`)

Given a source–target pair, the task is to generate a natural language explanation of the analogy. Six input configurations were evaluated (from concept names only up to fully paired property mappings with descriptions). Evaluated using SBERT semantic similarity against expert-written golden explanations.

---

## Stage 2 — LLM Baselines Pipeline

A complete evaluation pipeline for analogy source selection across 12 LLMs on the SCAR dataset. See [`stage_2_Modular_solution/LLM/README.md`](stage_2_Modular_solution/LLM/README.md) for full documentation.

### Models Evaluated (12)

```
gpt-oss-20b, gpt-oss-120b, gpt-4.1-mini, gpt-4.1-nano,
grok-4-fast, gemini-2.5-flash-lite, llama-3.1-405b-instruct,
meta-llama-3-1-70b-instruct, meta-llama-3-1-8b-instruct,
deepseek-r1, qwen3-14b, qwen3-32b
```

### Generation Modes

- **targetonly** — model receives only the target concept name
- **withsub** — model receives target + sub-concepts (key properties from SCAR)

Each model generates 20 candidate source analogies per target.

### Evaluation

| Method | Description |
|--------|-------------|
| Hit@K (exact) | Exact string match of generated analogy against gold sources, K = 1–20 |
| Hit@K (semantic) | Cosine similarity ≥ 0.5 against gold source embeddings (all-MiniLM-L6-v2) |
| Top-1 Baseline | First generated analogy |
| Top-1 Embedding | Analogy with highest embedding similarity to target (OpenAI embeddings) |
| LLM-as-Judge | Scores top-1 choices on Coherence, Mapping Soundness, and Explanatory Power (1–3 scale) |

### Reranking

All 20 generated analogies are re-ranked by an LLM reranker (`meta-llama-3-1-70b-instruct`) using target subconcepts. The top-1 reranked choice is then re-evaluated by the judge.

### Multi-Judge Evaluation

To assess judge reliability, the top-1 analogies were re-evaluated by 5 judge models across 2 prompting modes:

- **Judges**: `gpt-4.1-mini`, `gemini-2.5-flash-lite`, `deepseek-r1`, `claude-sonnet-4.6`, `mimo-v2-pro`
- **Modes**: `3scale` (standard), `3scale_fewshot` (with few-shot examples)
- Results and inter-judge agreement analysis in `results/upgraded_llm/` and `results/judge_analysis/`

### Human Annotation

A subset of 15 targets was evaluated by human annotators using the same 3-dimension scoring rubric. Annotator agreement and correlation with LLM judges are analyzed in `notebooks/human_annotation_analysis.ipynb` and `results/human_annotation/`.

---

## Setup

### Requirements

```bash
pip install -r requirements.txt
```

### API Keys

Create a `.env` file in the project root:

```
OPENAI_API_KEY=your_key_here
OPENROUTER_API_KEY=your_key_here
DEEPINFRA_API_KEY=your_key_here
```

---

## Key Results

All results are in `stage_2_Modular_solution/LLM/results/`. Publication-ready figures are in `results/final_visualizations/`. Key output files:

| File | Contents |
|------|----------|
| `results/all_results_targetonly.csv` | Aggregated results, all 12 models, targetonly mode |
| `results/all_results_withsub.csv` | Aggregated results, all 12 models, withsub mode |
| `results/all_results_*_rerank.csv` | Reranked results |
| `results/upgraded_llm/` | Multi-judge evaluation outputs |
| `results/human_annotation/` | Human vs. LLM judge comparison |
| `results/judge_analysis/` | Inter-judge agreement analysis |
