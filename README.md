# Toward Usable Scientific Analogies

*A Modular Multi-Agent Pipeline for Generation, Explanation, Evaluation, and Visualization*

## 📖 Overview

This project develops a **multi-agent pipeline** to improve the quality and usability of scientific analogies. It targets all stages of the analogy process—source selection, property mapping, analogy explanation, evaluation, and visualization—to create pedagogically sound analogies for science education.

## 🎯 Objectives

* **Analyze** how large language models (LLMs) perform on analogy tasks using both conceptual (SCAR) and process-based (Parallel PARC) datasets.
* **Design** modular agents specialized for source finding, mapping generation, and explanation generation.
* **Integrate** these agents into a unified pipeline with feedback loops and human-in-the-loop oversight.
* **Evaluate** and **visualize** analogies to make them interpretable for educators and learners.

## 🛠️ Key Tasks

### Stage 1: Analyze LLM Challenges 

* Test whether models can **identify suitable sources** under varying information levels.
* Conduct **closed search** within a controlled corpus.
* Perform **source identification with/without domain knowledge**.
* Perform **source identification with/without explicit source/target properties**.
* Compare performance on **conceptual vs. process-based mappings**.
* Investigate how **source search extends to open domains**.
* Assess **mapping generation** under different property conditions.
* Examine **explanation generation** under progressively richer input conditions.

### Stage 2: Design Modular Multi-Agent Solution 

* Build individual agents for **source selection**, **mapping generation**, and **explanation generation**.
* Experiment with different prompts, tools, and LLMs to find the most effective configuration for each agent.

### Stage 3: Build Full Pipeline with Feedback 

* Integrate agents into a **feedback-enabled pipeline** with human-in-the-loop oversight.
* Develop a **visualization module** to translate analogies into diagrams or schematics.
* Train or fine-tune a lightweight model specialized for analogy tasks to improve consistency.

## 📊 Datasets

* **SCAR** – Conceptual analogies across 13 scientific domains.
* **Parallel PARC** – Process analogies aligning dynamic events across domains.

## 📝 Evaluation

* Each module evaluated with the most relevant benchmark dataset (SCAR for concept analogies, Parallel PARC for process analogies).
* Develop a **holistic evaluation framework** combining structural similarity metrics with qualitative dimensions (novelty, relevance, pedagogical value).
* Integrate evaluator directly into pipeline for iterative improvement.

## 🧑‍🎓 Expected Outcomes

### Theoretical

* A modular multi-agent framework for analogy generation, mapping, explanation, evaluation, and visualization.
* A new holistic evaluation framework for scientific analogies.
* Comparative analysis of LLM performance (concept vs. process, closed vs. open search, explanation quality).
* Formalization of text-to-visual mapping.

### Practical

* Prototype pipeline for automatic analogy generation, evaluation, refinement, and visualization.
* New benchmark results on SCAR and Parallel PARC.
* Feedback-enabled evaluation module reusable across research and applications.

## 📂 Repository Structure

```
├── stage1_analysis/             # LLM challenge analysis (Stage 1)
│   ├── source_finding/          # Closed/open search experiments
│   ├── mapping_generation/      # Entity & relation alignment experiments
│   └── explanation_generation/  # Explanation quality experiments
│
├── stage_2_Modular_solution/    # Modular multi-agent solution (Stage 2)
│   └── LLM/                     # LLM baselines pipeline
│       ├── core/                # Core pipeline scripts (generation, evaluation, judging)
│       ├── utilities/           # Post-processing (aggregation, reranking, reruns)
│       ├── scripts/             # PowerShell launch scripts
│       │   ├── run_models/      # Per-model generation scripts (12 models)
│       │   └── run_judges/      # Per-judge evaluation scripts (5 judge models)
│       ├── analysis/            # Analysis notebooks
│       ├── notebooks/           # Supplementary notebooks (human annotation, WordNet)
│       ├── data/                # Precomputed embeddings
│       ├── human_annotation/    # Human annotation data and forms
│       └── results/             # All output CSVs and visualizations
│
├── stage2_modular_agents/       # Modular agent implementations
│
└── data/                        # SCAR, Parallel PARC, preprocessed data

```


