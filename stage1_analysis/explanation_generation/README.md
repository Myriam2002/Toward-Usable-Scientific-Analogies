# Explanation Generation Experiments

This directory contains the experimental framework for generating and evaluating scientific analogy explanations using various LLMs across different input configurations.

## Overview

This experiment evaluates how different Large Language Models generate explanations for scientific analogies under varying levels of input information. The goal is to understand which models and which input configurations produce the highest quality explanations when compared to expert-written golden explanations.

### Methodology

**Input Variables (6 Settings):**
1. **Concepts Only** (`none`) - Only source and target concept names
2. **Concepts + Descriptions** (`none_description`) - Concept names with background descriptions
3. **Concepts + Properties** (`unpaired_properties`) - Concept names with property lists (not paired)
4. **Concepts + Properties + Descriptions** (`unpaired_properties_description`)
5. **Concepts + Paired Mappings** (`paired_properties`) - Concept names with pre-mapped property pairs
6. **Concepts + Paired Mappings + Descriptions** (`paired_properties_description`)

**Evaluation Metric:**
- SBERT (Sentence-BERT) semantic similarity between generated and golden explanations
- Scale: 0.0-1.0, where higher values indicate greater semantic similarity

**Models Tested:**
All available models in the LLM pipeline are automatically included in experiments.

## Quick Start

### Running Experiments

```bash
# Test mode (5 rows, all models)
python run_experiments.py --test-mode

# Run single setting (full dataset, all models)
python run_experiments.py --setting none

# Run single model (full dataset, all settings)
python run_experiments.py --model gpt-4o-mini

# Run specific setting with specific model
python run_experiments.py --setting unpaired_properties --model gpt-4o-mini

# Run all settings and all models sequentially (slowest)
python run_experiments.py
```

### Parallel Execution (Recommended)

**Option 1: Parallelize by Settings** (run different settings simultaneously)
```bash
# Terminal 1
python run_experiments.py --setting none

# Terminal 2
python run_experiments.py --setting none_description

# Terminal 3
python run_experiments.py --setting unpaired_properties
# ... etc for remaining settings
```

**Option 2: Parallelize by Models** (run different models simultaneously for same setting)
```bash
# Terminal 1
python run_experiments.py --setting unpaired_properties --model gpt-4o-mini

# Terminal 2
python run_experiments.py --setting unpaired_properties --model claude-3-5-sonnet-20241022

# Terminal 3
python run_experiments.py --setting unpaired_properties --model gemini-1.5-flash
# ... etc for remaining models (12 terminals total)

# After all models complete, combine their checkpoints:
python combine_checkpoints.py --setting unpaired_properties
```

Or use the automated script (Option 1):
```bash
# Windows
.\run_parallel_experiments.ps1

# Linux/Mac
./run_parallel_experiments.sh
```

### Combining Checkpoints (After Parallel Model Execution)

When running models in parallel with `--model` flag, combine individual checkpoints:

```bash
# Combine checkpoints for one setting
python combine_checkpoints.py --setting unpaired_properties

# Combine checkpoints for all settings
python combine_checkpoints.py --setting all

# Use custom directory
python combine_checkpoints.py --setting none --output-dir custom/path
```

### Analyzing Results

```bash
# Generate all visualizations and statistics
python analyze_results.py

# Custom output location
python analyze_results.py --output-dir path/to/output
```

## Key Features

- ✅ **Automatic model detection** - All models are included automatically
- ✅ **Checkpointing** - Results saved per model immediately
- ✅ **Rate limit handling** - Exponential backoff and configurable delays
- ✅ **Progress tracking** - Real-time progress bars
- ✅ **Comprehensive analysis** - 8 visualization types + statistical summaries

## Analysis Pipeline

### Step 1: Data Preparation
Input data from `../../data/SCAR_cleaned_manually.csv` containing:
- Source and target concepts
- Property mappings
- Descriptions
- Golden explanations (ground truth)

### Step 2: Prompt Generation
For each setting, construct prompts with the appropriate information:
- Minimal: Only concept names
- Maximal: Concepts + paired property mappings + descriptions

### Step 3: Model Execution
- Each model generates explanations for all examples
- Results checkpointed immediately per model
- Errors logged but don't stop execution
- Rate limits handled with exponential backoff

### Step 4: Evaluation
- Generated explanations compared to golden explanations using SBERT
- Similarity scores calculated for each model-setting-example combination
- Combined results saved in CSV and JSON formats

### Step 5: Visualization & Analysis
Run `analyze_results.py` to generate:

**8 Core Visualizations:**
1. Heatmap of model × setting performance
2. Model comparison boxplots (overall distribution)
3. Setting comparison boxplots (overall distribution)
4. Model rankings by setting (6 subplots)
5. Error rate analysis (by model and setting)
6. Top vs bottom models comparison
7. Setting impact heatmap (relative to baseline)
8. Consistency analysis (standard deviation)

**Statistical Outputs:**
- `summary_statistics.json` - Complete numerical results
- `model_rankings.csv` - Model performance rankings
- `setting_rankings.csv` - Setting effectiveness rankings

## Output Structure

```
checkpoints/explanation_generation/
├── none_gpt4_checkpoint.csv              # Per-model checkpoints
├── none_gpt4_checkpoint.json
├── none_ALL_MODELS_combined.csv          # Combined per-setting
├── none_ALL_MODELS_combined.json
├── ...                                    # (repeated for each setting)

results/explanation_generation/
├── 1_heatmap_model_setting_performance.png
├── 2_model_comparison_boxplot.png
├── ...                                    # (all 8 visualizations)
├── summary_statistics.json
├── model_rankings.csv
└── setting_rankings.csv
```

## Advanced Configuration

### List Available Models
To see which models you can use with the `--model` parameter, check the output when running any experiment:

```bash
# The script will print all available models at startup
python run_experiments.py --test-mode
```

### Rate Limit Control
```bash
# Increase delays between API calls
python run_experiments.py --request-delay 2.0 --retry-delay 10 --max-retries 5

# Configuration options:
#   --max-retries N      : Retry attempts (default: 3)
#   --retry-delay N      : Initial retry delay with exponential backoff (default: 5s)
#   --request-delay N    : Delay between successful calls (default: 1.0s)
```

### Custom Paths
```bash
# Custom data and output locations
python run_experiments.py \
  --data-path /path/to/data.csv \
  --output-dir results/custom_experiment \
  --setting none
```

### Test Mode Options
```bash
# Test with custom row count
python run_experiments.py --test-mode --test-rows 20 --setting paired_properties
```

## Interpreting Results

### SBERT Similarity Scores
- **0.9-1.0:** Excellent semantic similarity
- **0.75-0.9:** Good similarity
- **0.6-0.75:** Moderate similarity
- **< 0.6:** Poor similarity

### Key Analysis Questions

**Which model performs best?**
→ Check `2_model_comparison_boxplot.png` and `model_rankings.csv`

**Which setting produces best explanations?**
→ Check `3_setting_comparison_boxplot.png` and `setting_rankings.csv`

**Does more information always help?**
→ Check `7_setting_impact_on_models.png` (relative to baseline)

**Which models are most consistent?**
→ Check `8_consistency_analysis.png` (lower std = more reliable)

**Best model-setting combination?**
→ Check `1_heatmap_model_setting_performance.png` and `summary_statistics.json` → `top_10_combinations`

## Recommended Workflow

### Workflow A: Parallel by Settings (6 terminals)
```bash
# 1. Test on small sample
python run_experiments.py --test-mode --test-rows 5 --setting none

# 2. Run all settings in parallel (6 terminals or script)
.\run_parallel_experiments.ps1  # or .sh on Linux/Mac

# 3. Generate analysis
python analyze_results.py

# 4. Review key outputs
#    - results/explanation_generation/1_heatmap_model_setting_performance.png
#    - results/explanation_generation/summary_statistics.json
```

### Workflow B: Parallel by Models (12 terminals, fastest for single setting)
```bash
# 1. Test with one model
python run_experiments.py --test-mode --setting unpaired_properties --model gpt-4o-mini

# 2. Run all 12 models in parallel (see commands in section above)
#    Terminal 1: python run_experiments.py --setting unpaired_properties --model gpt-oss-20b
#    Terminal 2: python run_experiments.py --setting unpaired_properties --model gpt-oss-120b
#    ... etc

# 3. Combine checkpoints
python combine_checkpoints.py --setting unpaired_properties

# 4. Generate analysis
python analyze_results.py

# 5. Review results
#    - results/explanation_generation/model_rankings.csv
```

## Troubleshooting

**Rate Limits:**
Increase delays: `--request-delay 2.0 --retry-delay 10`

**File Locked Errors:**
Close CSV viewers (Excel) and run one setting per terminal

**Script Execution (Windows):**
```powershell
Set-ExecutionPolicy RemoteSigned
```

## Dependencies

See parent directory `requirements.txt` for required packages. Key dependencies:
- `litellm` - LLM API interface
- `sentence-transformers` - SBERT similarity
- `pandas`, `numpy` - Data manipulation
- `matplotlib`, `seaborn` - Visualizations
- `tqdm` - Progress bars

## Files

- `run_experiments.py` - Main experiment runner (supports `--model` for individual models)
- `combine_checkpoints.py` - Combines individual model checkpoints into ALL_MODELS files
- `analyze_results.py` - Visualization and analysis generator
- `explanation_evaluation.py` - SBERT evaluation utilities
- `run_parallel_experiments.ps1/.sh` - Automated parallel execution scripts
- `Explanation_genertion_evaluation.ipynb` - Interactive analysis notebook

## Additional Help

```bash
python run_experiments.py --help
python combine_checkpoints.py --help
python analyze_results.py --help
```

