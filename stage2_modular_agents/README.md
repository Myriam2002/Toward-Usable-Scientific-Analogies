# Modular Analogy Generation Pipeline

A flexible, modular pipeline for generating scientific analogies with configurable modules, multiple input formats, and comprehensive evaluation.

## Overview

This pipeline allows you to test different module implementations and configurations to find the best settings for generating scientific analogies. Each module is implemented as a separate Python file with a base class and child classes for different versions.

## Architecture

### Module Structure

Each module is in its own file with:
- **Base class** defining the interface
- **Child classes** for different implementations
- **Standardized input/output** format using `AnalogyData`

### Modules

1. **Analogy Type Classifier** (`modules/analogy_type_classifier.py`)
   - Determines best analogy type (SIMILARITY, FUNCTION, PART-WHOLE, etc.)
   - Implementations: `DSPyAnalogyTypeClassifier`, `SimpleAnalogyTypeClassifier`

2. **Source Finder** (`modules/source_finder.py`)
   - Finds analogous source concepts for a given target
   - Implementations: `EmbeddingSourceFinder` (RAG/embeddings), `LLMSourceFinder` (LLM open search)

3. **Property Matcher** (`modules/property_matcher.py`)
   - Maps properties between target and source concepts
   - Implementations: `DSPyPropertyMatcher`

4. **Evaluator** (`modules/evaluator.py`)
   - Evaluates and reranks analogies
   - Implementations: `LLMEvaluator`

5. **Improver** (`modules/improver.py`)
   - Refines/improves analogies based on evaluation feedback
   - Implementations: `LLMImprover`

6. **Explanation Generator** (`modules/explanation_generator.py`)
   - Generates explanations for analogies
   - Implementations: `DSPyExplanationGenerator`

### Baselines

1. **EmbeddingBaseline** (`baselines/embedding_baseline.py`)
   - Embedding-based retrieval, take top 3

2. **EmbeddingLLMBaseline** (`baselines/embedding_llm_baseline.py`)
   - Embedding retrieval, LLM chooses 3 from top 10

### Evaluation

1. **SCAREvaluator** (`evaluation/scar_evaluator.py`)
   - Compares generated analogies against SCAR golden standard
   - Checks source match, mapping match, explanation match

2. **LLMJudge** (`evaluation/llm_judge.py`)
   - Evaluates analogy quality using LLM when SCAR match is not found
   - Returns quality score (0-1) with configurable threshold

## Quick Start

### 1. Open the Main Notebook

```bash
jupyter notebook pipeline_main.ipynb
```

### 2. Configure the Pipeline

In the notebook, choose your configuration:

```python
CONFIG_TYPE = "default"  # Options: "default", "full", "custom"
```

- **Default**: Source Finder → Property Matcher → Explanation Generator
- **Full**: All modules in sequence
- **Custom**: Build your own configuration

### 3. Run Single Example

```python
result = runner.run(
    target_name="biological clock",
    target_description="...",
    target_properties=["changes", "state", "adjust"]
)
```

### 4. Run Batch Evaluation

```python
results = runner.run_batch(inputs, save_results=True)
```

## Input Formats

The pipeline supports multiple input configurations:

- `TARGET_ONLY`: Just target concept name
- `TARGET_PROPERTIES`: Target + properties list
- `TARGET_DESCRIPTION`: Target + description
- `TARGET_PROPERTIES_DESCRIPTION`: Target + properties + description

Configure in the notebook:

```python
config.input_format = InputFormat.TARGET_PROPERTIES_DESCRIPTION
```

## Configuration

### Creating a Custom Configuration

```python
from pipeline_config import PipelineConfig, InputFormat

config = PipelineConfig()

# Add modules in desired order
config.add_module(
    "source_finder",
    "EmbeddingSourceFinder",
    corpus_path="../../data/SCAR_cleaned_manually.csv",
    embedding_mode="name_background",
    top_k=10
)

config.add_module(
    "property_matcher",
    "DSPyPropertyMatcher",
    model_name="gpt-4o-mini",
    use_description=True
)

config.add_module(
    "explanation_generator",
    "DSPyExplanationGenerator",
    model_name="gpt-4o-mini",
    use_description=True,
    use_paired_properties=True
)

# Configure evaluation
config.run_baselines = True
config.run_scar_evaluation = True
config.llm_judge_threshold = 0.7
```

### Module Ordering

Modules are executed in the order they are added. You can reorder them as needed:

```python
config = PipelineConfig()
config.add_module("analogy_type_classifier", "DSPyAnalogyTypeClassifier")
config.add_module("source_finder", "EmbeddingSourceFinder", ...)
config.add_module("property_matcher", "DSPyPropertyMatcher", ...)
config.add_module("evaluator", "LLMEvaluator", ...)
config.add_module("improver", "LLMImprover", ...)
config.add_module("explanation_generator", "DSPyExplanationGenerator", ...)
```

## Data Flow

```
Input (target + optional properties/description)
  ↓
[Analogy Type Classifier] (optional)
  ↓
[Source Finder]
  ↓
[Property Matcher]
  ↓
[Evaluator/Reranker] (optional)
  ↓
[LLM Improver] (optional)
  ↓
[Explanation Generator]
  ↓
Output (complete analogy)
  ↓
[Evaluation vs Baselines & Golden Standard]
```

## Results

Results are saved in two formats:

1. **JSON**: Complete results with all intermediate data
   - Location: `results/{experiment_name}_{timestamp}.json`
   - Includes: Full pipeline output, baseline results, SCAR evaluation

2. **CSV**: Summary for easy analysis
   - Location: `results/{experiment_name}_{timestamp}.csv`
   - Includes: Key metrics, scores, matches

## Evaluation Metrics

The pipeline automatically calculates:

- **Source Match**: Whether generated source matches SCAR golden source
- **Mapping Match**: Whether property mappings match golden mappings
- **Explanation Match**: Whether explanation matches golden explanation
- **LLM Judge Score**: Quality score (0-1) when source doesn't match
- **Evaluation Scores**: Relevance, clarity, accuracy, overall scores

## Adding New Module Implementations

To add a new implementation of a module:

1. Create a new class inheriting from the base module class
2. Implement the required abstract methods
3. Add it to the module factory in `pipeline_runner.py`:

```python
elif module_type == "your_module_type":
    if implementation == "YourNewImplementation":
        return YourNewImplementation(**params)
```

## Requirements

- Python 3.8+
- dspy-ai
- pandas
- openai (for embeddings)
- sentence-transformers (for evaluation)

See `requirements.txt` in the project root.

## File Structure

```
stage2_modular_agents/
├── modules/
│   ├── __init__.py
│   ├── base_module.py          # Base class and AnalogyData
│   ├── analogy_type_classifier.py
│   ├── source_finder.py
│   ├── property_matcher.py
│   ├── evaluator.py
│   ├── improver.py
│   └── explanation_generator.py
├── baselines/
│   ├── __init__.py
│   ├── embedding_baseline.py
│   └── embedding_llm_baseline.py
├── evaluation/
│   ├── __init__.py
│   ├── scar_evaluator.py
│   └── llm_judge.py
├── pipeline_main.ipynb         # Main notebook
├── pipeline_config.py           # Configuration system
├── pipeline_runner.py           # Pipeline execution
└── README.md                    # This file
```

## Examples

See `pipeline_main.ipynb` for complete examples including:
- Single example execution
- Batch processing
- Results analysis
- Baseline comparison

## Troubleshooting

### Import Errors

If you encounter import errors, make sure:
1. You're running from the `stage2_modular_agents/` directory
2. The `stage1_analysis/` directory is accessible (for `easy_llm_importer`)
3. All dependencies are installed

### Module Creation Errors

If a module fails to initialize:
- Check that all required parameters are provided
- Verify API keys are set (OPENAI_API_KEY, etc.)
- Check that corpus path is correct

### Evaluation Errors

If evaluation fails:
- Ensure SCAR data path is correct
- Check that LLM judge model is available
- Verify threshold is set appropriately

## License

See project root LICENSE file.

