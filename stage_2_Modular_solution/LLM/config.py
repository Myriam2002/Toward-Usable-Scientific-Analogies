"""
Configuration for LLM Baselines Pipeline
Contains model list, prompts, DSPy signatures, and shared settings
"""

import sys
import os

# Add path to easy_llm_importer
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', 'stage1_analysis', 'mapping_generation'))

# =============================================================================
# MODEL CONFIGURATION
# =============================================================================

# All 12 models from easy_llm_importer.py
MODEL_LIST = [
    "gpt-oss-20b",
    "gpt-oss-120b",
    "gpt-4.1-mini",
    "gpt-4.1-nano",
    "grok-4-fast",
    "gemini-2.5-flash-lite",
    "llama-3.1-405b-instruct",
    "meta-llama-3-1-70b-instruct",
    "meta-llama-3-1-8b-instruct",
    "deepseek-r1",
    "qwen3-14b",
    "qwen3-32b",
]

# Judge model for evaluation
JUDGE_MODEL = "gpt-4.1-mini"

# Model for sub-concept mapping (Stage 2 in withsub mode)
MAPPING_MODEL = "meta-llama-3-1-70b-instruct"

# =============================================================================
# GENERATION CONFIGURATION
# =============================================================================

NUM_ANALOGIES = 20  # Number of analogies to generate per target

# =============================================================================
# TEST MODE CONFIGURATION
# =============================================================================

TEST_MODE_RECORD_LIMIT = 3

# =============================================================================
# FILE PATHS
# =============================================================================

# Relative to this config.py file
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, '..', '..', 'data')
RESULTS_DIR = os.path.join(BASE_DIR, 'results')
SCAR_PATH = os.path.join(DATA_DIR, 'SCAR_cleaned_manually.csv')
EMBEDDINGS_PATH = os.path.join(BASE_DIR, 'gold_source_embeddings.pkl')

# Target embeddings (precomputed with OpenAI text-embedding-3-small)
TARGET_EMBEDDINGS_PATH = os.path.join(BASE_DIR, 'target_embeddings.pkl')  # Target only
TARGET_WITH_SUBCONCEPTS_EMBEDDINGS_PATH = os.path.join(BASE_DIR, 'target_with_subconcepts_embeddings.pkl')  # Target + sub-concepts

# Ensure results directory exists
os.makedirs(RESULTS_DIR, exist_ok=True)

# =============================================================================
# SEMANTIC SIMILARITY CONFIGURATION
# =============================================================================

EMBEDDING_MODEL = "all-MiniLM-L6-v2"  # SentenceTransformer model (same as mapping analysis)
SIMILARITY_THRESHOLD = 0.5  # Works well with all-MiniLM-L6-v2 (produces 0.5-0.9 for related concepts)

# NOTE: Evaluation uses ONLY source concept names (no sub-concepts) for both modes
# This provides unified, comparable metrics across targetonly and withsub modes
# EMBEDDINGS_PATH contains precomputed embeddings for all unique gold sources

# =============================================================================
# PROMPTS
# =============================================================================

# Note: This prompt will be formatted with NUM_ANALOGIES at runtime
ANALOGY_GENERATION_PROMPT_TEMPLATE = """You are an expert at creating educational analogies to help explain and simplify unfamiliar concepts.

Your task is to generate exactly {num_analogies} familiar source concepts that could serve as good analogies for the given target concept. 

A good analogy:
- source concepts are much easier to understand and more familiar to the learner
- Does not require any prior knowledge or background information to understand
- Does not use any complex or technical terms from the target or outside.
- Has meaningful structural or functional parallels between source and target
- Helps learners build understanding through comparison
- (1-2) words maximum per source

ONLY output the analogy, no need for any explanation.
"""

# Format the prompt with the actual number
ANALOGY_GENERATION_PROMPT = ANALOGY_GENERATION_PROMPT_TEMPLATE.format(num_analogies=NUM_ANALOGIES)

# =============================================================================
# LLM-AS-JUDGE CONFIGURATION
# =============================================================================

JUDGE_INSTRUCTIONS = """You are an expert evaluator of scientific analogies.

Given a target concept and a chosen source analogy, evaluate whether this is a good analogy.
A good analogy uses a FAMILIAR source concept to explain an UNFAMILIAR target concept through meaningful structural or functional parallels.

Score each dimension from 1-3:

ANALOGY_COHERENCE: Does the pairing make intuitive sense?
- 3: Immediately clear why these concepts relate
- 2: Connection exists but requires explanation
- 1: No meaningful connection; random or forced pairing

MAPPING_SOUNDNESS: Could properties/mechanisms of the source map to the target?
- 3: Clear structural or functional parallels exist
- 2: Some mappings work, others are weak or forced
- 1: No valid mappings possible; analogy is superficial

EXPLANATORY_POWER: Would this analogy help a learner understand the target?
- 3: Illuminates the target concept effectively using familiar source
- 2: Provides partial insight but with limitations
- 1: Fails to aid understanding; may confuse rather than clarify
"""


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def get_output_filename(model_name: str, mode: str, is_eval: bool = False) -> str:
    """
    Generate output filename for a model run.
    
    Args:
        model_name: Name of the model
        mode: 'targetonly' or 'withsub'
        is_eval: Whether this is an evaluation file
        
    Returns:
        Filename string
    """
    safe_model_name = model_name.replace("/", "-").replace(":", "-")
    suffix = "_eval" if is_eval else ""
    return f"LLM_{safe_model_name}_{mode}{suffix}.csv"


def get_output_path(model_name: str, mode: str, is_eval: bool = False) -> str:
    """
    Get full output path for a model run.
    
    Args:
        model_name: Name of the model
        mode: 'targetonly' or 'withsub'
        is_eval: Whether this is an evaluation file
        
    Returns:
        Full path string
    """
    filename = get_output_filename(model_name, mode, is_eval)
    return os.path.join(RESULTS_DIR, filename)
