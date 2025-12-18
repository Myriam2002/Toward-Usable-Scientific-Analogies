"""
Pipeline Configuration
Manages module selection, ordering, and input format configuration
"""

from typing import List, Dict, Any, Optional, Literal
from dataclasses import dataclass, field
from enum import Enum


class InputFormat(Enum):
    """Input format types"""
    TARGET_ONLY = "target_only"
    TARGET_PROPERTIES = "target_properties"
    TARGET_DESCRIPTION = "target_description"
    TARGET_PROPERTIES_DESCRIPTION = "target_properties_description"


@dataclass
class ModuleConfig:
    """Configuration for a single module"""
    module_type: str  # e.g., "analogy_type_classifier", "source_finder", etc.
    implementation: str  # e.g., "DSPyAnalogyTypeClassifier", "EmbeddingSourceFinder"
    enabled: bool = True
    params: Dict[str, Any] = field(default_factory=dict)


@dataclass
class PipelineConfig:
    """
    Configuration for the entire pipeline
    """
    # Module configurations (in order)
    modules: List[ModuleConfig] = field(default_factory=list)
    
    # Input format
    input_format: InputFormat = InputFormat.TARGET_PROPERTIES_DESCRIPTION
    
    # Evaluation settings
    run_baselines: bool = True
    run_scar_evaluation: bool = True
    llm_judge_threshold: float = 0.7
    
    # Paths
    corpus_path: str = "../../data/SCAR_cleaned_manually.csv"
    scar_data_path: str = "../../data/SCAR_cleaned_manually.csv"
    
    # LLM settings
    default_model: str = "gpt-4o-mini"
    
    # Experiment tracking
    experiment_name: Optional[str] = None
    save_results: bool = True
    results_dir: str = "./results"
    
    def add_module(
        self,
        module_type: str,
        implementation: str,
        enabled: bool = True,
        **params
    ):
        """
        Add a module to the pipeline
        
        Args:
            module_type: Type of module (e.g., "analogy_type_classifier")
            implementation: Implementation class name
            enabled: Whether module is enabled
            **params: Module-specific parameters
        """
        self.modules.append(ModuleConfig(
            module_type=module_type,
            implementation=implementation,
            enabled=enabled,
            params=params
        ))
    
    def get_enabled_modules(self) -> List[ModuleConfig]:
        """Get list of enabled modules in order"""
        return [m for m in self.modules if m.enabled]
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary"""
        return {
            'modules': [
                {
                    'module_type': m.module_type,
                    'implementation': m.implementation,
                    'enabled': m.enabled,
                    'params': m.params
                }
                for m in self.modules
            ],
            'input_format': self.input_format.value,
            'run_baselines': self.run_baselines,
            'run_scar_evaluation': self.run_scar_evaluation,
            'llm_judge_threshold': self.llm_judge_threshold,
            'corpus_path': self.corpus_path,
            'scar_data_path': self.scar_data_path,
            'default_model': self.default_model,
            'experiment_name': self.experiment_name
        }
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'PipelineConfig':
        """Create config from dictionary"""
        config = cls()
        config.modules = [
            ModuleConfig(
                module_type=m['module_type'],
                implementation=m['implementation'],
                enabled=m.get('enabled', True),
                params=m.get('params', {})
            )
            for m in config_dict.get('modules', [])
        ]
        config.input_format = InputFormat(config_dict.get('input_format', 'target_properties_description'))
        config.run_baselines = config_dict.get('run_baselines', True)
        config.run_scar_evaluation = config_dict.get('run_scar_evaluation', True)
        config.llm_judge_threshold = config_dict.get('llm_judge_threshold', 0.7)
        config.corpus_path = config_dict.get('corpus_path', '../../data/SCAR_cleaned_manually.csv')
        config.scar_data_path = config_dict.get('scar_data_path', '../../data/SCAR_cleaned_manually.csv')
        config.default_model = config_dict.get('default_model', 'gpt-4o-mini')
        config.experiment_name = config_dict.get('experiment_name')
        return config


# Predefined configurations
def get_default_config() -> PipelineConfig:
    """Get default pipeline configuration"""
    config = PipelineConfig()
    
    # Default pipeline: Source Finder -> Property Matcher -> Explanation Generator
    config.add_module(
        "source_finder",
        "EmbeddingSourceFinder",
        corpus_path=config.corpus_path,
        embedding_mode="name_background",
        top_k=10
    )
    config.add_module(
        "property_matcher",
        "DSPyPropertyMatcher",
        model_name=config.default_model,
        use_description=True
    )
    config.add_module(
        "explanation_generator",
        "DSPyExplanationGenerator",
        model_name=config.default_model,
        use_description=True,
        use_paired_properties=True
    )
    
    return config


def get_full_config() -> PipelineConfig:
    """Get full pipeline configuration with all modules"""
    config = PipelineConfig()
    
    # Full pipeline: Type Classifier -> Source Finder -> Property Matcher -> Evaluator -> Improver -> Explanation Generator
    config.add_module(
        "analogy_type_classifier",
        "DSPyAnalogyTypeClassifier",
        model_name=config.default_model
    )
    config.add_module(
        "source_finder",
        "EmbeddingSourceFinder",
        corpus_path=config.corpus_path,
        embedding_mode="name_background",
        top_k=10
    )
    config.add_module(
        "property_matcher",
        "DSPyPropertyMatcher",
        model_name=config.default_model,
        use_description=True
    )
    config.add_module(
        "evaluator",
        "LLMEvaluator",
        model_name=config.default_model
    )
    config.add_module(
        "improver",
        "LLMImprover",
        model_name=config.default_model
    )
    config.add_module(
        "explanation_generator",
        "DSPyExplanationGenerator",
        model_name=config.default_model,
        use_description=True,
        use_paired_properties=True
    )
    
    return config

