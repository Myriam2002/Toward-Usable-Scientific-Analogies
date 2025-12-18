"""
Pipeline Runner
Executes modules in sequence according to configuration
"""

import sys
from pathlib import Path
from typing import Optional, Dict, Any, List
import json
import pandas as pd
from datetime import datetime

from pipeline_config import PipelineConfig, InputFormat
from modules.base_module import AnalogyData
from modules.analogy_type_classifier import DSPyAnalogyTypeClassifier, SimpleAnalogyTypeClassifier
from modules.source_finder import EmbeddingSourceFinder, LLMSourceFinder
from modules.property_matcher import DSPyPropertyMatcher
from modules.evaluator import LLMEvaluator
from modules.improver import LLMImprover
from modules.explanation_generator import DSPyExplanationGenerator
from baselines.embedding_baseline import EmbeddingBaseline
from baselines.embedding_llm_baseline import EmbeddingLLMBaseline
from evaluation.scar_evaluator import SCAREvaluator
from evaluation.llm_judge import LLMJudge

# Import LLM client for shared instances
current_dir = Path(__file__).parent.parent
mapping_dir = current_dir / "stage1_analysis" / "mapping_generation"
if str(mapping_dir) not in sys.path:
    sys.path.insert(0, str(mapping_dir))

try:
    from easy_llm_importer import LLMClient
    LLM_CLIENT_AVAILABLE = True
except ImportError:
    LLM_CLIENT_AVAILABLE = False
    LLMClient = None


class PipelineRunner:
    """
    Main pipeline runner that executes modules in sequence
    """
    
    def __init__(
        self,
        config: PipelineConfig,
        llm_client: Optional[LLMClient] = None
    ):
        """
        Initialize pipeline runner
        
        Args:
            config: Pipeline configuration
            llm_client: Optional shared LLM client
        """
        self.config = config
        self.llm_client = llm_client or (LLMClient() if LLM_CLIENT_AVAILABLE else None)
        
        # Initialize modules based on config
        self.modules = []
        self._initialize_modules()
        
        # Initialize evaluators
        self.scar_evaluator = None
        self.llm_judge = None
        if config.run_scar_evaluation:
            self.scar_evaluator = SCAREvaluator(config.scar_data_path)
            self.llm_judge = LLMJudge(
                model_name=config.default_model,
                llm_client=self.llm_client
            )
        
        # Initialize baselines
        self.baselines = {}
        if config.run_baselines:
            self._initialize_baselines()
    
    def _initialize_modules(self):
        """Initialize modules based on configuration"""
        for module_config in self.config.get_enabled_modules():
            module = self._create_module(module_config)
            if module:
                self.modules.append(module)
    
    def _create_module(self, module_config) -> Optional[Any]:
        """Create a module instance from configuration"""
        module_type = module_config.module_type
        implementation = module_config.implementation
        params = module_config.params.copy()
        
        # Add shared LLM client if available
        if 'llm_client' not in params and self.llm_client:
            params['llm_client'] = self.llm_client
        
        # Add default model if not specified
        if 'model_name' not in params:
            params['model_name'] = self.config.default_model
        
        # Add corpus path if needed
        if 'corpus_path' not in params and module_type in ['source_finder']:
            params['corpus_path'] = self.config.corpus_path
        
        try:
            if module_type == "analogy_type_classifier":
                if implementation == "DSPyAnalogyTypeClassifier":
                    return DSPyAnalogyTypeClassifier(**params)
                elif implementation == "SimpleAnalogyTypeClassifier":
                    return SimpleAnalogyTypeClassifier(**params)
            
            elif module_type == "source_finder":
                if implementation == "EmbeddingSourceFinder":
                    return EmbeddingSourceFinder(**params)
                elif implementation == "LLMSourceFinder":
                    return LLMSourceFinder(**params)
            
            elif module_type == "property_matcher":
                if implementation == "DSPyPropertyMatcher":
                    return DSPyPropertyMatcher(**params)
            
            elif module_type == "evaluator":
                if implementation == "LLMEvaluator":
                    return LLMEvaluator(**params)
            
            elif module_type == "improver":
                if implementation == "LLMImprover":
                    return LLMImprover(**params)
            
            elif module_type == "explanation_generator":
                if implementation == "DSPyExplanationGenerator":
                    return DSPyExplanationGenerator(**params)
            
            else:
                print(f"Warning: Unknown module type: {module_type}")
                return None
                
        except Exception as e:
            print(f"Error creating module {module_type}/{implementation}: {e}")
            return None
    
    def _initialize_baselines(self):
        """Initialize baseline methods"""
        try:
            self.baselines['embedding'] = EmbeddingBaseline(
                corpus_path=self.config.corpus_path,
                embedding_mode="name_background"
            )
        except Exception as e:
            print(f"Warning: Could not initialize embedding baseline: {e}")
        
        try:
            self.baselines['embedding_llm'] = EmbeddingLLMBaseline(
                corpus_path=self.config.corpus_path,
                embedding_mode="name_background",
                model_name=self.config.default_model,
                llm_client=self.llm_client
            )
        except Exception as e:
            print(f"Warning: Could not initialize embedding+LLM baseline: {e}")
    
    def run(
        self,
        target_name: str,
        target_description: Optional[str] = None,
        target_properties: Optional[List[str]] = None,
        return_intermediate: bool = False
    ) -> Dict[str, Any]:
        """
        Run the pipeline on a single input
        
        Args:
            target_name: Name of target concept
            target_description: Optional description
            target_properties: Optional list of properties
            return_intermediate: Whether to return intermediate results
            
        Returns:
            Dictionary with results
        """
        # Create initial data based on input format
        data = self._create_input_data(
            target_name=target_name,
            target_description=target_description,
            target_properties=target_properties
        )
        
        # Track intermediate results
        intermediate_results = []
        if return_intermediate:
            intermediate_results.append({
                'stage': 'input',
                'data': self._serialize_data(data)
            })
        
        # Run modules in sequence
        for i, module in enumerate(self.modules):
            try:
                data = module.process(data)
                if return_intermediate:
                    intermediate_results.append({
                        'stage': module.name,
                        'data': self._serialize_data(data)
                    })
            except Exception as e:
                print(f"Error in module {module.name}: {e}")
                data.metadata['error'] = str(e)
                break
        
        # Prepare final results
        results = {
            'target_name': data.target_name,
            'target_description': data.target_description,
            'target_properties': data.target_properties,
            'analogy_type': data.analogy_type,
            'selected_source': data.selected_source,
            'source_candidates': data.source_candidates,
            'property_mappings': data.property_mappings,
            'evaluation_scores': data.evaluation_scores,
            'improved_analogy': data.improved_analogy,
            'explanation': data.explanation,
            'explanation_list': data.explanation_list,
            'metadata': data.metadata
        }
        
        if return_intermediate:
            results['intermediate_results'] = intermediate_results
        
        # Run baselines if enabled
        if self.config.run_baselines:
            results['baseline_results'] = self._run_baselines(
                target_name=target_name,
                target_description=target_description,
                target_properties=target_properties
            )
        
        # Run SCAR evaluation if enabled
        if self.config.run_scar_evaluation and self.scar_evaluator:
            results['scar_evaluation'] = self._run_scar_evaluation(data)
        
        return results
    
    def _create_input_data(
        self,
        target_name: str,
        target_description: Optional[str],
        target_properties: Optional[List[str]]
    ) -> AnalogyData:
        """Create input data based on input format configuration"""
        # Handle different input formats
        if self.config.input_format == InputFormat.TARGET_ONLY:
            return AnalogyData(target_name=target_name)
        
        elif self.config.input_format == InputFormat.TARGET_PROPERTIES:
            return AnalogyData(
                target_name=target_name,
                target_properties=target_properties or []
            )
        
        elif self.config.input_format == InputFormat.TARGET_DESCRIPTION:
            return AnalogyData(
                target_name=target_name,
                target_description=target_description or ""
            )
        
        elif self.config.input_format == InputFormat.TARGET_PROPERTIES_DESCRIPTION:
            return AnalogyData(
                target_name=target_name,
                target_description=target_description or "",
                target_properties=target_properties or []
            )
        
        else:
            # Default: use all available
            return AnalogyData(
                target_name=target_name,
                target_description=target_description or "",
                target_properties=target_properties or []
            )
    
    def _run_baselines(
        self,
        target_name: str,
        target_description: Optional[str],
        target_properties: Optional[List[str]]
    ) -> Dict[str, Any]:
        """Run baseline methods"""
        baseline_results = {}
        
        if 'embedding' in self.baselines:
            try:
                baseline_results['embedding'] = self.baselines['embedding'].generate_analogy(
                    target_name=target_name,
                    target_description=target_description,
                    target_properties=target_properties,
                    top_k=3
                )
            except Exception as e:
                baseline_results['embedding'] = {'error': str(e)}
        
        if 'embedding_llm' in self.baselines:
            try:
                baseline_results['embedding_llm'] = self.baselines['embedding_llm'].generate_analogy(
                    target_name=target_name,
                    target_description=target_description,
                    target_properties=target_properties,
                    top_k_embedding=10,
                    top_k_final=3
                )
            except Exception as e:
                baseline_results['embedding_llm'] = {'error': str(e)}
        
        return baseline_results
    
    def _run_scar_evaluation(self, data: AnalogyData) -> Dict[str, Any]:
        """Run SCAR evaluation"""
        source_name = data.selected_source['name'] if data.selected_source else None
        
        return self.scar_evaluator.evaluate(
            target_name=data.target_name,
            generated_source=source_name,
            generated_mappings=data.property_mappings,
            generated_explanation=data.explanation,
            use_llm_judge=True,
            llm_judge=self.llm_judge,
            threshold=self.config.llm_judge_threshold
        )
    
    def _serialize_data(self, data: AnalogyData) -> Dict[str, Any]:
        """Serialize AnalogyData for storage"""
        return {
            'target_name': data.target_name,
            'target_description': data.target_description,
            'target_properties': data.target_properties,
            'analogy_type': data.analogy_type,
            'selected_source': data.selected_source,
            'source_candidates': data.source_candidates[:5] if data.source_candidates else None,  # Limit for storage
            'property_mappings': data.property_mappings,
            'evaluation_scores': data.evaluation_scores,
            'improved_analogy': data.improved_analogy,
            'explanation': data.explanation,
            'explanation_list': data.explanation_list
        }
    
    def run_batch(
        self,
        inputs: List[Dict[str, Any]],
        save_results: bool = True
    ) -> List[Dict[str, Any]]:
        """
        Run pipeline on multiple inputs
        
        Args:
            inputs: List of input dictionaries with target_name, target_description, target_properties
            save_results: Whether to save results to file
            
        Returns:
            List of result dictionaries
        """
        results = []
        
        for i, input_data in enumerate(inputs):
            print(f"Processing {i+1}/{len(inputs)}: {input_data.get('target_name', 'Unknown')}")
            
            result = self.run(
                target_name=input_data.get('target_name', ''),
                target_description=input_data.get('target_description'),
                target_properties=input_data.get('target_properties'),
                return_intermediate=False
            )
            
            results.append(result)
        
        # Save results if requested
        if save_results and self.config.save_results:
            self._save_results(results)
        
        return results
    
    def _save_results(self, results: List[Dict[str, Any]]):
        """Save results to file"""
        # Create results directory
        results_dir = Path(self.config.results_dir)
        results_dir.mkdir(parents=True, exist_ok=True)
        
        # Generate filename
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        exp_name = self.config.experiment_name or "experiment"
        filename = f"{exp_name}_{timestamp}.json"
        filepath = results_dir / filename
        
        # Save results
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump({
                'config': self.config.to_dict(),
                'results': results,
                'timestamp': timestamp
            }, f, indent=2, ensure_ascii=False)
        
        print(f"Results saved to: {filepath}")
        
        # Also save as CSV for easy analysis
        csv_filepath = results_dir / f"{exp_name}_{timestamp}.csv"
        self._save_results_csv(results, csv_filepath)
    
    def _save_results_csv(self, results: List[Dict[str, Any]], filepath: Path):
        """Save results as CSV"""
        rows = []
        for result in results:
            row = {
                'target_name': result.get('target_name', ''),
                'target_description': result.get('target_description', ''),
                'analogy_type': result.get('analogy_type', ''),
                'selected_source_name': result.get('selected_source', {}).get('name', '') if result.get('selected_source') else '',
                'selected_source_domain': result.get('selected_source', {}).get('domain', '') if result.get('selected_source') else '',
                'num_property_mappings': len(result.get('property_mappings', [])),
                'explanation': str(result.get('explanation', '')),
            }
            
            # Add evaluation scores
            eval_scores = result.get('evaluation_scores', {})
            if eval_scores:
                row['relevance_score'] = eval_scores.get('relevance_score', '')
                row['clarity_score'] = eval_scores.get('clarity_score', '')
                row['accuracy_score'] = eval_scores.get('accuracy_score', '')
                row['overall_score'] = eval_scores.get('overall_score', '')
            
            # Add SCAR evaluation
            scar_eval = result.get('scar_evaluation', {})
            if scar_eval:
                row['scar_source_match'] = scar_eval.get('source_match', False)
                row['scar_mapping_match'] = scar_eval.get('mapping_match', False)
                row['scar_overall_match'] = scar_eval.get('overall_match', False)
                row['llm_judge_score'] = scar_eval.get('llm_judge_score', '')
                row['llm_judge_passed'] = scar_eval.get('llm_judge_passed', '')
            
            rows.append(row)
        
        df = pd.DataFrame(rows)
        df.to_csv(filepath, index=False)
        print(f"CSV results saved to: {filepath}")

