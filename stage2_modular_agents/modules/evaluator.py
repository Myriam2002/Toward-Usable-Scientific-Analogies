"""
Evaluation and Reranking Module
Evaluates and reranks candidate analogies
"""

from typing import Optional, List, Dict, Any
import sys
from pathlib import Path
from .base_module import BaseModule, AnalogyData

# Import LLM client
current_dir = Path(__file__).parent.parent.parent
mapping_dir = current_dir / "stage1_analysis" / "mapping_generation"
if str(mapping_dir) not in sys.path:
    sys.path.insert(0, str(mapping_dir))

try:
    from easy_llm_importer import LLMClient, DSPyAdapter
    import dspy
    DSPY_AVAILABLE = True
except ImportError:
    DSPY_AVAILABLE = False


class Evaluator(BaseModule):
    """
    Base class for evaluation and reranking modules
    """
    
    def process(self, data: AnalogyData) -> AnalogyData:
        """
        Evaluate and rerank analogies
        
        Args:
            data: AnalogyData with analogy information
            
        Returns:
            AnalogyData with evaluation_scores set
        """
        scores = self._evaluate(
            target_name=data.target_name,
            target_description=data.target_description,
            source_name=data.selected_source['name'] if data.selected_source else None,
            source_description=data.selected_source.get('description', '') if data.selected_source else None,
            property_mappings=data.property_mappings,
            analogy_type=data.analogy_type
        )
        
        data.evaluation_scores = scores
        
        # If multiple source candidates, rerank them
        if data.source_candidates and len(data.source_candidates) > 1:
            data.source_candidates = self._rerank(data.source_candidates, scores)
            data.selected_source = data.source_candidates[0] if data.source_candidates else None
        
        return data
    
    def _evaluate(
        self,
        target_name: str,
        target_description: Optional[str],
        source_name: Optional[str],
        source_description: Optional[str],
        property_mappings: Optional[List[List[str]]],
        analogy_type: Optional[str]
    ) -> Dict[str, float]:
        """
        Evaluate an analogy
        
        Args:
            target_name: Name of target concept
            target_description: Optional target description
            source_name: Name of source concept
            source_description: Optional source description
            property_mappings: List of property mappings
            analogy_type: Optional analogy type
            
        Returns:
            Dictionary of evaluation scores
        """
        raise NotImplementedError("Subclasses must implement _evaluate method")
    
    def _rerank(
        self,
        candidates: List[Dict[str, Any]],
        scores: Dict[str, float]
    ) -> List[Dict[str, Any]]:
        """
        Rerank candidates based on scores
        
        Args:
            candidates: List of candidate dictionaries
            scores: Evaluation scores
            
        Returns:
            Reranked list of candidates
        """
        # Default: sort by score if available
        if 'overall_score' in scores:
            # Sort candidates by their scores (if scores are per-candidate)
            # For now, just return as-is
            return candidates
        return candidates


class LLMEvaluator(Evaluator):
    """
    LLM-based evaluator using DSPy
    """
    
    def __init__(
        self,
        model_name: str = "gpt-4o-mini",
        llm_client: Optional[LLMClient] = None,
        name: str = None
    ):
        """
        Initialize LLM-based evaluator
        
        Args:
            model_name: Name of the LLM model to use
            llm_client: Optional LLM client
            name: Optional module name
        """
        super().__init__(name=name)
        if not DSPY_AVAILABLE:
            raise ImportError("DSPy not available. Install with: pip install dspy-ai")
        
        self.model_name = model_name
        self.llm_client = llm_client or LLMClient()
        self.adapter = DSPyAdapter(self.llm_client, model_name)
        self.lm = self.adapter.get_dspy_lm()
        dspy.configure(lm=self.lm)
        
        # Define DSPy signature
        class AnalogyEvaluation(dspy.Signature):
            """Evaluate the quality of a scientific analogy."""
            target_concept: str = dspy.InputField(desc="The target concept")
            source_concept: str = dspy.InputField(desc="The source concept")
            property_mappings: str = dspy.InputField(desc="Property mappings between target and source")
            analogy_type: str = dspy.InputField(desc="Type of analogy")
            relevance_score: float = dspy.OutputField(desc="Relevance score (0-1): How relevant is the source to the target?")
            clarity_score: float = dspy.OutputField(desc="Clarity score (0-1): How clear is the analogy?")
            accuracy_score: float = dspy.OutputField(desc="Accuracy score (0-1): How accurate are the property mappings?")
            overall_score: float = dspy.OutputField(desc="Overall quality score (0-1): Overall analogy quality")
        
        self.predictor = dspy.ChainOfThought(AnalogyEvaluation)
    
    def _evaluate(
        self,
        target_name: str,
        target_description: Optional[str],
        source_name: Optional[str],
        source_description: Optional[str],
        property_mappings: Optional[List[List[str]]],
        analogy_type: Optional[str]
    ) -> Dict[str, float]:
        """Evaluate using LLM"""
        if not source_name:
            return {
                'relevance_score': 0.0,
                'clarity_score': 0.0,
                'accuracy_score': 0.0,
                'overall_score': 0.0
            }
        
        # Format property mappings
        mappings_str = ", ".join([f"{m[0]}→{m[1]}" for m in (property_mappings or [])])
        analogy_type_str = analogy_type or "UNKNOWN"
        
        result = self.predictor(
            target_concept=target_name,
            source_concept=source_name,
            property_mappings=mappings_str,
            analogy_type=analogy_type_str
        )
        
        # Parse scores (handle string or float)
        def parse_score(score):
            if isinstance(score, (int, float)):
                return float(score)
            elif isinstance(score, str):
                try:
                    return float(score)
                except:
                    # Try to extract number from string
                    import re
                    match = re.search(r'[\d.]+', score)
                    return float(match.group()) if match else 0.0
            return 0.0
        
        scores = {
            'relevance_score': parse_score(result.relevance_score),
            'clarity_score': parse_score(result.clarity_score),
            'accuracy_score': parse_score(result.accuracy_score),
            'overall_score': parse_score(result.overall_score)
        }
        
        return scores

