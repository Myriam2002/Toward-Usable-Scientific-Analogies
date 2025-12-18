"""
LLM Improver Module
Refines and improves analogies based on evaluation feedback
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


class Improver(BaseModule):
    """
    Base class for analogy improvement modules
    """
    
    def process(self, data: AnalogyData) -> AnalogyData:
        """
        Improve the analogy based on evaluation feedback
        
        Args:
            data: AnalogyData with analogy and evaluation scores
            
        Returns:
            AnalogyData with improved_analogy set
        """
        improved = self._improve(
            target_name=data.target_name,
            target_description=data.target_description,
            source_name=data.selected_source['name'] if data.selected_source else None,
            source_description=data.selected_source.get('description', '') if data.selected_source else None,
            property_mappings=data.property_mappings,
            evaluation_scores=data.evaluation_scores,
            analogy_type=data.analogy_type
        )
        
        data.improved_analogy = improved
        return data
    
    def _improve(
        self,
        target_name: str,
        target_description: Optional[str],
        source_name: Optional[str],
        source_description: Optional[str],
        property_mappings: Optional[List[List[str]]],
        evaluation_scores: Optional[Dict[str, float]],
        analogy_type: Optional[str]
    ) -> Dict[str, Any]:
        """
        Improve the analogy
        
        Args:
            target_name: Name of target concept
            target_description: Optional target description
            source_name: Name of source concept
            source_description: Optional source description
            property_mappings: Current property mappings
            evaluation_scores: Evaluation scores
            analogy_type: Analogy type
            
        Returns:
            Dictionary with improved analogy components
        """
        raise NotImplementedError("Subclasses must implement _improve method")


class LLMImprover(Improver):
    """
    LLM-based improver using DSPy
    """
    
    def __init__(
        self,
        model_name: str = "gpt-4o-mini",
        llm_client: Optional[LLMClient] = None,
        name: str = None
    ):
        """
        Initialize LLM-based improver
        
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
        class AnalogyImprovement(dspy.Signature):
            """Improve a scientific analogy based on evaluation feedback."""
            target_concept: str = dspy.InputField(desc="The target concept")
            source_concept: str = dspy.InputField(desc="The current source concept")
            current_mappings: str = dspy.InputField(desc="Current property mappings")
            evaluation_feedback: str = dspy.InputField(desc="Evaluation scores and feedback")
            improved_source: str = dspy.OutputField(desc="Improved or alternative source concept (or same if good)")
            improved_mappings: str = dspy.OutputField(desc="Improved property mappings as list of pairs")
            improvement_rationale: str = dspy.OutputField(desc="Explanation of improvements made")
        
        self.predictor = dspy.ChainOfThought(AnalogyImprovement)
    
    def _improve(
        self,
        target_name: str,
        target_description: Optional[str],
        source_name: Optional[str],
        source_description: Optional[str],
        property_mappings: Optional[List[List[str]]],
        evaluation_scores: Optional[Dict[str, float]],
        analogy_type: Optional[str]
    ) -> Dict[str, Any]:
        """Improve using LLM"""
        if not source_name:
            return {
                'improved_source': None,
                'improved_mappings': property_mappings or [],
                'improvement_rationale': "No source to improve"
            }
        
        # Format inputs
        mappings_str = ", ".join([f"{m[0]}→{m[1]}" for m in (property_mappings or [])])
        feedback_str = f"Scores: {evaluation_scores}" if evaluation_scores else "No evaluation scores available"
        
        result = self.predictor(
            target_concept=target_name,
            source_concept=source_name,
            current_mappings=mappings_str,
            evaluation_feedback=feedback_str
        )
        
        # Parse improved mappings
        improved_mappings = property_mappings or []
        if result.improved_mappings:
            import ast
            try:
                # Try to parse as list
                parsed = ast.literal_eval(result.improved_mappings)
                if isinstance(parsed, list):
                    improved_mappings = parsed
            except:
                # Try to parse from string format
                pass
        
        return {
            'improved_source': result.improved_source,
            'improved_mappings': improved_mappings,
            'improvement_rationale': result.improvement_rationale
        }

