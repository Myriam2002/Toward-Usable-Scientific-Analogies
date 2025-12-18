"""
LLM Judge
Evaluates analogy quality using LLM when SCAR match is not found
"""

from typing import Optional, List, Dict, Any
import sys
from pathlib import Path

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


class LLMJudge:
    """
    LLM-based judge for evaluating analogy quality
    """
    
    def __init__(
        self,
        model_name: str = "gpt-4o-mini",
        llm_client: Optional[LLMClient] = None
    ):
        """
        Initialize LLM judge
        
        Args:
            model_name: Name of the LLM model to use
            llm_client: Optional LLM client
        """
        if not DSPY_AVAILABLE:
            raise ImportError("DSPy not available. Install with: pip install dspy-ai")
        
        self.model_name = model_name
        self.llm_client = llm_client or LLMClient()
        self.adapter = DSPyAdapter(self.llm_client, model_name)
        self.lm = self.adapter.get_dspy_lm()
        dspy.configure(lm=self.lm)
        
        # Define DSPy signature
        class AnalogyJudgment(dspy.Signature):
            """Judge the quality of a scientific analogy on a scale of 0-1."""
            target_concept: str = dspy.InputField(desc="The target concept")
            generated_source: str = dspy.InputField(desc="The generated source concept")
            golden_source: str = dspy.InputField(desc="The golden standard source concept")
            generated_mappings: str = dspy.InputField(desc="Generated property mappings")
            golden_mappings: str = dspy.InputField(desc="Golden standard property mappings")
            quality_score: float = dspy.OutputField(desc="Quality score from 0.0 to 1.0")
            reasoning: str = dspy.OutputField(desc="Explanation of the quality score")
        
        self.predictor = dspy.ChainOfThought(AnalogyJudgment)
    
    def evaluate(
        self,
        target_name: str,
        generated_source: Optional[str] = None,
        golden_source: Optional[str] = None,
        generated_mappings: Optional[List[List[str]]] = None,
        golden_mappings: Optional[List[List[str]]] = None
    ) -> float:
        """
        Evaluate analogy quality
        
        Args:
            target_name: Name of target concept
            generated_source: Generated source concept
            golden_source: Golden standard source (for comparison)
            generated_mappings: Generated property mappings
            golden_mappings: Golden standard mappings (for comparison)
            
        Returns:
            Quality score (0.0 to 1.0)
        """
        if not generated_source:
            return 0.0
        
        # Format mappings
        gen_mappings_str = ", ".join([f"{m[0]}→{m[1]}" for m in (generated_mappings or [])])
        gold_mappings_str = ", ".join([f"{m[0]}→{m[1]}" for m in (golden_mappings or [])]) if golden_mappings else "N/A"
        
        result = self.predictor(
            target_concept=target_name,
            generated_source=generated_source or "",
            golden_source=golden_source or "",
            generated_mappings=gen_mappings_str,
            golden_mappings=gold_mappings_str
        )
        
        # Parse score
        score = result.quality_score
        if isinstance(score, str):
            try:
                score = float(score)
            except:
                # Try to extract number
                import re
                match = re.search(r'[\d.]+', score)
                score = float(match.group()) if match else 0.0
        
        # Clamp to [0, 1]
        score = max(0.0, min(1.0, float(score)))
        
        return score

