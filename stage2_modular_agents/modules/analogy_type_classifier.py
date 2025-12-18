"""
Analogy Type Classification Module
Determines the best type of analogy for a given target concept
"""

from typing import Optional, List
import sys
from pathlib import Path
from .base_module import BaseModule, AnalogyData

# Add mapping_generation directory to path to import easy_llm_importer
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


RELATION_TYPES = [
    "SIMILARITY",
    "CONTRAST",
    "ATTRIBUTE",
    "CLASS-INCLUSION (TYPE-OF)",
    "PART–WHOLE (SUBSYSTEM–SYSTEM)",
    "CAUSE–EFFECT (MECHANISM)",
    "TEMPORAL (SEQUENCE / DEVELOPMENT)",
    "SPATIAL (STRUCTURE / ARRANGEMENT)",
    "FUNCTION (ROLE / PURPOSE)",
    "AGENCY / ACTION"
]


class AnalogyTypeClassifier(BaseModule):
    """
    Base class for analogy type classification
    """
    
    def process(self, data: AnalogyData) -> AnalogyData:
        """
        Classify the analogy type for the given target
        
        Args:
            data: AnalogyData with target information
            
        Returns:
            AnalogyData with analogy_type set
        """
        analogy_type = self._classify(
            target_name=data.target_name,
            target_description=data.target_description,
            target_properties=data.target_properties
        )
        data.analogy_type = analogy_type
        return data
    
    def _classify(
        self,
        target_name: str,
        target_description: Optional[str] = None,
        target_properties: Optional[List[str]] = None
    ) -> str:
        """
        Classify the analogy type
        
        Args:
            target_name: Name of target concept
            target_description: Optional description
            target_properties: Optional list of properties
            
        Returns:
            Analogy type string
        """
        raise NotImplementedError("Subclasses must implement _classify method")


class DSPyAnalogyTypeClassifier(AnalogyTypeClassifier):
    """
    DSPy-based analogy type classifier using LLM
    """
    
    def __init__(
        self,
        model_name: str = "gpt-4o-mini",
        llm_client: Optional[LLMClient] = None,
        name: str = None
    ):
        """
        Initialize DSPy-based classifier
        
        Args:
            model_name: Name of the LLM model to use
            llm_client: Optional LLM client (creates new one if not provided)
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
        class AnalogyTypeClassification(dspy.Signature):
            """Classify the analogy type for a given target concept."""
            target_concept: str = dspy.InputField(desc="The target concept to classify")
            target_description: str = dspy.InputField(desc="Description of the target concept")
            target_properties: str = dspy.InputField(desc="Key properties of the target concept")
            analogy_type: str = dspy.OutputField(
                desc=f"Single analogy type from: {', '.join(RELATION_TYPES)}"
            )
        
        self.predictor = dspy.ChainOfThought(AnalogyTypeClassification)
    
    def _classify(
        self,
        target_name: str,
        target_description: Optional[str] = None,
        target_properties: Optional[List[str]] = None
    ) -> str:
        """Classify using DSPy"""
        desc = target_description or ""
        props = ", ".join(target_properties) if target_properties else ""
        
        result = self.predictor(
            target_concept=target_name,
            target_description=desc,
            target_properties=props
        )
        
        # Validate and return
        analogy_type = result.analogy_type.strip().upper()
        
        # Check if it matches any relation type (fuzzy match)
        for rel_type in RELATION_TYPES:
            if rel_type.upper() in analogy_type or analogy_type in rel_type.upper():
                return rel_type
        
        # If no match, return the first part (before any colon or dash)
        return analogy_type.split(":")[0].split("–")[0].split("-")[0].strip()


class SimpleAnalogyTypeClassifier(AnalogyTypeClassifier):
    """
    Simple rule-based classifier (fallback)
    """
    
    def _classify(
        self,
        target_name: str,
        target_description: Optional[str] = None,
        target_properties: Optional[List[str]] = None
    ) -> str:
        """Simple rule-based classification"""
        # Default to FUNCTION as it's most common for scientific analogies
        return "FUNCTION (ROLE / PURPOSE)"

