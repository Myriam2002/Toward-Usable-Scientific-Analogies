"""
Explanation Generation Module
Generates explanations for analogies
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


class ExplanationGenerator(BaseModule):
    """
    Base class for explanation generation modules
    """
    
    def process(self, data: AnalogyData) -> AnalogyData:
        """
        Generate explanation for the analogy
        
        Args:
            data: AnalogyData with complete analogy information
            
        Returns:
            AnalogyData with explanation set
        """
        # Use improved analogy if available, otherwise use original
        source_name = data.selected_source['name'] if data.selected_source else None
        source_desc = data.selected_source.get('description', '') if data.selected_source else None
        mappings = data.property_mappings
        
        if data.improved_analogy:
            source_name = data.improved_analogy.get('improved_source', source_name)
            mappings = data.improved_analogy.get('improved_mappings', mappings)
        
        explanation = self._generate_explanation(
            target_name=data.target_name,
            target_description=data.target_description,
            source_name=source_name,
            source_description=source_desc,
            property_mappings=mappings,
            analogy_type=data.analogy_type
        )
        
        data.explanation = explanation
        # Also create list format if explanation is string
        if isinstance(explanation, str):
            # Split by sentences or newlines
            data.explanation_list = [s.strip() for s in explanation.replace('\n', '.').split('.') if s.strip()]
        elif isinstance(explanation, list):
            data.explanation_list = explanation
        
        return data
    
    def _generate_explanation(
        self,
        target_name: str,
        target_description: Optional[str],
        source_name: Optional[str],
        source_description: Optional[str],
        property_mappings: Optional[List[List[str]]],
        analogy_type: Optional[str]
    ) -> str:
        """
        Generate explanation
        
        Args:
            target_name: Name of target concept
            target_description: Optional target description
            source_name: Name of source concept
            source_description: Optional source description
            property_mappings: List of property mappings
            analogy_type: Analogy type
            
        Returns:
            Explanation string or list of explanation strings
        """
        raise NotImplementedError("Subclasses must implement _generate_explanation method")


class DSPyExplanationGenerator(ExplanationGenerator):
    """
    DSPy-based explanation generator
    """
    
    def __init__(
        self,
        model_name: str = "gpt-4o-mini",
        llm_client: Optional[LLMClient] = None,
        use_description: bool = True,
        use_paired_properties: bool = True,
        name: str = None
    ):
        """
        Initialize DSPy-based explanation generator
        
        Args:
            model_name: Name of the LLM model to use
            llm_client: Optional LLM client
            use_description: Whether to use descriptions
            use_paired_properties: Whether to use paired property mappings
            name: Optional module name
        """
        super().__init__(name=name)
        if not DSPY_AVAILABLE:
            raise ImportError("DSPy not available. Install with: pip install dspy-ai")
        
        self.model_name = model_name
        self.use_description = use_description
        self.use_paired_properties = use_paired_properties
        self.llm_client = llm_client or LLMClient()
        self.adapter = DSPyAdapter(self.llm_client, model_name)
        self.lm = self.adapter.get_dspy_lm()
        dspy.configure(lm=self.lm)
        
        # Define DSPy signature based on configuration
        if use_paired_properties and use_description:
            class ExplanationGeneration(dspy.Signature):
                """Generate explanation for a scientific analogy."""
                unfamiliar_concept: str = dspy.InputField(desc="The unfamiliar (target) concept")
                familiar_concept: str = dspy.InputField(desc="The familiar (source) concept")
                description_of_unfamiliar_concept: str = dspy.InputField(desc="Description of the unfamiliar concept")
                description_of_familiar_concept: str = dspy.InputField(desc="Description of the familiar concept")
                paired_properties: List[List[str]] = dspy.InputField(desc="List of paired properties [[target_prop, source_prop], ...]")
                Explanation: List[str] = dspy.OutputField(desc="List of explanation sentences, one for each property pair")
            
            self.predictor = dspy.ChainOfThought(ExplanationGeneration)
        elif use_paired_properties:
            class ExplanationGenerationNoDesc(dspy.Signature):
                """Generate explanation for a scientific analogy."""
                unfamiliar_concept: str = dspy.InputField(desc="The unfamiliar (target) concept")
                familiar_concept: str = dspy.InputField(desc="The familiar (source) concept")
                paired_properties: List[List[str]] = dspy.InputField(desc="List of paired properties [[target_prop, source_prop], ...]")
                Explanation: List[str] = dspy.OutputField(desc="List of explanation sentences, one for each property pair")
            
            self.predictor = dspy.ChainOfThought(ExplanationGenerationNoDesc)
        elif use_description:
            class ExplanationGenerationUnpaired(dspy.Signature):
                """Generate explanation for a scientific analogy."""
                unfamiliar_concept: str = dspy.InputField(desc="The unfamiliar (target) concept")
                familiar_concept: str = dspy.InputField(desc="The familiar (source) concept")
                description_of_unfamiliar_concept: str = dspy.InputField(desc="Description of the unfamiliar concept")
                description_of_familiar_concept: str = dspy.InputField(desc="Description of the familiar concept")
                properties_of_unfamiliar_concept: List[str] = dspy.InputField(desc="Properties of the unfamiliar concept")
                properties_of_familiar_concept: List[str] = dspy.InputField(desc="Properties of the familiar concept")
                Explanation: List[str] = dspy.OutputField(desc="List of explanation sentences")
            
            self.predictor = dspy.ChainOfThought(ExplanationGenerationUnpaired)
        else:
            class ExplanationGenerationBasic(dspy.Signature):
                """Generate explanation for a scientific analogy."""
                unfamiliar_concept: str = dspy.InputField(desc="The unfamiliar (target) concept")
                familiar_concept: str = dspy.InputField(desc="The familiar (source) concept")
                properties_of_unfamiliar_concept: List[str] = dspy.InputField(desc="Properties of the unfamiliar concept")
                properties_of_familiar_concept: List[str] = dspy.InputField(desc="Properties of the familiar concept")
                Explanation: List[str] = dspy.OutputField(desc="List of explanation sentences")
            
            self.predictor = dspy.ChainOfThought(ExplanationGenerationBasic)
    
    def _generate_explanation(
        self,
        target_name: str,
        target_description: Optional[str],
        source_name: Optional[str],
        source_description: Optional[str],
        property_mappings: Optional[List[List[str]]],
        analogy_type: Optional[str]
    ) -> str:
        """Generate explanation using DSPy"""
        if not source_name:
            return "No source concept available for explanation."
        
        if self.use_paired_properties and property_mappings:
            if self.use_description:
                result = self.predictor(
                    unfamiliar_concept=target_name,
                    familiar_concept=source_name,
                    description_of_unfamiliar_concept=target_description or "",
                    description_of_familiar_concept=source_description or "",
                    paired_properties=property_mappings
                )
            else:
                result = self.predictor(
                    unfamiliar_concept=target_name,
                    familiar_concept=source_name,
                    paired_properties=property_mappings
                )
        else:
            # Extract properties from mappings
            target_props = [m[0] for m in (property_mappings or [])]
            source_props = [m[1] for m in (property_mappings or [])]
            
            if self.use_description:
                result = self.predictor(
                    unfamiliar_concept=target_name,
                    familiar_concept=source_name,
                    description_of_unfamiliar_concept=target_description or "",
                    description_of_familiar_concept=source_description or "",
                    properties_of_unfamiliar_concept=target_props,
                    properties_of_familiar_concept=source_props
                )
            else:
                result = self.predictor(
                    unfamiliar_concept=target_name,
                    familiar_concept=source_name,
                    properties_of_unfamiliar_concept=target_props,
                    properties_of_familiar_concept=source_props
                )
        
        # Handle different return types
        if isinstance(result.Explanation, list):
            return result.Explanation
        elif isinstance(result.Explanation, str):
            # Try to parse as list
            import ast
            try:
                parsed = ast.literal_eval(result.Explanation)
                if isinstance(parsed, list):
                    return parsed
            except:
                # Split by sentences
                return [s.strip() for s in result.Explanation.replace('\n', '.').split('.') if s.strip()]
        else:
            return ["Explanation generation failed"]

