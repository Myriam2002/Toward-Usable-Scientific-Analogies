"""
Property Matching Module
Maps properties between target and source concepts
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


class PropertyMatcher(BaseModule):
    """
    Base class for property matching modules
    """
    
    def process(self, data: AnalogyData) -> AnalogyData:
        """
        Match properties between target and source
        
        Args:
            data: AnalogyData with target, source, and properties
            
        Returns:
            AnalogyData with property_mappings set
        """
        if not data.selected_source:
            raise ValueError("No source selected. Run source finder first.")
        
        if not data.target_properties:
            # If no properties provided, extract them first
            data.target_properties = self._extract_properties(
                target_name=data.target_name,
                target_description=data.target_description
            )
        
        # Match properties
        mappings = self._match_properties(
            target_name=data.target_name,
            target_description=data.target_description,
            target_properties=data.target_properties,
            source_name=data.selected_source['name'],
            source_description=data.selected_source.get('description', '')
        )
        
        data.property_mappings = mappings
        return data
    
    def _extract_properties(
        self,
        target_name: str,
        target_description: Optional[str] = None
    ) -> List[str]:
        """
        Extract properties from target concept
        
        Args:
            target_name: Name of target concept
            target_description: Optional description
            
        Returns:
            List of property strings
        """
        raise NotImplementedError("Subclasses must implement _extract_properties method")
    
    def _match_properties(
        self,
        target_name: str,
        target_description: Optional[str],
        target_properties: List[str],
        source_name: str,
        source_description: str
    ) -> List[List[str]]:
        """
        Match properties between target and source
        
        Args:
            target_name: Name of target concept
            target_description: Optional target description
            target_properties: List of target properties
            source_name: Name of source concept
            source_description: Source description
            
        Returns:
            List of property mappings [[target_prop, source_prop], ...]
        """
        raise NotImplementedError("Subclasses must implement _match_properties method")


class DSPyPropertyMatcher(PropertyMatcher):
    """
    DSPy-based property matcher using LLM
    """
    
    def __init__(
        self,
        model_name: str = "gpt-4o-mini",
        llm_client: Optional[LLMClient] = None,
        use_description: bool = True,
        name: str = None
    ):
        """
        Initialize DSPy-based property matcher
        
        Args:
            model_name: Name of the LLM model to use
            llm_client: Optional LLM client
            use_description: Whether to use descriptions in matching
            name: Optional module name
        """
        super().__init__(name=name)
        if not DSPY_AVAILABLE:
            raise ImportError("DSPy not available. Install with: pip install dspy-ai")
        
        self.model_name = model_name
        self.llm_client = llm_client or LLMClient()
        self.use_description = use_description
        self.adapter = DSPyAdapter(self.llm_client, model_name)
        self.lm = self.adapter.get_dspy_lm()
        dspy.configure(lm=self.lm)
        
        # Define DSPy signatures
        if use_description:
            class PropertyMatching(dspy.Signature):
                """Match properties between target and source concepts for analogy creation."""
                unfamiliar_concept: str = dspy.InputField(desc="The unfamiliar (target) concept")
                description_of_unfamiliar_concept: str = dspy.InputField(desc="Description of the unfamiliar concept")
                properties_of_unfamiliar_concept: List[str] = dspy.InputField(desc="List of key properties of the unfamiliar concept")
                familiar_concept: str = dspy.InputField(desc="The familiar (source) concept")
                description_of_familiar_concept: str = dspy.InputField(desc="Description of the familiar concept")
                mapped_source_properties: Dict[str, str] = dspy.OutputField(desc="Dictionary mapping each unfamiliar property to corresponding familiar property")
            
            self.predictor = dspy.ChainOfThought(PropertyMatching)
        else:
            class PropertyMatchingNoDesc(dspy.Signature):
                """Match properties between target and source concepts for analogy creation."""
                unfamiliar_concept: str = dspy.InputField(desc="The unfamiliar (target) concept")
                properties_of_unfamiliar_concept: List[str] = dspy.InputField(desc="List of key properties of the unfamiliar concept")
                familiar_concept: str = dspy.InputField(desc="The familiar (source) concept")
                mapped_source_properties: Dict[str, str] = dspy.OutputField(desc="Dictionary mapping each unfamiliar property to corresponding familiar property")
            
            self.predictor = dspy.ChainOfThought(PropertyMatchingNoDesc)
        
        # Property extraction signature
        class PropertyExtraction(dspy.Signature):
            """Extract key properties from a concept."""
            concept_name: str = dspy.InputField(desc="The concept to extract properties from")
            concept_description: str = dspy.InputField(desc="Description of the concept")
            properties: List[str] = dspy.OutputField(desc="List of 1-2 word key properties that characterize the concept")
        
        self.extractor = dspy.ChainOfThought(PropertyExtraction)
    
    def _extract_properties(
        self,
        target_name: str,
        target_description: Optional[str] = None
    ) -> List[str]:
        """Extract properties using DSPy"""
        result = self.extractor(
            concept_name=target_name,
            concept_description=target_description or ""
        )
        
        # Handle different return types
        if isinstance(result.properties, list):
            return result.properties
        elif isinstance(result.properties, str):
            # Try to parse as list
            import ast
            try:
                return ast.literal_eval(result.properties)
            except:
                # Split by comma or newline
                return [p.strip() for p in result.properties.replace('\n', ',').split(',') if p.strip()]
        else:
            return []
    
    def _match_properties(
        self,
        target_name: str,
        target_description: Optional[str],
        target_properties: List[str],
        source_name: str,
        source_description: str
    ) -> List[List[str]]:
        """Match properties using DSPy"""
        if self.use_description:
            result = self.predictor(
                unfamiliar_concept=target_name,
                description_of_unfamiliar_concept=target_description or "",
                properties_of_unfamiliar_concept=target_properties,
                familiar_concept=source_name,
                description_of_familiar_concept=source_description
            )
        else:
            result = self.predictor(
                unfamiliar_concept=target_name,
                properties_of_unfamiliar_concept=target_properties,
                familiar_concept=source_name
            )
        
        # Convert dict to list of pairs
        mappings = []
        if isinstance(result.mapped_source_properties, dict):
            for target_prop, source_prop in result.mapped_source_properties.items():
                mappings.append([target_prop, source_prop])
        elif isinstance(result.mapped_source_properties, str):
            # Try to parse as dict
            import ast
            try:
                mapping_dict = ast.literal_eval(result.mapped_source_properties)
                if isinstance(mapping_dict, dict):
                    for target_prop, source_prop in mapping_dict.items():
                        mappings.append([target_prop, source_prop])
            except:
                pass
        
        return mappings

