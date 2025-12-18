"""
Base Module Interface for Analogy Generation Pipeline
"""

from abc import ABC, abstractmethod
from typing import Optional, List, Dict, Any
from dataclasses import dataclass, field


@dataclass
class AnalogyData:
    """
    Standardized data structure for passing information between modules
    """
    # Input fields
    target_name: str
    target_description: Optional[str] = None
    target_properties: Optional[List[str]] = None
    
    # Analogy type classification
    analogy_type: Optional[str] = None
    
    # Source finding results
    source_candidates: Optional[List[Dict[str, Any]]] = None
    selected_source: Optional[Dict[str, Any]] = None
    
    # Property matching results
    property_mappings: Optional[List[List[str]]] = None  # [[target_prop, source_prop], ...]
    
    # Evaluation results
    evaluation_scores: Optional[Dict[str, float]] = None
    
    # Improvement results
    improved_analogy: Optional[Dict[str, Any]] = None
    
    # Explanation
    explanation: Optional[str] = None
    explanation_list: Optional[List[str]] = None
    
    # Metadata
    metadata: Dict[str, Any] = field(default_factory=dict)


class BaseModule(ABC):
    """
    Base class for all pipeline modules
    All modules must implement the process() method
    """
    
    def __init__(self, name: str = None):
        """
        Initialize the module
        
        Args:
            name: Optional name for the module instance
        """
        self.name = name or self.__class__.__name__
    
    @abstractmethod
    def process(self, data: AnalogyData) -> AnalogyData:
        """
        Process the analogy data and return updated data
        
        Args:
            data: Input AnalogyData object
            
        Returns:
            Updated AnalogyData object with module's output
        """
        pass
    
    def __call__(self, data: AnalogyData) -> AnalogyData:
        """Allow module to be called directly"""
        return self.process(data)
    
    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(name='{self.name}')"

