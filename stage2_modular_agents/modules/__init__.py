"""
Modular Analogy Generation Pipeline - Modules Package
"""

from .base_module import BaseModule, AnalogyData
from .analogy_type_classifier import AnalogyTypeClassifier
from .source_finder import SourceFinder, EmbeddingSourceFinder, LLMSourceFinder
from .property_matcher import PropertyMatcher
from .evaluator import Evaluator
from .improver import Improver
from .explanation_generator import ExplanationGenerator

__all__ = [
    'BaseModule',
    'AnalogyData',
    'AnalogyTypeClassifier',
    'SourceFinder',
    'EmbeddingSourceFinder',
    'LLMSourceFinder',
    'PropertyMatcher',
    'Evaluator',
    'Improver',
    'ExplanationGenerator',
]

