"""
Source Finding Module
Finds analogous source concepts for a given target
"""

from typing import Optional, List, Dict, Any
import sys
from pathlib import Path
from .base_module import BaseModule, AnalogyData

# Import RAG source finder from stage1
current_dir = Path(__file__).parent.parent.parent
source_finding_dir = current_dir / "stage1_analysis" / "source_finding"
if str(source_finding_dir) not in sys.path:
    sys.path.insert(0, str(source_finding_dir))

try:
    from rag_source_finder import RAGSourceFinder, SourceCandidate
    RAG_AVAILABLE = True
except ImportError:
    RAG_AVAILABLE = False
    SourceCandidate = None

# Import LLM client
mapping_dir = current_dir / "stage1_analysis" / "mapping_generation"
if str(mapping_dir) not in sys.path:
    sys.path.insert(0, str(mapping_dir))

try:
    from easy_llm_importer import LLMClient, DSPyAdapter
    import dspy
    DSPY_AVAILABLE = True
except ImportError:
    DSPY_AVAILABLE = False


class SourceFinder(BaseModule):
    """
    Base class for source finding modules
    """
    
    def process(self, data: AnalogyData) -> AnalogyData:
        """
        Find source candidates for the target
        
        Args:
            data: AnalogyData with target information
            
        Returns:
            AnalogyData with source_candidates and selected_source set
        """
        source_candidates = self._find_sources(
            target_name=data.target_name,
            target_description=data.target_description,
            target_properties=data.target_properties,
            analogy_type=data.analogy_type
        )
        
        # Convert to dict format for storage
        candidates_dict = [
            {
                'name': c.get('name', c.name if hasattr(c, 'name') else ''),
                'description': c.get('description', c.description if hasattr(c, 'description') else ''),
                'domain': c.get('domain', c.domain if hasattr(c, 'domain') else ''),
                'score': c.get('score', c.similarity_score if hasattr(c, 'similarity_score') else 0.0),
                'rank': c.get('rank', c.rank if hasattr(c, 'rank') else i+1)
            }
            for i, c in enumerate(source_candidates)
        ]
        
        data.source_candidates = candidates_dict
        data.selected_source = candidates_dict[0] if candidates_dict else None
        
        return data
    
    def _find_sources(
        self,
        target_name: str,
        target_description: Optional[str] = None,
        target_properties: Optional[List[str]] = None,
        analogy_type: Optional[str] = None,
        top_k: int = 10
    ) -> List[Dict[str, Any]]:
        """
        Find source candidates
        
        Args:
            target_name: Name of target concept
            target_description: Optional description
            target_properties: Optional list of properties
            analogy_type: Optional analogy type classification
            top_k: Number of candidates to return
            
        Returns:
            List of source candidate dictionaries
        """
        raise NotImplementedError("Subclasses must implement _find_sources method")


class EmbeddingSourceFinder(SourceFinder):
    """
    Embedding-based source finder using RAG
    """
    
    def __init__(
        self,
        corpus_path: str,
        embedding_mode: str = "name_background",
        openai_api_key: Optional[str] = None,
        top_k: int = 10,
        name: str = None
    ):
        """
        Initialize embedding-based source finder
        
        Args:
            corpus_path: Path to SCAR CSV file
            embedding_mode: Embedding mode (name_only, name_background, name_properties, name_properties_background)
            openai_api_key: Optional OpenAI API key
            top_k: Number of top candidates to return
            name: Optional module name
        """
        super().__init__(name=name)
        if not RAG_AVAILABLE:
            raise ImportError("RAG source finder not available. Check stage1_analysis/source_finding/rag_source_finder.py")
        
        self.top_k = top_k
        self.rag_finder = RAGSourceFinder(
            openai_api_key=openai_api_key,
            embedding_mode=embedding_mode
        )
        self.rag_finder.load_corpus_from_csv(corpus_path)
        self.rag_finder.embed_corpus()
    
    def _find_sources(
        self,
        target_name: str,
        target_description: Optional[str] = None,
        target_properties: Optional[List[str]] = None,
        analogy_type: Optional[str] = None,
        top_k: int = None
    ) -> List[Dict[str, Any]]:
        """Find sources using embeddings"""
        if top_k is None:
            top_k = self.top_k
        
        # Convert properties list to string
        properties_str = ", ".join(target_properties) if target_properties else ""
        
        # Find sources
        candidates = self.rag_finder.find_source(
            target_name=target_name,
            target_background=target_description or "",
            target_properties=properties_str,
            top_k=top_k
        )
        
        # Convert SourceCandidate objects to dicts
        return [
            {
                'name': c.name,
                'description': c.description,
                'domain': c.domain,
                'score': c.similarity_score,
                'rank': c.rank
            }
            for c in candidates
        ]


class LLMSourceFinder(SourceFinder):
    """
    LLM-based source finder using open search
    """
    
    def __init__(
        self,
        corpus_path: str,
        model_name: str = "gpt-4o-mini",
        llm_client: Optional[LLMClient] = None,
        top_k: int = 10,
        name: str = None
    ):
        """
        Initialize LLM-based source finder
        
        Args:
            corpus_path: Path to SCAR CSV file
            model_name: Name of the LLM model to use
            llm_client: Optional LLM client
            top_k: Number of top candidates to return
            name: Optional module name
        """
        super().__init__(name=name)
        if not DSPY_AVAILABLE:
            raise ImportError("DSPy not available. Install with: pip install dspy-ai")
        
        import pandas as pd
        
        self.corpus_path = corpus_path
        self.corpus_df = pd.read_csv(corpus_path)
        self.model_name = model_name
        self.llm_client = llm_client or LLMClient()
        self.top_k = top_k
        
        # Setup DSPy
        self.adapter = DSPyAdapter(self.llm_client, model_name)
        self.lm = self.adapter.get_dspy_lm()
        dspy.configure(lm=self.lm)
        
        # Define DSPy signature
        class SourceFinding(dspy.Signature):
            """Find the best source concept for creating an analogy with a target concept."""
            target_concept: str = dspy.InputField(desc="The target concept to find a source for")
            target_description: str = dspy.InputField(desc="Description of the target concept")
            target_properties: str = dspy.InputField(desc="Key properties of the target concept")
            analogy_type: str = dspy.InputField(desc="The type of analogy to create")
            candidate_sources: str = dspy.InputField(desc="List of candidate source concepts with descriptions")
            best_source: str = dspy.OutputField(desc="The best source concept name for the analogy")
            reasoning: str = dspy.OutputField(desc="Explanation of why this source is best")
        
        self.predictor = dspy.ChainOfThought(SourceFinding)
    
    def _find_sources(
        self,
        target_name: str,
        target_description: Optional[str] = None,
        target_properties: Optional[List[str]] = None,
        analogy_type: Optional[str] = None,
        top_k: int = None
    ) -> List[Dict[str, Any]]:
        """Find sources using LLM"""
        if top_k is None:
            top_k = self.top_k
        
        # Get unique sources from corpus
        unique_sources = self.corpus_df[['system_b', 'system_b_background', 'system_b_domain']].drop_duplicates()
        
        # Format candidate sources
        candidates_list = []
        for _, row in unique_sources.head(50).iterrows():  # Limit to 50 for prompt size
            candidates_list.append(f"{row['system_b']} ({row['system_b_domain']}): {row['system_b_background'][:200]}")
        
        candidates_str = "\n".join(candidates_list)
        
        # Use LLM to find best source
        props_str = ", ".join(target_properties) if target_properties else ""
        analogy_type_str = analogy_type or "FUNCTION (ROLE / PURPOSE)"
        
        result = self.predictor(
            target_concept=target_name,
            target_description=target_description or "",
            target_properties=props_str,
            analogy_type=analogy_type_str,
            candidate_sources=candidates_str
        )
        
        best_source_name = result.best_source.strip()
        
        # Find the source in corpus
        source_row = unique_sources[unique_sources['system_b'].str.contains(best_source_name, case=False, na=False)]
        if len(source_row) > 0:
            source_row = source_row.iloc[0]
            return [{
                'name': source_row['system_b'],
                'description': source_row['system_b_background'],
                'domain': source_row['system_b_domain'],
                'score': 1.0,  # LLM selected, so high confidence
                'rank': 1,
                'reasoning': result.reasoning
            }]
        else:
            # Fallback: return first candidate
            if len(unique_sources) > 0:
                first = unique_sources.iloc[0]
                return [{
                    'name': first['system_b'],
                    'description': first['system_b_background'],
                    'domain': first['system_b_domain'],
                    'score': 0.5,
                    'rank': 1,
                    'reasoning': "Fallback: LLM selection not found in corpus"
                }]
            return []

