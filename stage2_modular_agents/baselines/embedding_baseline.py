"""
Embedding Baseline
Embedding-based retrieval, take top 3
"""

from typing import List, Dict, Any, Optional
import sys
from pathlib import Path

# Import RAG source finder
current_dir = Path(__file__).parent.parent.parent
source_finding_dir = current_dir / "stage1_analysis" / "source_finding"
if str(source_finding_dir) not in sys.path:
    sys.path.insert(0, str(source_finding_dir))

try:
    from rag_source_finder import RAGSourceFinder
    RAG_AVAILABLE = True
except ImportError:
    RAG_AVAILABLE = False


class EmbeddingBaseline:
    """
    Baseline: Embedding-based retrieval, take top 3
    """
    
    def __init__(
        self,
        corpus_path: str,
        embedding_mode: str = "name_background",
        openai_api_key: Optional[str] = None
    ):
        """
        Initialize embedding baseline
        
        Args:
            corpus_path: Path to SCAR CSV file
            embedding_mode: Embedding mode
            openai_api_key: Optional OpenAI API key
        """
        if not RAG_AVAILABLE:
            raise ImportError("RAG source finder not available")
        
        self.rag_finder = RAGSourceFinder(
            openai_api_key=openai_api_key,
            embedding_mode=embedding_mode
        )
        self.rag_finder.load_corpus_from_csv(corpus_path)
        self.rag_finder.embed_corpus()
    
    def generate_analogy(
        self,
        target_name: str,
        target_description: Optional[str] = None,
        target_properties: Optional[List[str]] = None,
        top_k: int = 3
    ) -> List[Dict[str, Any]]:
        """
        Generate analogy using embedding baseline
        
        Args:
            target_name: Name of target concept
            target_description: Optional description
            target_properties: Optional properties
            top_k: Number of top sources to return (default: 3)
            
        Returns:
            List of top-k source candidates
        """
        # Convert properties to string
        properties_str = ", ".join(target_properties) if target_properties else ""
        
        # Find top-k sources
        candidates = self.rag_finder.find_source(
            target_name=target_name,
            target_background=target_description or "",
            target_properties=properties_str,
            top_k=top_k
        )
        
        # Convert to dict format
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

