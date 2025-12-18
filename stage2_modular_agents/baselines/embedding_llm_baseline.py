"""
Embedding + LLM Baseline
Embedding-based retrieval, LLM chooses 3 from top 10
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


class EmbeddingLLMBaseline:
    """
    Baseline: Embedding retrieval, LLM chooses 3 from top 10
    """
    
    def __init__(
        self,
        corpus_path: str,
        embedding_mode: str = "name_background",
        model_name: str = "gpt-4o-mini",
        openai_api_key: Optional[str] = None,
        llm_client: Optional[LLMClient] = None
    ):
        """
        Initialize embedding + LLM baseline
        
        Args:
            corpus_path: Path to SCAR CSV file
            embedding_mode: Embedding mode
            model_name: LLM model name
            openai_api_key: Optional OpenAI API key
            llm_client: Optional LLM client
        """
        if not RAG_AVAILABLE:
            raise ImportError("RAG source finder not available")
        if not DSPY_AVAILABLE:
            raise ImportError("DSPy not available")
        
        # Setup embedding finder
        self.rag_finder = RAGSourceFinder(
            openai_api_key=openai_api_key,
            embedding_mode=embedding_mode
        )
        self.rag_finder.load_corpus_from_csv(corpus_path)
        self.rag_finder.embed_corpus()
        
        # Setup LLM
        self.model_name = model_name
        self.llm_client = llm_client or LLMClient()
        self.adapter = DSPyAdapter(self.llm_client, model_name)
        self.lm = self.adapter.get_dspy_lm()
        dspy.configure(lm=self.lm)
        
        # Define DSPy signature for selection
        class SourceSelection(dspy.Signature):
            """Select the best 3 source concepts from candidates for creating an analogy."""
            target_concept: str = dspy.InputField(desc="The target concept")
            candidate_sources: str = dspy.InputField(desc="List of candidate source concepts with descriptions and scores")
            selected_sources: List[str] = dspy.OutputField(desc="List of 3 best source concept names")
            reasoning: str = dspy.OutputField(desc="Explanation of why these sources were selected")
        
        self.predictor = dspy.ChainOfThought(SourceSelection)
    
    def generate_analogy(
        self,
        target_name: str,
        target_description: Optional[str] = None,
        target_properties: Optional[List[str]] = None,
        top_k_embedding: int = 10,
        top_k_final: int = 3
    ) -> List[Dict[str, Any]]:
        """
        Generate analogy using embedding + LLM baseline
        
        Args:
            target_name: Name of target concept
            target_description: Optional description
            target_properties: Optional properties
            top_k_embedding: Number of candidates from embedding (default: 10)
            top_k_final: Number of final selections (default: 3)
            
        Returns:
            List of selected source candidates
        """
        # Step 1: Get top-k from embedding
        properties_str = ", ".join(target_properties) if target_properties else ""
        
        embedding_candidates = self.rag_finder.find_source(
            target_name=target_name,
            target_background=target_description or "",
            target_properties=properties_str,
            top_k=top_k_embedding
        )
        
        # Step 2: Format candidates for LLM
        candidates_list = []
        for c in embedding_candidates:
            candidates_list.append(
                f"{c.name} (score: {c.similarity_score:.3f}): {c.description[:200]}"
            )
        
        candidates_str = "\n".join([f"{i+1}. {c}" for i, c in enumerate(candidates_list)])
        
        # Step 3: LLM selects best 3
        result = self.predictor(
            target_concept=target_name,
            candidate_sources=candidates_str
        )
        
        # Step 4: Match selected sources back to candidates
        selected_names = result.selected_sources
        if isinstance(selected_names, str):
            # Try to parse as list
            import ast
            try:
                selected_names = ast.literal_eval(selected_names)
            except:
                # Split by comma or newline
                selected_names = [s.strip() for s in selected_names.replace('\n', ',').split(',') if s.strip()]
        
        # Find matches in embedding candidates
        selected_candidates = []
        for name in selected_names[:top_k_final]:
            # Find best match
            best_match = None
            best_score = 0.0
            
            for c in embedding_candidates:
                # Check if name matches (case-insensitive, partial match)
                if name.lower() in c.name.lower() or c.name.lower() in name.lower():
                    if c.similarity_score > best_score:
                        best_score = c.similarity_score
                        best_match = c
            
            if best_match:
                selected_candidates.append({
                    'name': best_match.name,
                    'description': best_match.description,
                    'domain': best_match.domain,
                    'score': best_match.similarity_score,
                    'rank': len(selected_candidates) + 1,
                    'llm_selected': True,
                    'reasoning': result.reasoning
                })
        
        # If LLM didn't select enough, fill with top embedding results
        while len(selected_candidates) < top_k_final and len(selected_candidates) < len(embedding_candidates):
            next_candidate = embedding_candidates[len(selected_candidates)]
            # Check if already added
            if not any(c['name'] == next_candidate.name for c in selected_candidates):
                selected_candidates.append({
                    'name': next_candidate.name,
                    'description': next_candidate.description,
                    'domain': next_candidate.domain,
                    'score': next_candidate.similarity_score,
                    'rank': len(selected_candidates) + 1,
                    'llm_selected': False
                })
        
        return selected_candidates

