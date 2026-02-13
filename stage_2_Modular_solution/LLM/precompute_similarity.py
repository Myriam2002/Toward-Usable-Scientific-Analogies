"""
Precompute Semantic Similarity for Gold Sources
Computes and saves embeddings for all unique gold sources from SCAR dataset
Uses SentenceTransformer's all-MiniLM-L6-v2 model (same as mapping analysis)
"""

import os
import pickle
import pandas as pd
import numpy as np
from typing import List, Dict, Tuple, Optional
from tqdm import tqdm
import json
import ast

# Load environment variables from .env file
from dotenv import load_dotenv
env_paths = [
    os.path.join(os.path.dirname(__file__), '..', '..', '.env'),  # ../../.env (root)
    os.path.join(os.path.dirname(__file__), '.env'),  # ./env (local)
]
for env_path in env_paths:
    if os.path.exists(env_path):
        load_dotenv(env_path)
        break

# Import config
from config import (
    SCAR_PATH, 
    EMBEDDINGS_PATH, 
    EMBEDDING_MODEL, 
    SIMILARITY_THRESHOLD,
    TARGET_EMBEDDINGS_PATH,
    TARGET_WITH_SUBCONCEPTS_EMBEDDINGS_PATH
)


class SentenceTransformerEmbedder:
    """
    Wrapper for SentenceTransformer embeddings.
    Uses all-MiniLM-L6-v2 for consistency with mapping analysis evaluation.
    """
    
    _model = None  # Class-level cache for lazy loading
    
    def __init__(self):
        """Initialize the SentenceTransformer model (lazy loaded)."""
        self.model_name = EMBEDDING_MODEL
    
    def _get_model(self):
        """Lazy load the model (shared across instances)."""
        if SentenceTransformerEmbedder._model is None:
            from sentence_transformers import SentenceTransformer
            print(f"Loading SentenceTransformer model: {self.model_name}")
            SentenceTransformerEmbedder._model = SentenceTransformer(self.model_name)
        return SentenceTransformerEmbedder._model
    
    def encode(self, texts: List[str], show_progress_bar: bool = False) -> np.ndarray:
        """
        Encode texts into embeddings using SentenceTransformer.
        
        Args:
            texts: List of texts to encode
            show_progress_bar: Whether to show progress bar
            
        Returns:
            numpy array of embeddings
        """
        model = self._get_model()
        embeddings = model.encode(texts, show_progress_bar=show_progress_bar)
        return np.array(embeddings)
    
    def encode_single(self, text: str) -> np.ndarray:
        """
        Encode a single text into an embedding.
        
        Args:
            text: Text to encode
            
        Returns:
            numpy array embedding
        """
        model = self._get_model()
        return model.encode([text])[0]


class OpenAIEmbedder:
    """
    Wrapper for OpenAI text-embedding-3-small embeddings.
    Used for target-analogy comparison (Top-1-embedding).
    """
    
    _client = None  # Class-level cache
    
    def __init__(self, model: str = "text-embedding-3-small"):
        """Initialize the OpenAI embedder."""
        self.model = model
    
    def _get_client(self):
        """Lazy load the OpenAI client."""
        if OpenAIEmbedder._client is None:
            from openai import OpenAI
            OpenAIEmbedder._client = OpenAI()
        return OpenAIEmbedder._client
    
    def encode(self, texts: List[str], show_progress_bar: bool = False) -> np.ndarray:
        """
        Encode texts into embeddings using OpenAI API.
        
        Args:
            texts: List of texts to encode
            show_progress_bar: Whether to show progress bar (ignored for API calls)
            
        Returns:
            numpy array of embeddings
        """
        client = self._get_client()
        response = client.embeddings.create(input=texts, model=self.model)
        embeddings = [item.embedding for item in response.data]
        return np.array(embeddings)
    
    def encode_single(self, text: str) -> np.ndarray:
        """
        Encode a single text into an embedding.
        
        Args:
            text: Text to encode
            
        Returns:
            numpy array embedding
        """
        client = self._get_client()
        response = client.embeddings.create(input=[text], model=self.model)
        return np.array(response.data[0].embedding)


class TargetEmbeddingCache:
    """
    Manages precomputed target embeddings for efficient reuse.
    Loads embeddings from disk and provides lookup by target name.
    """
    
    _instance = None  # Singleton instance
    _target_only_cache = None
    _target_with_subconcepts_cache = None
    
    def __init__(self):
        """Initialize the cache (lazy loading)."""
        self.target_only_embeddings = {}  # target -> embedding
        self.target_with_subconcepts_embeddings = {}  # (target, subconcepts) -> embedding
        self._loaded = False
    
    def _load_embeddings(self):
        """Lazy load embeddings from disk."""
        if self._loaded:
            return
        
        # Load target-only embeddings
        if os.path.exists(TARGET_EMBEDDINGS_PATH):
            with open(TARGET_EMBEDDINGS_PATH, 'rb') as f:
                data = pickle.load(f)
                self.target_only_embeddings = data.get('embeddings', {})
                print(f"Loaded {len(self.target_only_embeddings)} target-only embeddings")
        else:
            print(f"Warning: Target embeddings not found at {TARGET_EMBEDDINGS_PATH}")
            print("Run precompute_target_embeddings() first")
        
        # Load target-with-subconcepts embeddings
        if os.path.exists(TARGET_WITH_SUBCONCEPTS_EMBEDDINGS_PATH):
            with open(TARGET_WITH_SUBCONCEPTS_EMBEDDINGS_PATH, 'rb') as f:
                data = pickle.load(f)
                self.target_with_subconcepts_embeddings = data.get('embeddings', {})
                print(f"Loaded {len(self.target_with_subconcepts_embeddings)} target-with-subconcepts embeddings")
        else:
            print(f"Warning: Target-with-subconcepts embeddings not found at {TARGET_WITH_SUBCONCEPTS_EMBEDDINGS_PATH}")
            print("Run precompute_target_embeddings() first")
        
        self._loaded = True
    
    def get_target_embedding(self, target: str, target_subconcepts: Optional[str] = None, use_subconcepts: bool = False) -> Optional[np.ndarray]:
        """
        Get precomputed target embedding.
        
        Args:
            target: Target concept name
            target_subconcepts: Comma-separated sub-concepts (or None)
            use_subconcepts: Whether to use sub-concepts version
            
        Returns:
            Precomputed embedding or None if not found
        """
        self._load_embeddings()
        
        target_key = target.lower().strip()
        
        if use_subconcepts and target_subconcepts:
            # Look up in target-with-subconcepts cache
            subconcepts_key = target_subconcepts.lower().strip()
            cache_key = (target_key, subconcepts_key)
            return self.target_with_subconcepts_embeddings.get(cache_key)
        else:
            # Look up in target-only cache
            return self.target_only_embeddings.get(target_key)
    
    @classmethod
    def get_instance(cls):
        """Get singleton instance."""
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance


def find_top1_by_embedding(
    target: str,
    target_subconcepts: Optional[str],
    generated_analogies: List[str],
    analogy_subconcepts: Optional[List[str]] = None,
    use_subconcepts: bool = False
) -> Dict:
    """
    Find the top-1 analogy by embedding similarity between TARGET and generated analogies.
    Uses OpenAI text-embedding-3-small for comparison.
    Uses precomputed target embeddings when available.
    
    Args:
        target: The target concept name
        target_subconcepts: Comma-separated target sub-concepts (or None)
        generated_analogies: List of generated analogy concept names
        analogy_subconcepts: List of sub-concepts for each analogy (or None)
        use_subconcepts: If True, include sub-concepts in the embedding comparison
        
    Returns:
        Dict with:
            - top1_embedding: The analogy with highest similarity to target
            - top1_embedding_score: The similarity score
            - all_scores: Dict of analogy -> similarity score
    """
    if not generated_analogies:
        return {
            'top1_embedding': '',
            'top1_embedding_score': 0.0,
            'all_scores': {}
        }
    
    embedder = OpenAIEmbedder()
    cache = TargetEmbeddingCache.get_instance()
    
    # Try to get precomputed target embedding
    target_embedding = cache.get_target_embedding(target, target_subconcepts, use_subconcepts)
    
    if target_embedding is None:
        # Fallback: compute on-the-fly if not precomputed
        target_text = target.lower().strip()
        if use_subconcepts and target_subconcepts:
            target_text = f"{target_text}: {target_subconcepts.lower()}"
        target_embedding = embedder.encode_single(target_text)
    
    # Build analogy texts (with or without sub-concepts)
    analogy_texts = []
    for i, analogy in enumerate(generated_analogies):
        analogy_text = analogy.lower().strip()
        if use_subconcepts and analogy_subconcepts and i < len(analogy_subconcepts) and analogy_subconcepts[i]:
            analogy_text = f"{analogy_text}: {analogy_subconcepts[i].lower()}"
        analogy_texts.append(analogy_text)
    
    # Compute analogy embeddings (always computed on-the-fly since they're generated)
    analogy_embeddings = embedder.encode(analogy_texts)
    
    # Compute cosine similarities
    all_scores = {}
    best_idx = 0
    best_score = -1
    
    for i, (analogy, analogy_emb) in enumerate(zip(generated_analogies, analogy_embeddings)):
        sim = np.dot(target_embedding, analogy_emb) / (
            np.linalg.norm(target_embedding) * np.linalg.norm(analogy_emb) + 1e-8
        )
        sim = float(sim)
        all_scores[analogy] = round(sim, 4)
        
        if sim > best_score:
            best_score = sim
            best_idx = i
    
    return {
        'top1_embedding': generated_analogies[best_idx],
        'top1_embedding_score': round(best_score, 4),
        'all_scores': all_scores
    }


class SemanticMatcher:
    """
    Handles semantic similarity matching between generated analogies and gold sources.
    Uses precomputed embeddings for efficiency.
    
    Compares ONLY source concept names (no sub-concepts) for unified evaluation
    across both targetonly and withsub generation modes.
    """
    
    def __init__(self, embeddings_path: str = EMBEDDINGS_PATH):
        """
        Initialize the semantic matcher.
        
        Args:
            embeddings_path: Path to the precomputed embeddings pickle file
        """
        self.embeddings_path = embeddings_path
        self.embedder = None
        self.gold_sources = []
        self.gold_embeddings = None
        
        self._load_embeddings()
    
    def _load_embeddings(self):
        """Load precomputed embeddings from disk."""
        if os.path.exists(self.embeddings_path):
            with open(self.embeddings_path, 'rb') as f:
                data = pickle.load(f)
                self.gold_sources = data['gold_sources']
                self.gold_embeddings = data['embeddings']
                print(f"Loaded {len(self.gold_sources)} gold source embeddings")
        else:
            print(f"Warning: Embeddings file not found at {self.embeddings_path}")
            print("Run precompute_similarity.py first to generate embeddings")
    
    def _get_embedder(self) -> SentenceTransformerEmbedder:
        """Lazy load the SentenceTransformer embedder."""
        if self.embedder is None:
            self.embedder = SentenceTransformerEmbedder()
        return self.embedder
    
    def compute_similarity(self, text: str) -> List[Tuple[str, float]]:
        """
        Compute semantic similarity between a text and all gold sources.
        
        Args:
            text: The text to compare (e.g., a generated analogy concept name)
            
        Returns:
            List of (gold_source, similarity_score) tuples for sources >= threshold
        """
        if self.gold_embeddings is None:
            return []
        
        query_text = text.lower().strip()
        embedder = self._get_embedder()
        text_embedding = embedder.encode_single(query_text)
        
        # Compute cosine similarities
        similarities = np.dot(self.gold_embeddings, text_embedding) / (
            np.linalg.norm(self.gold_embeddings, axis=1) * np.linalg.norm(text_embedding)
        )
        
        # Get sources above threshold
        matches = []
        for i, sim in enumerate(similarities):
            if sim >= SIMILARITY_THRESHOLD:
                matches.append((self.gold_sources[i], float(sim)))
        
        # Sort by similarity descending
        matches.sort(key=lambda x: x[1], reverse=True)
        return matches
    
    def find_best_gold_match(self, analogy: str) -> Dict:
        """
        Find the best matching gold source for a generated analogy.
        
        Uses precomputed embeddings for efficient lookup.
        
        Args:
            analogy: Generated analogy concept name
            
        Returns:
            Dict with best match info: {gold_source, similarity}
        """
        if self.gold_embeddings is None:
            return {'gold_source': None, 'similarity': 0.0}
        
        query_text = analogy.lower().strip()
        embedder = self._get_embedder()
        query_embedding = embedder.encode_single(query_text)
        
        # Compute cosine similarities with all precomputed embeddings
        similarities = np.dot(self.gold_embeddings, query_embedding) / (
            np.linalg.norm(self.gold_embeddings, axis=1) * np.linalg.norm(query_embedding) + 1e-8
        )
        
        # Find best match
        best_idx = np.argmax(similarities)
        best_sim = float(similarities[best_idx])
        
        return {
            'gold_source': self.gold_sources[best_idx],
            'similarity': round(best_sim, 4)
        }
    
    def find_semantic_match(
        self, 
        generated_analogies: List[str], 
        all_gold_sources: List[str]
    ) -> Dict:
        """
        Find semantic matches for gold source(s) in generated analogies.
        
        Compares ONLY the source concept names (no sub-concepts enrichment).
        This provides unified evaluation for both targetonly and withsub modes.
        
        Args:
            generated_analogies: List of generated analogy concepts (source names only)
            all_gold_sources: List of ALL gold sources for this target
            
        Returns:
            Dict with:
                - gold_ranks_list: List of all exact match ranks in order (e.g., [1, 3, 5])
                - sem_gold_ranks_list: List of all semantic match ranks in order (e.g., [2, 4])
                - gold_ranks: Dict mapping generated_analogy -> exact rank (stores the generated analogy that matched)
                - sem_gold_ranks: Dict mapping generated_analogy -> semantic rank (stores the generated analogy that matched)
                - found_gold_sources: List of generated analogies that matched exactly (not the gold sources)
                - sem_gold_sources: List of generated analogies that matched semantically (not the gold sources)
                - similarity_per_gold: Dict of gold_source -> {scores, highest, avg}
        """
        # Clean and validate gold sources
        gold_sources_to_check = [str(gs).strip() for gs in all_gold_sources if str(gs).strip()]
        
        if not gold_sources_to_check:
            return {
                'gold_ranks_list': [],
                'sem_gold_ranks_list': [],
                'gold_ranks': {},
                'sem_gold_ranks': {},
                'found_gold_sources': [],
                'sem_gold_sources': [],
                'similarity_per_gold': {}
            }
        
        # Compute analogy embeddings (concept names only)
        embedder = self._get_embedder()
        analogy_texts = [analogy.lower().strip() for analogy in generated_analogies]
        analogy_embeddings = [embedder.encode_single(text) for text in analogy_texts]
        
        # Track results: store the GENERATED ANALOGY that matched (not the gold source)
        gold_ranks = {}  # generated_analogy -> exact rank (best rank if multiple gold sources match same analogy)
        sem_gold_ranks = {}  # generated_analogy -> semantic rank (best rank if multiple gold sources match same analogy)
        found_gold_sources = []  # List of generated analogies that matched exactly
        sem_gold_sources = []  # List of generated analogies that matched semantically
        similarity_per_gold = {}  # gold_source -> {scores, highest, avg}
        
        for gold in gold_sources_to_check:
            gold_lower = gold.lower().strip()
            gold_embedding = embedder.encode_single(gold_lower)
            
            # Compute similarity scores for this gold source vs all analogies
            scores = {}
            for i, (analogy, analogy_embedding) in enumerate(zip(generated_analogies, analogy_embeddings)):
                sim = np.dot(gold_embedding, analogy_embedding) / (
                    np.linalg.norm(gold_embedding) * np.linalg.norm(analogy_embedding) + 1e-8
                )
                scores[analogy] = round(float(sim), 4)
            
            sim_values = list(scores.values())
            highest_sim = max(sim_values) if sim_values else 0.0
            avg_sim = sum(sim_values) / len(sim_values) if sim_values else 0.0
            
            similarity_per_gold[gold] = {
                'scores': scores,
                'highest': round(highest_sim, 4),
                'avg': round(avg_sim, 4)
            }
            
            # Check for exact match - store the GENERATED ANALOGY that matched
            for i, analogy in enumerate(generated_analogies):
                analogy_lower = analogy.lower().strip()
                if gold_lower == analogy_lower or gold_lower in analogy_lower or analogy_lower in gold_lower:
                    rank = i + 1  # 1-indexed
                    # Store the generated analogy (not the gold source)
                    # If multiple gold sources match the same analogy, keep the best (lowest) rank
                    if analogy not in gold_ranks or rank < gold_ranks[analogy]:
                        gold_ranks[analogy] = rank
                    if analogy not in found_gold_sources:
                        found_gold_sources.append(analogy)
                    break
            
            # Check for semantic match - store the GENERATED ANALOGY that matched
            for i, analogy in enumerate(generated_analogies):
                if scores[analogy] >= SIMILARITY_THRESHOLD:
                    rank = i + 1
                    # Store the generated analogy (not the gold source)
                    # If multiple gold sources match the same analogy, keep the best (lowest) rank
                    if analogy not in sem_gold_ranks or rank < sem_gold_ranks[analogy]:
                        sem_gold_ranks[analogy] = rank
                    # Only add if not already found exactly
                    if analogy not in found_gold_sources and analogy not in sem_gold_sources:
                        sem_gold_sources.append(analogy)
                    break
        
        # Build ordered lists of ranks
        gold_ranks_list = sorted(gold_ranks.values())
        sem_gold_ranks_list = sorted(sem_gold_ranks.values())
        
        return {
            'gold_ranks_list': gold_ranks_list,  # All exact ranks in order [1, 3, 5]
            'sem_gold_ranks_list': sem_gold_ranks_list,  # All semantic ranks in order [2, 4]
            'gold_ranks': gold_ranks,  # Dict: generated_analogy -> exact rank (stores the generated analogy that matched)
            'sem_gold_ranks': sem_gold_ranks,  # Dict: generated_analogy -> semantic rank (stores the generated analogy that matched)
            'found_gold_sources': found_gold_sources,  # List of generated analogies that matched exactly
            'sem_gold_sources': sem_gold_sources,  # List of generated analogies that matched semantically
            'similarity_per_gold': similarity_per_gold  # Dict: gold -> {scores, highest, avg}
        }


def precompute_gold_embeddings(force: bool = False):
    """
    Precompute and save embeddings for all unique gold sources in SCAR dataset.
    Uses SentenceTransformer's all-MiniLM-L6-v2 model.
    
    Args:
        force: If True, recompute even if pkl file exists. If False, skip if file exists.
    """
    # Check if file already exists
    if not force and os.path.exists(EMBEDDINGS_PATH):
        print("=" * 60)
        print("Gold Source Embeddings Already Exist")
        print(f"File: {EMBEDDINGS_PATH}")
        print("Skipping precomputation. Use force=True to recompute.")
        print("=" * 60)
        # Load and return existing data
        with open(EMBEDDINGS_PATH, 'rb') as f:
            data = pickle.load(f)
        return data['gold_sources'], data['embeddings']
    
    print("=" * 60)
    print("Precomputing Gold Source Embeddings")
    print(f"Model: {EMBEDDING_MODEL}")
    print("=" * 60)
    
    # Load SCAR dataset
    print(f"\nLoading SCAR dataset from {SCAR_PATH}")
    df = pd.read_csv(SCAR_PATH)
    print(f"Loaded {len(df)} records")
    
    # Extract unique gold sources (system_b)
    gold_sources = df['system_b'].dropna().unique().tolist()
    gold_sources = [str(s).strip() for s in gold_sources if str(s).strip()]
    print(f"Found {len(gold_sources)} unique gold sources")
    
    # Initialize SentenceTransformer embedder
    print(f"\nInitializing SentenceTransformer embedder with model: {EMBEDDING_MODEL}")
    embedder = SentenceTransformerEmbedder()
    
    # Compute embeddings
    print("Computing embeddings...")
    embeddings = embedder.encode(
        [s.lower() for s in gold_sources], 
        show_progress_bar=True
    )
    
    # Save to pickle
    data = {
        'gold_sources': gold_sources,
        'embeddings': embeddings,
        'model_name': EMBEDDING_MODEL
    }
    
    print(f"\nSaving embeddings to {EMBEDDINGS_PATH}")
    with open(EMBEDDINGS_PATH, 'wb') as f:
        pickle.dump(data, f)
    
    print(f"Done! Saved {len(gold_sources)} embeddings")
    print("=" * 60)
    
    return gold_sources, embeddings


def precompute_target_embeddings(force: bool = False):
    """
    Precompute and save embeddings for all unique targets in SCAR dataset.
    Creates two files:
    1. Target-only embeddings (for targetonly mode)
    2. Target-with-subconcepts embeddings (for withsub mode)
    
    Uses OpenAI text-embedding-3-small model.
    
    Args:
        force: If True, recompute even if pkl files exist. If False, skip if both files exist.
    """
    # Check if both files already exist
    target_only_exists = os.path.exists(TARGET_EMBEDDINGS_PATH)
    target_with_sub_exists = os.path.exists(TARGET_WITH_SUBCONCEPTS_EMBEDDINGS_PATH)
    
    if not force and target_only_exists and target_with_sub_exists:
        print("=" * 60)
        print("Target Embeddings Already Exist")
        print(f"File 1: {TARGET_EMBEDDINGS_PATH}")
        print(f"File 2: {TARGET_WITH_SUBCONCEPTS_EMBEDDINGS_PATH}")
        print("Skipping precomputation. Use force=True to recompute.")
        print("=" * 60)
        # Load and return existing data
        with open(TARGET_EMBEDDINGS_PATH, 'rb') as f:
            data1 = pickle.load(f)
        with open(TARGET_WITH_SUBCONCEPTS_EMBEDDINGS_PATH, 'rb') as f:
            data2 = pickle.load(f)
        return data1['embeddings'], data2['embeddings']
    
    print("=" * 60)
    print("Precomputing Target Embeddings")
    print("Model: text-embedding-3-small (OpenAI)")
    print("=" * 60)
    
    # Load SCAR dataset
    print(f"\nLoading SCAR dataset from {SCAR_PATH}")
    df = pd.read_csv(SCAR_PATH)
    print(f"Loaded {len(df)} records")
    
    embedder = OpenAIEmbedder()
    
    # ============================================================================
    # 1. Precompute Target-Only Embeddings
    # ============================================================================
    print("\n" + "-" * 60)
    print("Step 1: Computing target-only embeddings")
    print("-" * 60)
    
    # Get unique targets
    unique_targets = df['system_a'].dropna().unique().tolist()
    unique_targets = [str(t).strip() for t in unique_targets if str(t).strip()]
    print(f"Found {len(unique_targets)} unique targets")
    
    # Compute embeddings
    print("Computing embeddings...")
    target_embeddings = {}
    for target in tqdm(unique_targets, desc="Encoding targets"):
        target_key = target.lower().strip()
        embedding = embedder.encode_single(target_key)
        target_embeddings[target_key] = embedding
    
    # Save
    data = {
        'embeddings': target_embeddings,
        'model_name': 'text-embedding-3-small',
        'num_targets': len(target_embeddings)
    }
    
    print(f"\nSaving to {TARGET_EMBEDDINGS_PATH}")
    with open(TARGET_EMBEDDINGS_PATH, 'wb') as f:
        pickle.dump(data, f)
    print(f"Done! Saved {len(target_embeddings)} target-only embeddings")
    
    # ============================================================================
    # 2. Precompute Target-With-Subconcepts Embeddings
    # ============================================================================
    print("\n" + "-" * 60)
    print("Step 2: Computing target-with-subconcepts embeddings")
    print("-" * 60)
    
    # Extract sub-concepts from mappings_parsed (system_a side)
    target_with_subconcepts = {}  # (target, subconcepts) -> embedding
    
    for _, row in df.iterrows():
        target = str(row['system_a']).strip()
        if not target or pd.isna(row['system_a']):
            continue
        
        # Extract sub-concepts from mappings_parsed
        subconcepts = ""
        mappings_str = row.get('mappings_parsed', '[]')
        if pd.notna(mappings_str):
            try:
                mappings = ast.literal_eval(mappings_str)
                # Extract the first element (system_a side) from each mapping
                sub_list = [m[0] for m in mappings if isinstance(m, list) and len(m) >= 1]
                if sub_list:
                    subconcepts = ", ".join(sub_list)
            except (ValueError, SyntaxError, TypeError):
                pass
        
        # Create key
        target_key = target.lower().strip()
        subconcepts_key = subconcepts.lower().strip() if subconcepts else ""
        cache_key = (target_key, subconcepts_key)
        
        # Only compute if not already done
        if cache_key not in target_with_subconcepts:
            if subconcepts:
                target_text = f"{target_key}: {subconcepts_key}"
            else:
                target_text = target_key
            
            embedding = embedder.encode_single(target_text)
            target_with_subconcepts[cache_key] = embedding
    
    print(f"Found {len(target_with_subconcepts)} unique (target, subconcepts) combinations")
    
    # Save
    data = {
        'embeddings': target_with_subconcepts,
        'model_name': 'text-embedding-3-small',
        'num_combinations': len(target_with_subconcepts)
    }
    
    print(f"\nSaving to {TARGET_WITH_SUBCONCEPTS_EMBEDDINGS_PATH}")
    with open(TARGET_WITH_SUBCONCEPTS_EMBEDDINGS_PATH, 'wb') as f:
        pickle.dump(data, f)
    print(f"Done! Saved {len(target_with_subconcepts)} target-with-subconcepts embeddings")
    
    print("\n" + "=" * 60)
    print("Precomputation complete!")
    print("=" * 60)
    
    return target_embeddings, target_with_subconcepts


def test_semantic_matcher():
    """Test the semantic matcher with some examples."""
    print("\n" + "=" * 60)
    print("Testing Semantic Matcher")
    print("=" * 60)
    print("NOTE: Evaluation compares ONLY source concept names (no sub-concepts)")
    print("This provides unified metrics for both targetonly and withsub modes")
    print("=" * 60)
    
    matcher = SemanticMatcher()
    
    # Test examples: comparing source concept names only
    test_cases = [
        (["timepiece", "watch", "alarm", "timer", "metronome"], ["clock"]),
        (["clock"], ["biological clock"]),
        # Test with multiple gold sources
        (["timepiece", "watch", "alarm"], ["clock", "watch", "timer"]),
        # More diverse examples
        (["engine", "motor", "pump", "generator"], ["engine", "motor"]),
        (["library", "archive", "database", "filing cabinet"], ["library"]),
    ]
    
    for generated_list, gold_sources in test_cases:
        result = matcher.find_semantic_match(generated_list, gold_sources)
        print(f"\nGenerated: {generated_list}")
        print(f"Gold Sources: {gold_sources}")
        print(f"  Exact Ranks (all in order): {result['gold_ranks_list']}")
        print(f"  Semantic Ranks (all in order): {result['sem_gold_ranks_list']}")
        print(f"  Found Generated Analogies (exact match): {result['found_gold_sources']}")
        print(f"  Found Generated Analogies (semantic match): {result['sem_gold_sources']}")
        print(f"  Exact Ranks by generated analogy: {result['gold_ranks']}")
        print(f"  Semantic Ranks by generated analogy: {result['sem_gold_ranks']}")
        print(f"\n  Similarity per Gold Source (threshold: {SIMILARITY_THRESHOLD}):")
        for gs, stats in result['similarity_per_gold'].items():
            print(f"    '{gs}': highest={stats['highest']}, avg={stats['avg']}")
            for analogy, sim in stats['scores'].items():
                above = "✓" if sim >= SIMILARITY_THRESHOLD else "✗"
                print(f"      {above} vs '{analogy}': {sim:.4f}")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Precompute semantic embeddings")
    parser.add_argument("--mode", choices=["gold", "target", "both", "test"], default="both",
                        help="Mode: gold (gold sources), target (targets), both, test")
    parser.add_argument("--force", action="store_true",
                        help="Force recomputation even if pkl files already exist")
    args = parser.parse_args()
    
    if args.mode in ["gold", "both"]:
        print("\n" + "=" * 70)
        print("Computing gold source embeddings (source concepts only)")
        print("NOTE: Unified evaluation for both targetonly and withsub modes")
        print("=" * 70)
        precompute_gold_embeddings(force=args.force)
    
    if args.mode in ["target", "both"]:
        print("\n" + "=" * 70)
        print("Computing target embeddings (for top-1-embedding selection)")
        print("=" * 70)
        precompute_target_embeddings(force=args.force)
    
    if args.mode == "test":
        print("\n" + "=" * 70)
        print("Running semantic matcher tests")
        print("=" * 70)
        test_semantic_matcher()
