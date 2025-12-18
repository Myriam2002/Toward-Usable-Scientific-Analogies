"""
RAG-Based Source Finder
Uses OpenAI embeddings and vector similarity search to find analogous sources
"""

import os
import numpy as np
import pandas as pd
from typing import List, Dict, Tuple, Optional, Literal
from dataclasses import dataclass
import time
import ast


@dataclass
class SourceCandidate:
    """A potential source candidate with metadata"""
    name: str
    description: str
    domain: str
    similarity_score: float
    rank: int


class RAGSourceFinder:
    """
    RAG-based source finder using embedding similarity
    """
    
    # Define embedding modes
    EMBEDDING_MODES = Literal["name_only", "name_background", "name_properties", "name_properties_background"]
    
    def __init__(self, openai_api_key: Optional[str] = None, embedding_mode: EMBEDDING_MODES = "name_background"):
        """
        Initialize RAG source finder
        
        Args:
            openai_api_key: OpenAI API key (or use OPENAI_API_KEY env var)
            embedding_mode: What to embed for sources and queries:
                - "name_only": Just the system name
                - "name_background": Name + background description
                - "name_properties": Name + properties
                - "name_properties_background": Name + properties + background
        """
        self.api_key = openai_api_key or os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError("OpenAI API key required. Set OPENAI_API_KEY env var or pass as parameter")
        
        # Initialize OpenAI client
        from openai import OpenAI
        self.client = OpenAI(api_key=self.api_key)
        
        # Embedding model
        self.embedding_model = "text-embedding-3-small"
        
        # Embedding mode
        self.embedding_mode = embedding_mode
        
        # Corpus storage
        self.corpus_df = None
        self.corpus_embeddings = None
    
    def _extract_properties_from_mappings(self, mappings_parsed, extract_target: bool = True) -> str:
        """
        Extract properties from mappings_parsed column
        
        Args:
            mappings_parsed: String representation of list of mappings
            extract_target: If True, extract target properties (first element of each pair),
                          If False, extract source properties (second element of each pair)
        
        Returns:
            String of comma-separated properties
        """
        if pd.isna(mappings_parsed) or not mappings_parsed:
            return ""
        
        try:
            mappings_list = ast.literal_eval(mappings_parsed) if isinstance(mappings_parsed, str) else mappings_parsed
            if not mappings_list:
                return ""
            
            # Extract properties based on whether we want target (index 0) or source (index 1)
            prop_index = 0 if extract_target else 1
            properties = [pair[prop_index] for pair in mappings_list if len(pair) > prop_index]
            
            # Join properties with commas
            return ", ".join(properties) if properties else ""
        except (ValueError, SyntaxError, IndexError) as e:
            # If parsing fails, return empty string
            return ""
        
    def _create_embedding_text(self, name: str, background: str = "", properties: str = "") -> str:
        """
        Create embedding text based on the current embedding mode
        
        Args:
            name: System name
            background: Background description
            properties: Properties description
            
        Returns:
            Text to embed based on the embedding mode
        """
        if self.embedding_mode == "name_only":
            return name
        elif self.embedding_mode == "name_background":
            return f"{name}. {background}" if background else name
        elif self.embedding_mode == "name_properties":
            return f"{name}. {properties}" if properties else name
        elif self.embedding_mode == "name_properties_background":
            parts = [name]
            if properties:
                parts.append(properties)
            if background:
                parts.append(background)
            return ". ".join(parts)
        else:
            raise ValueError(f"Unknown embedding mode: {self.embedding_mode}")
        
    def load_corpus_from_csv(self, csv_path: str):
        """
        Load corpus from SCAR CSV file
        
        Args:
            csv_path: Path to SCAR_cleaned_manually.csv
        """
        df = pd.read_csv(csv_path)
        
        # Extract unique sources with their metadata
        # Aggregate properties from all rows where each source appears
        sources_dict = {}
        
        for _, row in df.iterrows():
            source_name = row['system_b']
            
            if source_name not in sources_dict:
                # First occurrence - initialize
                sources_dict[source_name] = {
                    'source_name': source_name,
                    'source_domain': row['system_b_domain'],
                    'source_background': row['system_b_background'],
                    'all_properties': set()  # Use set to avoid duplicates
                }
            
            # Extract and add source properties from this row
            if 'mappings_parsed' in row:
                properties_str = self._extract_properties_from_mappings(
                    row['mappings_parsed'], 
                    extract_target=False  # False = extract source properties (system_b)
                )
                if properties_str:
                    # Split by comma and add individual properties to set
                    properties_list = [p.strip() for p in properties_str.split(',') if p.strip()]
                    sources_dict[source_name]['all_properties'].update(properties_list)
            # Fallback to system_b_properties column if it exists
            elif 'system_b_properties' in row and pd.notna(row['system_b_properties']):
                properties_str = str(row['system_b_properties'])
                properties_list = [p.strip() for p in properties_str.split(',') if p.strip()]
                sources_dict[source_name]['all_properties'].update(properties_list)
        
        # Convert to list and create embedding texts
        sources_data = []
        for source_name, source_info in sources_dict.items():
            # Join all unique properties
            properties = ", ".join(sorted(source_info['all_properties'])) if source_info['all_properties'] else ""
            
            # Create embedding text based on mode
            embedding_text = self._create_embedding_text(
                name=source_info['source_name'],
                background=source_info['source_background'],
                properties=properties
            )
            
            sources_data.append({
                'source_name': source_info['source_name'],
                'source_domain': source_info['source_domain'],
                'source_background': source_info['source_background'],
                'source_properties': properties,
                'embedding_text': embedding_text
            })
        
        self.corpus_df = pd.DataFrame(sources_data)
        print(f"Loaded {len(self.corpus_df)} unique sources from corpus (mode: {self.embedding_mode})")
        
    def embed_corpus(self):
        """
        Generate embeddings for all sources in corpus
        """
        if self.corpus_df is None:
            raise ValueError("Corpus not loaded. Call load_corpus_from_csv first")
        
        print("Generating embeddings for corpus...")
        texts = self.corpus_df['embedding_text'].tolist()
        
        # Batch embed (OpenAI supports up to 2048 texts per batch)
        embeddings = self._batch_embed(texts)
        self.corpus_embeddings = np.array(embeddings)
        print(f"Generated {len(self.corpus_embeddings)} embeddings")
        
    def _batch_embed(self, texts: List[str], batch_size: int = 100) -> List[List[float]]:
        """
        Batch embed texts using OpenAI API
        
        Args:
            texts: List of texts to embed
            batch_size: Batch size for API calls
            
        Returns:
            List of embedding vectors
        """
        all_embeddings = []
        
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            response = self.client.embeddings.create(
                input=batch,
                model=self.embedding_model
            )
            batch_embeddings = [item.embedding for item in response.data]
            all_embeddings.extend(batch_embeddings)
            time.sleep(0.1)  # Rate limiting
            
        return all_embeddings
    
    def find_source(
        self, 
        target_name: str, 
        target_background: str = "",
        target_properties: str = "",
        top_k: int = 10
    ) -> List[SourceCandidate]:
        """
        Find most similar sources for given target
        
        Args:
            target_name: Name of target concept
            target_background: Background description of target
            target_properties: Properties of target (optional)
            top_k: Number of top results to return
            
        Returns:
            List of SourceCandidate objects ranked by similarity
        """
        if self.corpus_embeddings is None:
            raise ValueError("Corpus not embedded. Call embed_corpus first")
        
        # Create query text based on embedding mode
        query_text = self._create_embedding_text(
            name=target_name,
            background=target_background,
            properties=target_properties
        )
        
        # Embed query
        query_embedding = self._embed_single(query_text)
        
        # Calculate cosine similarities
        similarities = self._cosine_similarity(query_embedding, self.corpus_embeddings)
        
        # Get top K indices
        top_k_indices = np.argsort(similarities)[::-1][:top_k]
        
        # Build results
        results = []
        for rank, idx in enumerate(top_k_indices, 1):
            results.append(SourceCandidate(
                name=self.corpus_df.iloc[idx]['source_name'],
                description=self.corpus_df.iloc[idx]['source_background'],
                domain=self.corpus_df.iloc[idx]['source_domain'],
                similarity_score=float(similarities[idx]),
                rank=rank
            ))
        
        return results
    
    def _embed_single(self, text: str) -> np.ndarray:
        """Embed a single text"""
        response = self.client.embeddings.create(
            input=[text],
            model=self.embedding_model
        )
        return np.array(response.data[0].embedding)
    
    def _cosine_similarity(self, query_vec: np.ndarray, corpus_vecs: np.ndarray) -> np.ndarray:
        """
        Calculate cosine similarity between query and corpus vectors
        
        Args:
            query_vec: Query embedding (1D array)
            corpus_vecs: Corpus embeddings (2D array)
            
        Returns:
            Array of similarity scores
        """
        # Normalize vectors
        query_norm = query_vec / np.linalg.norm(query_vec)
        corpus_norms = corpus_vecs / np.linalg.norm(corpus_vecs, axis=1, keepdims=True)
        
        # Dot product = cosine similarity for normalized vectors
        similarities = np.dot(corpus_norms, query_norm)
        return similarities
    
    def _normalize_text(self, text: str) -> str:
        """
        Normalize text for case-insensitive matching
        
        Args:
            text: Text to normalize
            
        Returns:
            Normalized text (lowercase, stripped)
        """
        return text.lower().strip()
    
    def _get_all_golden_sources(self, df: pd.DataFrame, target_name: str) -> List[str]:
        """
        Get all golden sources for a given target from the dataset
        
        Args:
            df: Dataset DataFrame
            target_name: Target system name
            
        Returns:
            List of all golden source names for this target
        """
        # Find all rows with the same system_a
        target_rows = df[df['system_a'] == target_name]
        
        # Extract all unique system_b values
        golden_sources = target_rows['system_b'].unique().tolist()
        
        return golden_sources
    
    def _find_gold_ranks(self, source_names: List[str], golden_sources: List[str]) -> Dict:
        """
        Find all golden sources in the top-k results and their ranks
        
        Args:
            source_names: List of source names from RAG results (in rank order)
            golden_sources: List of all valid golden source names
            
        Returns:
            Dictionary with:
            - best_rank: Best (lowest) rank among all golden sources, -1 if none found
            - found_golden_sources: List of golden sources found in top-k
            - golden_ranks: List of ranks where golden sources were found
            - num_golden_found: Number of golden sources found in top-k
        """
        best_rank = -1
        found_golden_sources = []
        golden_ranks = []
        
        for golden_source in golden_sources:
            # Normalize the golden source for comparison
            normalized_golden = self._normalize_text(golden_source)
            
            # Check each position in the RAG results
            for i, source_name in enumerate(source_names):
                normalized_source = self._normalize_text(source_name)
                
                # Check for exact match or if golden source is contained in source name
                if (normalized_golden == normalized_source or 
                    normalized_golden in normalized_source or 
                    normalized_source in normalized_golden):
                    
                    rank = i + 1  # Convert to 1-based ranking
                    found_golden_sources.append(golden_source)
                    golden_ranks.append(rank)
                    
                    if best_rank == -1 or rank < best_rank:
                        best_rank = rank
                    break  # Found this golden source, move to next one
        
        return {
            'best_rank': best_rank,
            'found_golden_sources': found_golden_sources,
            'golden_ranks': golden_ranks,
            'num_golden_found': len(found_golden_sources)
        }

    def evaluate_on_dataset(
        self, 
        csv_path: str, 
        top_k: int = 10,
        sample_size: Optional[int] = None
    ) -> pd.DataFrame:
        """
        Evaluate RAG on full dataset with support for multiple golden answers.
        Finds the best (lowest) rank among all golden answers in the top-k results.
        Also tracks how many golden answers were found and their specific ranks.
        
        Args:
            csv_path: Path to SCAR CSV
            top_k: Number of results to retrieve
            sample_size: If set, only evaluate on random sample
            
        Returns:
            DataFrame with results for each target, including:
            - gold_rank: Best rank among all golden sources
            - num_golden_found: Number of golden sources found in top-k
            - found_golden_sources: List of golden sources found
            - golden_ranks: List of ranks where golden sources were found
        """
        df = pd.read_csv(csv_path)
        
        if sample_size:
            df = df.sample(n=sample_size, random_state=42)
        
        results = []
        
        print(f"Evaluating on {len(df)} examples...")
        for idx, row in df.iterrows():
            target_name = row['system_a']
            target_bg = row['system_a_background']
            primary_gold_source = row['system_b']  # The primary golden source from this row
            
            # Extract target properties from mappings_parsed (first element of each pair)
            target_properties = ""
            if 'mappings_parsed' in row:
                target_properties = self._extract_properties_from_mappings(
                    row['mappings_parsed'], 
                    extract_target=True  # True = extract target properties (system_a)
                )
            # Fallback to system_a_properties column if it exists (for backward compatibility)
            elif 'system_a_properties' in row and pd.notna(row['system_a_properties']):
                target_properties = str(row['system_a_properties'])
            
            # Create query embedding text for transparency
            query_embedding_text = self._create_embedding_text(
                name=target_name,
                background=target_bg,
                properties=target_properties
            )
            
            # Find sources
            candidates = self.find_source(
                target_name=target_name, 
                target_background=target_bg,
                target_properties=target_properties,
                top_k=top_k
            )
            
            # Extract rankings
            source_names = [c.name for c in candidates]
            source_scores = [c.similarity_score for c in candidates]
            
            # Get all golden sources for this target (including variations)
            all_golden_sources = self._get_all_golden_sources(df, target_name)
            
            # Find all golden sources in top-k results and their ranks
            gold_info = self._find_gold_ranks(source_names, all_golden_sources)
            
            results.append({
                'id': row['id'],
                'target': target_name,
                'gold_source': primary_gold_source,
                'all_golden_sources': all_golden_sources,
                'predicted_rank_1': source_names[0] if len(source_names) > 0 else None,
                'gold_rank': gold_info['best_rank'],
                'num_golden_found': gold_info['num_golden_found'],
                'found_golden_sources': gold_info['found_golden_sources'],
                'golden_ranks': gold_info['golden_ranks'],
                'top_k_sources': source_names,
                'top_k_scores': source_scores,
                'embedding_mode': self.embedding_mode,
                'query_embedding_text': query_embedding_text,
                'target_background': target_bg,
                'target_properties': target_properties,
            })
            
            if (idx + 1) % 10 == 0:
                print(f"Processed {idx + 1}/{len(df)} examples")
        
        return pd.DataFrame(results)
    
    def get_corpus_embedding_texts(self) -> pd.DataFrame:
        """
        Get the embedding texts used for the corpus sources
        
        Returns:
            DataFrame with source information and their embedding texts
        """
        if self.corpus_df is None:
            raise ValueError("Corpus not loaded. Call load_corpus_from_csv first")
        
        return self.corpus_df[['source_name', 'source_domain', 'source_background', 
                              'source_properties', 'embedding_text']].copy()


if __name__ == "__main__":
    # Example usage
    print("RAG Source Finder")
    print("=" * 50)
    
    # Example with different embedding modes
    modes = ["name_only", "name_background", "name_properties", "name_properties_background"]
    
    for mode in modes:
        print(f"\n--- Testing mode: {mode} ---")
        
        # Initialize with specific mode
        finder = RAGSourceFinder(embedding_mode=mode)
        
        # Load and embed corpus
        finder.load_corpus_from_csv("../../data/SCAR_cleaned_manually.csv")
        finder.embed_corpus()
        
        # Example query
        results = finder.find_source(
            target_name="biological clock",
            target_background="The biological clock is a fundamental aspect of human physiology that regulates sleep-wake cycles.",
            target_properties="circadian rhythm, temporal regulation, physiological timing",
            top_k=3
        )
        
        print(f"Top 3 sources for mode '{mode}':")
        for result in results:
            print(f"  {result.rank}. {result.name} (score: {result.similarity_score:.4f})")

