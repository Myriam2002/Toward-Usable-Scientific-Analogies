"""
SCAR Evaluator
Compares generated analogies against SCAR golden standard
"""

import pandas as pd
from typing import Dict, Any, Optional, List
from pathlib import Path


class SCAREvaluator:
    """
    Evaluator for comparing against SCAR golden standard
    """
    
    def __init__(self, scar_data_path: str):
        """
        Initialize SCAR evaluator
        
        Args:
            scar_data_path: Path to SCAR_cleaned_manually.csv
        """
        self.scar_df = pd.read_csv(scar_data_path)
        
        # Parse mappings and explanations if needed
        import ast
        if 'mappings_list' not in self.scar_df.columns:
            self.scar_df['mappings_list'] = self.scar_df['mappings_parsed'].apply(
                lambda x: ast.literal_eval(x) if pd.notna(x) and x else []
            )
        if 'explanation_list' not in self.scar_df.columns:
            self.scar_df['explanation_list'] = self.scar_df['explanation_parsed'].apply(
                lambda x: ast.literal_eval(x) if pd.notna(x) and x else []
            )
    
    def find_golden_record(
        self,
        target_name: str
    ) -> Optional[pd.Series]:
        """
        Find golden record for a target concept
        
        Args:
            target_name: Name of target concept
            
        Returns:
            pandas Series with golden record, or None if not found
        """
        # Normalize target name for matching
        target_normalized = target_name.lower().strip()
        
        # Try exact match first
        matches = self.scar_df[
            self.scar_df['system_a'].str.lower().str.strip() == target_normalized
        ]
        
        if len(matches) > 0:
            return matches.iloc[0]
        
        # Try partial match
        matches = self.scar_df[
            self.scar_df['system_a'].str.lower().str.contains(target_normalized, na=False)
        ]
        
        if len(matches) > 0:
            return matches.iloc[0]
        
        return None
    
    def evaluate(
        self,
        target_name: str,
        generated_source: Optional[str] = None,
        generated_mappings: Optional[List[List[str]]] = None,
        generated_explanation: Optional[str] = None,
        use_llm_judge: bool = True,
        llm_judge: Optional[Any] = None,
        threshold: float = 0.7
    ) -> Dict[str, Any]:
        """
        Evaluate generated analogy against SCAR golden standard
        
        Args:
            target_name: Name of target concept
            generated_source: Generated source concept name
            generated_mappings: Generated property mappings
            generated_explanation: Generated explanation
            use_llm_judge: Whether to use LLM judge if no match
            llm_judge: Optional LLM judge instance
            threshold: Threshold for LLM judge score
            
        Returns:
            Dictionary with evaluation results
        """
        # Find golden record
        golden_record = self.find_golden_record(target_name)
        
        if golden_record is None:
            return {
                'found_in_scar': False,
                'source_match': False,
                'mapping_match': False,
                'explanation_match': False,
                'overall_match': False,
                'llm_judge_score': None,
                'llm_judge_passed': None,
                'error': 'Target not found in SCAR dataset'
            }
        
        # Check source match
        golden_source = golden_record['system_b']
        source_match = self._check_source_match(generated_source, golden_source)
        
        # Check mapping match
        golden_mappings = golden_record.get('mappings_list', [])
        mapping_match = False
        mapping_score = 0.0
        if generated_mappings and golden_mappings:
            mapping_match, mapping_score = self._check_mapping_match(
                generated_mappings,
                golden_mappings
            )
        
        # Check explanation match (if available)
        golden_explanation = golden_record.get('explanation_list', [])
        explanation_match = False
        explanation_score = 0.0
        if generated_explanation and golden_explanation:
            explanation_match, explanation_score = self._check_explanation_match(
                generated_explanation,
                golden_explanation
            )
        
        # Overall match: source must match
        overall_match = source_match
        
        # If source doesn't match, use LLM judge if available
        llm_judge_score = None
        llm_judge_passed = None
        
        if not source_match and use_llm_judge and llm_judge:
            llm_judge_score = llm_judge.evaluate(
                target_name=target_name,
                generated_source=generated_source,
                golden_source=golden_source,
                generated_mappings=generated_mappings,
                golden_mappings=golden_mappings
            )
            llm_judge_passed = llm_judge_score >= threshold
        
        return {
            'found_in_scar': True,
            'golden_source': golden_source,
            'generated_source': generated_source,
            'source_match': source_match,
            'mapping_match': mapping_match,
            'mapping_score': mapping_score,
            'explanation_match': explanation_match,
            'explanation_score': explanation_score,
            'overall_match': overall_match,
            'llm_judge_score': llm_judge_score,
            'llm_judge_passed': llm_judge_passed,
            'golden_record_id': golden_record.get('id', None)
        }
    
    def _check_source_match(
        self,
        generated_source: Optional[str],
        golden_source: str
    ) -> bool:
        """Check if generated source matches golden source"""
        if not generated_source:
            return False
        
        # Normalize for comparison
        gen_norm = generated_source.lower().strip()
        gold_norm = golden_source.lower().strip()
        
        # Exact match
        if gen_norm == gold_norm:
            return True
        
        # Partial match (one contains the other)
        if gen_norm in gold_norm or gold_norm in gen_norm:
            return True
        
        return False
    
    def _check_mapping_match(
        self,
        generated_mappings: List[List[str]],
        golden_mappings: List[List[str]]
    ) -> tuple[bool, float]:
        """
        Check if generated mappings match golden mappings
        
        Returns:
            (match: bool, score: float)
        """
        if not generated_mappings or not golden_mappings:
            return False, 0.0
        
        # Normalize mappings
        def normalize_mapping(m):
            return [str(x).lower().strip() for x in m]
        
        gen_normalized = [normalize_mapping(m) for m in generated_mappings]
        gold_normalized = [normalize_mapping(m) for m in golden_mappings]
        
        # Count matches
        matches = 0
        used_gold = set()
        
        for gen_map in gen_normalized:
            for i, gold_map in enumerate(gold_normalized):
                if i in used_gold:
                    continue
                # Check if mappings match (order-independent)
                if (gen_map[0] == gold_map[0] and gen_map[1] == gold_map[1]) or \
                   (gen_map[0] == gold_map[1] and gen_map[1] == gold_map[0]):
                    matches += 1
                    used_gold.add(i)
                    break
        
        score = matches / len(golden_mappings) if golden_mappings else 0.0
        match = score >= 0.8  # 80% of mappings must match
        
        return match, score
    
    def _check_explanation_match(
        self,
        generated_explanation: str,
        golden_explanation: List[str]
    ) -> tuple[bool, float]:
        """
        Check if generated explanation matches golden explanation
        
        Returns:
            (match: bool, score: float)
        """
        # Use semantic similarity (simple word overlap for now)
        # In production, could use SBERT like in stage1
        if isinstance(generated_explanation, list):
            gen_text = " ".join(generated_explanation)
        else:
            gen_text = str(generated_explanation)
        
        gold_text = " ".join(golden_explanation)
        
        # Simple word overlap score
        gen_words = set(gen_text.lower().split())
        gold_words = set(gold_text.lower().split())
        
        if not gold_words:
            return False, 0.0
        
        overlap = len(gen_words & gold_words)
        score = overlap / len(gold_words)
        
        match = score >= 0.5  # 50% word overlap
        
        return match, score

