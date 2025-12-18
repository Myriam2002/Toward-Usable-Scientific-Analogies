"""
Iterative LLM-Based Source Finder
Implements tournament-style and sequential refinement approaches
"""

import os
import sys
import pandas as pd
import numpy as np
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass, asdict
import json
import time
import math

# Add parent directory to path to import easy_llm_importer
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'mapping_generation'))
from easy_llm_importer import LLMClient


@dataclass
class RoundResult:
    """Result from a single tournament round or refinement step"""
    round_number: int
    batch_number: int
    candidates: List[str]
    winner: str
    reasoning: str
    timestamp: float


@dataclass
class IterativeResult:
    """Final result from iterative source finding"""
    target: str
    predicted_source: str
    method: str  # 'tournament' or 'sequential'
    rounds: List[RoundResult]
    total_llm_calls: int
    total_time: float


class TournamentSourceFinder:
    """
    Tournament-style elimination for source finding
    """
    
    def __init__(self, llm_client: LLMClient, model_name: str, batch_size: int = 6):
        """
        Initialize tournament finder
        
        Args:
            llm_client: Unified LLM client
            model_name: Model to use for evaluation
            batch_size: Number of sources per batch
        """
        self.client = llm_client
        self.model_name = model_name
        self.batch_size = batch_size
        
    def find_source(
        self, 
        target_name: str, 
        target_background: str,
        source_candidates: List[Dict[str, str]]
    ) -> IterativeResult:
        """
        Find best source using tournament elimination
        
        Args:
            target_name: Name of target concept
            target_background: Background description
            source_candidates: List of dicts with 'name', 'description', 'domain'
            
        Returns:
            IterativeResult with winner and reasoning
        """
        start_time = time.time()
        rounds = []
        current_candidates = source_candidates.copy()
        round_num = 1
        total_calls = 0
        
        print(f"\nTournament for target: {target_name}")
        print(f"Starting with {len(current_candidates)} candidates")
        
        while len(current_candidates) > 1:
            print(f"\n=== Round {round_num} ===")
            print(f"Candidates: {len(current_candidates)}")
            
            # Split into batches
            batches = self._create_batches(current_candidates)
            round_winners = []
            
            for batch_num, batch in enumerate(batches, 1):
                print(f"  Batch {batch_num}/{len(batches)}: {len(batch)} sources")
                
                # Evaluate batch
                winner, reasoning = self._evaluate_batch(
                    target_name, 
                    target_background, 
                    batch
                )
                total_calls += 1
                
                round_winners.append(winner)
                
                # Record round result
                rounds.append(RoundResult(
                    round_number=round_num,
                    batch_number=batch_num,
                    candidates=[c['name'] for c in batch],
                    winner=winner['name'],
                    reasoning=reasoning,
                    timestamp=time.time()
                ))
                
                print(f"    Winner: {winner['name']}")
            
            # Prepare for next round
            current_candidates = round_winners
            round_num += 1
        
        # Final winner
        final_winner = current_candidates[0]
        total_time = time.time() - start_time
        
        print(f"\n{'='*50}")
        print(f"Final Winner: {final_winner['name']}")
        print(f"Total rounds: {round_num - 1}")
        print(f"Total LLM calls: {total_calls}")
        print(f"Total time: {total_time:.2f}s")
        
        return IterativeResult(
            target=target_name,
            predicted_source=final_winner['name'],
            method='tournament',
            rounds=rounds,
            total_llm_calls=total_calls,
            total_time=total_time
        )
    
    def _create_batches(self, candidates: List[Dict]) -> List[List[Dict]]:
        """Split candidates into batches"""
        batches = []
        for i in range(0, len(candidates), self.batch_size):
            batches.append(candidates[i:i + self.batch_size])
        return batches
    
    def _evaluate_batch(
        self, 
        target_name: str, 
        target_background: str,
        batch: List[Dict[str, str]]
    ) -> Tuple[Dict[str, str], str]:
        """
        Evaluate a batch and select the best source
        
        Returns:
            (winner_dict, reasoning)
        """
        # Build prompt
        prompt = self._build_tournament_prompt(target_name, target_background, batch)
        
        # Call LLM
        messages = [{"role": "user", "content": prompt}]
        response = self.client.chat(
            model_name=self.model_name,
            messages=messages,
            temperature=0.2,
            max_tokens=1000
        )
        
        # Parse response
        winner, reasoning = self._parse_tournament_response(response, batch)
        
        return winner, reasoning
    
    def _build_tournament_prompt(
        self, 
        target_name: str, 
        target_background: str,
        batch: List[Dict[str, str]]
    ) -> str:
        """Build prompt for tournament evaluation"""
        prompt = f"""You are helping find the best source analogy for a target concept.

TARGET CONCEPT: {target_name}
TARGET DESCRIPTION: {target_background}

Below are {len(batch)} potential SOURCE concepts. Your task is to select the ONE source that would make the BEST analogy for the target.

SOURCES:
"""
        for i, source in enumerate(batch, 1):
            prompt += f"\n{i}. {source['name']} ({source['domain']})\n"
            prompt += f"   Description: {source['description']}\n"
        
        prompt += """
Consider:
- Structural similarity (shared relational patterns)
- Conceptual alignment (similar underlying principles)
- Potential for meaningful mapping

Respond in this format:
WINNER: [source name]
REASONING: [2-3 sentences explaining why this source is the best match]
"""
        return prompt
    
    def _parse_tournament_response(
        self, 
        response: str, 
        batch: List[Dict[str, str]]
    ) -> Tuple[Dict[str, str], str]:
        """Parse LLM response to extract winner and reasoning"""
        lines = response.strip().split('\n')
        
        winner_name = None
        reasoning = ""
        
        for line in lines:
            if line.startswith("WINNER:"):
                winner_name = line.replace("WINNER:", "").strip()
            elif line.startswith("REASONING:"):
                reasoning = line.replace("REASONING:", "").strip()
            elif reasoning and line.strip():
                reasoning += " " + line.strip()
        
        # Find winner in batch
        winner = None
        for source in batch:
            if source['name'].lower() in winner_name.lower():
                winner = source
                break
        
        # Fallback to first if parsing fails
        if winner is None:
            print(f"    Warning: Could not parse winner '{winner_name}', using first source")
            winner = batch[0]
        
        return winner, reasoning


class SequentialSourceFinder:
    """
    Sequential refinement approach for source finding
    """
    
    def __init__(
        self, 
        llm_client: LLMClient, 
        model_name: str, 
        initial_shortlist_size: int = 12,
        chunk_size: int = 8
    ):
        """
        Initialize sequential finder
        
        Args:
            llm_client: Unified LLM client
            model_name: Model to use
            initial_shortlist_size: Initial shortlist size
            chunk_size: Sources to show per refinement step
        """
        self.client = llm_client
        self.model_name = model_name
        self.initial_shortlist_size = initial_shortlist_size
        self.chunk_size = chunk_size
    
    def find_source(
        self, 
        target_name: str, 
        target_background: str,
        source_candidates: List[Dict[str, str]]
    ) -> IterativeResult:
        """
        Find best source using sequential refinement
        
        Args:
            target_name: Name of target concept
            target_background: Background description
            source_candidates: List of source dicts
            
        Returns:
            IterativeResult with final choice
        """
        start_time = time.time()
        rounds = []
        total_calls = 0
        
        print(f"\nSequential Refinement for target: {target_name}")
        print(f"Total candidates: {len(source_candidates)}")
        
        # Initialize shortlist (empty to start)
        shortlist = []
        remaining = source_candidates.copy()
        round_num = 1
        
        # Process chunks until all candidates reviewed
        while remaining:
            chunk = remaining[:self.chunk_size]
            remaining = remaining[self.chunk_size:]
            
            print(f"\n=== Round {round_num} ===")
            print(f"Reviewing {len(chunk)} new sources")
            print(f"Current shortlist size: {len(shortlist)}")
            print(f"Remaining to review: {len(remaining)}")
            
            # Get refinement
            new_shortlist, reasoning = self._refine_shortlist(
                target_name,
                target_background,
                current_shortlist=shortlist,
                new_candidates=chunk,
                max_shortlist=self.initial_shortlist_size
            )
            total_calls += 1
            
            # Record
            rounds.append(RoundResult(
                round_number=round_num,
                batch_number=1,
                candidates=[c['name'] for c in chunk],
                winner=f"Shortlist: {[s['name'] for s in new_shortlist]}",
                reasoning=reasoning,
                timestamp=time.time()
            ))
            
            shortlist = new_shortlist
            round_num += 1
        
        # Final narrowing to single source
        print(f"\n=== Final Selection ===")
        print(f"Narrowing from {len(shortlist)} to 1")
        
        if len(shortlist) > 1:
            final_winner, final_reasoning = self._select_final(
                target_name, 
                target_background, 
                shortlist
            )
            total_calls += 1
            
            rounds.append(RoundResult(
                round_number=round_num,
                batch_number=1,
                candidates=[s['name'] for s in shortlist],
                winner=final_winner['name'],
                reasoning=final_reasoning,
                timestamp=time.time()
            ))
        else:
            final_winner = shortlist[0]
        
        total_time = time.time() - start_time
        
        print(f"\n{'='*50}")
        print(f"Final Winner: {final_winner['name']}")
        print(f"Total rounds: {round_num}")
        print(f"Total LLM calls: {total_calls}")
        print(f"Total time: {total_time:.2f}s")
        
        return IterativeResult(
            target=target_name,
            predicted_source=final_winner['name'],
            method='sequential',
            rounds=rounds,
            total_llm_calls=total_calls,
            total_time=total_time
        )
    
    def _refine_shortlist(
        self,
        target_name: str,
        target_background: str,
        current_shortlist: List[Dict],
        new_candidates: List[Dict],
        max_shortlist: int
    ) -> Tuple[List[Dict], str]:
        """Refine shortlist with new candidates"""
        prompt = self._build_refinement_prompt(
            target_name,
            target_background,
            current_shortlist,
            new_candidates,
            max_shortlist
        )
        
        messages = [{"role": "user", "content": prompt}]
        response = self.client.chat(
            model_name=self.model_name,
            messages=messages,
            temperature=0.2,
            max_tokens=1500
        )
        
        new_shortlist, reasoning = self._parse_refinement_response(
            response,
            current_shortlist,
            new_candidates,
            max_shortlist
        )
        
        return new_shortlist, reasoning
    
    def _build_refinement_prompt(
        self,
        target_name: str,
        target_background: str,
        current_shortlist: List[Dict],
        new_candidates: List[Dict],
        max_shortlist: int
    ) -> str:
        """Build prompt for shortlist refinement"""
        prompt = f"""You are refining a shortlist of source analogies for a target concept.

TARGET: {target_name}
DESCRIPTION: {target_background}

"""
        if current_shortlist:
            prompt += f"CURRENT SHORTLIST ({len(current_shortlist)} sources):\n"
            for i, source in enumerate(current_shortlist, 1):
                prompt += f"{i}. {source['name']}\n"
            prompt += "\n"
        else:
            prompt += "CURRENT SHORTLIST: Empty (starting fresh)\n\n"
        
        prompt += f"NEW CANDIDATES ({len(new_candidates)} sources to consider):\n"
        for i, source in enumerate(new_candidates, 1):
            prompt += f"{i}. {source['name']} ({source['domain']})\n"
            prompt += f"   {source['description'][:150]}...\n"
        
        prompt += f"""
Task: Update the shortlist by adding the BEST new candidates and removing weaker ones if needed.
- Keep at most {max_shortlist} sources in the shortlist
- Prioritize sources with strongest structural and conceptual similarity to the target

Respond in this format:
UPDATED_SHORTLIST: [comma-separated list of source names]
REASONING: [Explain which sources you added/kept/removed and why]
"""
        return prompt
    
    def _parse_refinement_response(
        self,
        response: str,
        current_shortlist: List[Dict],
        new_candidates: List[Dict],
        max_shortlist: int
    ) -> Tuple[List[Dict], str]:
        """Parse refinement response"""
        lines = response.strip().split('\n')
        
        shortlist_line = ""
        reasoning = ""
        
        for line in lines:
            if line.startswith("UPDATED_SHORTLIST:"):
                shortlist_line = line.replace("UPDATED_SHORTLIST:", "").strip()
            elif line.startswith("REASONING:"):
                reasoning = line.replace("REASONING:", "").strip()
            elif reasoning and line.strip():
                reasoning += " " + line.strip()
        
        # Parse shortlist names
        if shortlist_line:
            shortlist_names = [name.strip() for name in shortlist_line.split(',')]
        else:
            shortlist_names = []
        
        # Build new shortlist from all available sources
        all_sources = current_shortlist + new_candidates
        new_shortlist = []
        
        for name in shortlist_names:
            for source in all_sources:
                if source['name'].lower() in name.lower() or name.lower() in source['name'].lower():
                    if source not in new_shortlist:  # Avoid duplicates
                        new_shortlist.append(source)
                    break
        
        # Fallback: if parsing fails, keep current + add best new
        if len(new_shortlist) == 0:
            print("    Warning: Could not parse shortlist, using fallback")
            new_shortlist = current_shortlist + new_candidates[:max(1, max_shortlist - len(current_shortlist))]
        
        # Limit to max size
        new_shortlist = new_shortlist[:max_shortlist]
        
        return new_shortlist, reasoning
    
    def _select_final(
        self,
        target_name: str,
        target_background: str,
        shortlist: List[Dict]
    ) -> Tuple[Dict, str]:
        """Select final winner from shortlist"""
        prompt = f"""Select the SINGLE BEST source analogy for this target.

TARGET: {target_name}
DESCRIPTION: {target_background}

FINAL CANDIDATES:
"""
        for i, source in enumerate(shortlist, 1):
            prompt += f"{i}. {source['name']} ({source['domain']})\n"
            prompt += f"   {source['description']}\n\n"
        
        prompt += """
FINAL_CHOICE: [source name]
REASONING: [Why this is the best match]
"""
        
        messages = [{"role": "user", "content": prompt}]
        response = self.client.chat(
            model_name=self.model_name,
            messages=messages,
            temperature=0.2,
            max_tokens=800
        )
        
        # Parse
        lines = response.strip().split('\n')
        choice_name = ""
        reasoning = ""
        
        for line in lines:
            if line.startswith("FINAL_CHOICE:"):
                choice_name = line.replace("FINAL_CHOICE:", "").strip()
            elif line.startswith("REASONING:"):
                reasoning = line.replace("REASONING:", "").strip()
            elif reasoning and line.strip():
                reasoning += " " + line.strip()
        
        # Find in shortlist
        winner = None
        for source in shortlist:
            if source['name'].lower() in choice_name.lower():
                winner = source
                break
        
        if winner is None:
            winner = shortlist[0]
        
        return winner, reasoning


if __name__ == "__main__":
    print("Iterative Source Finder")
    print("=" * 50)
    print("\nThis module requires LLM API keys and corpus data")
    print("Use via notebook or import into your code")

