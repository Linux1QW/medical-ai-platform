"""
Configuration for RAG evaluation system.
"""
from typing import Optional
from pathlib import Path
from pydantic import BaseModel


class EvalConfig(BaseModel):
    """Configuration for evaluation runs."""
    
    # Dataset paths
    cases_path: Path = Path("backend/evaluation/rag_cases/rag_gold_cases.jsonl")
    output_dir: Path = Path("backend/evaluation/reports")
    
    # Evaluation settings
    mode: str = "tooluse"  # legacy, tooluse, both, mock
    split: str = "dev"
    limit: Optional[int] = None
    
    # Thresholds
    fail_on_threshold: bool = False
    
    # Metrics settings
    top_k: int = 5
    recall_k: int = 5
    ndcg_k: int = 5
    
    class Config:
        arbitrary_types_allowed = True


DEFAULT_CONFIG = EvalConfig()