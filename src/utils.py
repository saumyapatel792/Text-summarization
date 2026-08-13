from datasets import load_dataset
from rouge_score import rouge_scorer
import pandas as pd
from typing import List, Dict
import logging
from src.config import DATASET_NAME, DATASET_VERSION

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def load_data_sample(split: str = "validation", sample_size: int = 10) -> pd.DataFrame:
    """
    Loads a small sample from the CNN/DailyMail dataset.
    """
    logger.info(f"Loading {split} split of {DATASET_NAME} (config: {DATASET_VERSION})...")
    # Load dataset. In datasets library, cnn_dailymail has 3.0.0 as config name.
    # We can try loading "cnn_dailymail" with config "3.0.0", which is the standard canonical format on HF.
    try:
        dataset = load_dataset(DATASET_NAME, DATASET_VERSION, split=split)
    except Exception as e:
        logger.warning(f"Failed to load {DATASET_NAME} with version {DATASET_VERSION}, trying config name directly: {e}")
        dataset = load_dataset("cnn_dailymail", "3.0.0", split=split)
        
    # Shuffling and selecting a deterministic sample
    shuffled_dataset = dataset.shuffle(seed=42)
    sample = shuffled_dataset.select(range(min(sample_size, len(dataset))))
    
    df = pd.DataFrame(sample)
    # The dataset has 'article' and 'highlights' columns
    logger.info(f"Loaded {len(df)} samples from {split} split.")
    return df

def calculate_rouge_scores(predictions: List[str], references: List[str]) -> Dict[str, float]:
    """
    Calculates ROUGE-1, ROUGE-2, and ROUGE-L F1 scores for predictions and references.
    """
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
    
    total_scores = {'rouge1': 0.0, 'rouge2': 0.0, 'rougeL': 0.0}
    
    n = len(predictions)
    if n == 0:
        return total_scores
        
    for pred, ref in zip(predictions, references):
        scores = scorer.score(ref, pred)
        total_scores['rouge1'] += scores['rouge1'].fmeasure
        total_scores['rouge2'] += scores['rouge2'].fmeasure
        total_scores['rougeL'] += scores['rougeL'].fmeasure
        
    # Calculate average
    avg_scores = {k: v / n for k, v in total_scores.items()}
    return avg_scores
