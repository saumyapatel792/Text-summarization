import pytest
from src.config import DEFAULT_MODEL_NAME, TEST_SAMPLE_SIZE
from src.utils import load_data_sample, calculate_rouge_scores
from transformers import pipeline, AutoTokenizer, AutoModelForSeq2SeqLM

def test_rouge_regression():
    """
    Regression test to ensure summarization model does not drop in quality.
    Evaluates the model against a sample of CNN/DailyMail and checks ROUGE-L.
    """
    print(f"\nLoading sample data (size={TEST_SAMPLE_SIZE})...")
    df = load_data_sample(split="validation", sample_size=TEST_SAMPLE_SIZE)
    articles = df['article'].tolist()
    references = df['highlights'].tolist()
    
    assert len(articles) > 0, "Failed to load sample dataset articles"
    
    print(f"Loading default model {DEFAULT_MODEL_NAME} on CPU...")
    tokenizer = AutoTokenizer.from_pretrained(DEFAULT_MODEL_NAME)
    model = AutoModelForSeq2SeqLM.from_pretrained(DEFAULT_MODEL_NAME)
    summarizer = pipeline("summarization", model=model, tokenizer=tokenizer, device=-1)
    
    print("Generating summaries for testing...")
    predictions = []
    for article in articles:
        truncated_article = article[:4000]
        try:
            summary = summarizer(
                truncated_article,
                max_length=142,
                min_length=30,
                num_beams=4,
                length_penalty=2.0,
                do_sample=False
            )[0]['summary_text']
        except Exception as e:
            print(f"Error during summarization of an article: {e}")
            summary = ""
        predictions.append(summary)
        
    scores = calculate_rouge_scores(predictions, references)
    print(f"Calculated ROUGE scores: {scores}")
    
    # Threshold for ROUGE-L
    threshold = 0.12
    assert scores['rougeL'] >= threshold, f"ROUGE-L F1 score of {scores['rougeL']:.4f} is below the regression threshold of {threshold}."
    print("ROUGE Regression Test Passed successfully!")
