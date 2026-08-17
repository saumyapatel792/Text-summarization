import time
import mlflow
import torch
from transformers import pipeline, AutoModelForSeq2SeqLM, AutoTokenizer
from src.config import (
    MLFLOW_TRACKING_URI,
    MLFLOW_EXPERIMENT_NAME,
    CANDIDATE_MODELS,
    BENCHMARK_SAMPLE_SIZE
)
from src.utils import load_data_sample, calculate_rouge_scores, SystemMonitor
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def benchmark_models():
    # Set up MLflow
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(MLFLOW_EXPERIMENT_NAME)
    
    # Load dataset sample
    df = load_data_sample(split="validation", sample_size=BENCHMARK_SAMPLE_SIZE)
    articles = df['article'].tolist()
    references = df['highlights'].tolist()
    
    device = 0 if torch.cuda.is_available() else -1
    logger.info(f"Using device: {'GPU' if device == 0 else 'CPU'}")
    
    best_model_name = None
    best_rouge_l = -1.0
    
    for model_name in CANDIDATE_MODELS:
        logger.info(f"Benchmarking model: {model_name}")
        
        # Start MLflow run
        with mlflow.start_run(run_name=f"benchmark_{model_name.replace('/', '_')}"):
            # Log params
            mlflow.log_param("model_name", model_name)
            mlflow.log_param("dataset_size", len(articles))
            mlflow.log_param("device", "cuda" if device == 0 else "cpu")
            
            # Start resource monitoring
            monitor = SystemMonitor()
            monitor.start()
            
            try:
                # Load tokenizer and model
                tokenizer = AutoTokenizer.from_pretrained(model_name)
                model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
                monitor.sample() # Sample after loading model in memory
                
                # Setup Pipeline
                summarizer = pipeline(
                    "summarization",
                    model=model,
                    tokenizer=tokenizer,
                    device=device
                )
                monitor.sample() # Sample after pipeline initialization
                
                # Perform inference and time it
                start_time = time.time()
                predictions = []
                for article in articles:
                    # Truncate article to avoid overflow (most models handle max 1024 or 512 tokens)
                    truncated_article = article[:4000] # simple character truncation to avoid huge inputs
                    try:
                        summary = summarizer(
                            truncated_article,
                            max_length=142,
                            min_length=30,
                            do_sample=False
                        )[0]['summary_text']
                    except Exception as e:
                        logger.error(f"Inference failed for article: {e}")
                        summary = ""
                    predictions.append(summary)
                    monitor.sample() # Sample after each inference iteration
                
                elapsed_time = time.time() - start_time
                avg_latency = elapsed_time / len(articles)
                
                # Stop monitoring and collect metrics
                sys_metrics = monitor.stop()
                
                # Compute ROUGE scores
                scores = calculate_rouge_scores(predictions, references)
                
                # Log metrics
                mlflow.log_metric("avg_latency_sec", avg_latency)
                mlflow.log_metric("rouge1", scores['rouge1'])
                mlflow.log_metric("rouge2", scores['rouge2'])
                mlflow.log_metric("rougeL", scores['rougeL'])
                
                # Log system metrics
                mlflow.log_metrics(sys_metrics)

                
                # Log Model artifact using mlflow.transformers
                components = {
                    "model": model,
                    "tokenizer": tokenizer,
                }
                mlflow.transformers.log_model(
                    transformers_model=components,
                    artifact_path="model",
                    task="summarization"
                )
                
                logger.info(f"Model: {model_name} | ROUGE-L: {scores['rougeL']:.4f} | Avg Latency: {avg_latency:.2f}s")
                
                if scores['rougeL'] > best_rouge_l:
                    best_rouge_l = scores['rougeL']
                    best_model_name = model_name
                    
            except Exception as e:
                logger.error(f"Failed to benchmark model {model_name}: {e}")
                mlflow.log_param("status", "failed")
                mlflow.log_param("error_message", str(e))
                
    logger.info(f"Benchmarking complete. Best model is {best_model_name} with ROUGE-L of {best_rouge_l:.4f}")
    return best_model_name

if __name__ == "__main__":
    benchmark_models()
