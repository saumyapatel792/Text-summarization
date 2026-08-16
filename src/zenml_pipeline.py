import os
import time
import logging
import pandas as pd
import torch
import mlflow
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline
from zenml import step, pipeline as zenml_pipeline

from src.config import (
    MLFLOW_TRACKING_URI,
    MLFLOW_EXPERIMENT_NAME,
    MODEL_REGISTRY_NAME,
    DEFAULT_MODEL_NAME,
    CANDIDATE_MODELS
)
from src.utils import load_data_sample, calculate_rouge_scores

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# --- STEP 1: Ingest ---
@step
def data_ingest_step() -> pd.DataFrame:
    """
    Ingests validation data from CNN/DailyMail dataset.
    """
    logger.info("Starting Data Ingestion Step...")
    # Load a small sample size of 5 for demonstration purposes
    df = load_data_sample(split="validation", sample_size=5)
    return df

# --- STEP 2: Validate ---
@step
def data_validate_step(df: pd.DataFrame) -> pd.DataFrame:
    """
    Validates that the dataset schema is correct, no null values exist,
    and columns are present. Halts pipeline on failure.
    """
    logger.info("Starting Data Validation Step...")
    
    # 1. Check if DataFrame is empty
    if df.empty:
        raise ValueError("Validation Failed: The dataset is empty!")
        
    # 2. Check schema (column existence)
    required_columns = ["article", "highlights"]
    for col in required_columns:
        if col not in df.columns:
            raise ValueError(f"Validation Failed: Missing required column '{col}'!")
            
    # 3. Check for null values in required columns
    null_counts = df[required_columns].isnull().sum()
    for col, count in null_counts.items():
        if count > 0:
            raise ValueError(f"Validation Failed: Column '{col}' contains {count} null values!")
            
    logger.info("Data Validation Passed: Schema is correct and no null values found.")
    return df

# --- STEP 3: Transform ---
@step
def data_transform_step(df: pd.DataFrame) -> pd.DataFrame:
    """
    Transforms/prepares the text by truncating long inputs to avoid model constraints.
    """
    logger.info("Starting Data Transformation Step...")
    df_transformed = df.copy()
    
    # Basic text cleaning: strip trailing/leading spaces and truncate to 4000 characters
    df_transformed["article"] = df_transformed["article"].apply(lambda x: str(x).strip()[:4000])
    df_transformed["highlights"] = df_transformed["highlights"].apply(lambda x: str(x).strip())
    
    logger.info("Data Transformation complete.")
    return df_transformed

# --- STEP 4: Train ---
@step
def model_train_step(df: pd.DataFrame) -> str:
    """
    Simulates training/benchmarking of candidate models on validation samples.
    Logs metrics and parameters to MLflow. Returns the best model name.
    """
    logger.info("Starting Model Training/Benchmarking Step...")
    
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(MLFLOW_EXPERIMENT_NAME)
    
    articles = df["article"].tolist()
    references = df["highlights"].tolist()
    
    device = 0 if torch.cuda.is_available() else -1
    best_model_name = DEFAULT_MODEL_NAME
    best_rouge_l = -1.0
    
    # Benchmark the candidate models
    for model_name in CANDIDATE_MODELS[:2]:  # Limit to first two for speed
        logger.info(f"Benchmarking model: {model_name}")
        
        with mlflow.start_run(run_name=f"zenml_benchmark_{model_name.replace('/', '_')}"):
            mlflow.log_param("model_name", model_name)
            mlflow.log_param("dataset_size", len(articles))
            
            try:
                tokenizer = AutoTokenizer.from_pretrained(model_name)
                model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
                summarizer = pipeline(
                    "summarization",
                    model=model,
                    tokenizer=tokenizer,
                    device=device
                )
                
                start_time = time.time()
                predictions = []
                for article in articles:
                    summary = summarizer(
                        article,
                        max_length=142,
                        min_length=30,
                        do_sample=False
                    )[0]['summary_text']
                    predictions.append(summary)
                    
                elapsed_time = time.time() - start_time
                avg_latency = elapsed_time / len(articles)
                
                scores = calculate_rouge_scores(predictions, references)
                rouge_l = scores["rougeL"]
                
                mlflow.log_metric("avg_latency_sec", avg_latency)
                mlflow.log_metric("rouge1", scores["rouge1"])
                mlflow.log_metric("rouge2", scores["rouge2"])
                mlflow.log_metric("rougeL", rouge_l)
                
                logger.info(f"Model: {model_name} | ROUGE-L: {rouge_l:.4f}")
                
                if rouge_l > best_rouge_l:
                    best_rouge_l = rouge_l
                    best_model_name = model_name
                    
            except Exception as e:
                logger.error(f"Error benchmarking {model_name}: {e}")
                mlflow.log_param("status", "failed")
                
    logger.info(f"Training/benchmarking complete. Best model: {best_model_name}")
    return best_model_name

# --- STEP 5: Evaluate ---
@step
def model_evaluate_step(df: pd.DataFrame, best_model_name: str) -> dict:
    """
    Performs a final evaluation on the best model, outputting a dictionary of scores.
    """
    logger.info(f"Starting Final Model Evaluation Step for model: {best_model_name}")
    
    articles = df["article"].tolist()
    references = df["highlights"].tolist()
    device = 0 if torch.cuda.is_available() else -1
    
    tokenizer = AutoTokenizer.from_pretrained(best_model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(best_model_name)
    summarizer = pipeline(
        "summarization",
        model=model,
        tokenizer=tokenizer,
        device=device
    )
    
    predictions = []
    for article in articles:
        summary = summarizer(
            article,
            max_length=142,
            min_length=30,
            do_sample=False
        )[0]['summary_text']
        predictions.append(summary)
        
    scores = calculate_rouge_scores(predictions, references)
    logger.info(f"Final Evaluation Scores: {scores}")
    return scores

# --- STEP 6: Deploy / Register ---
@step
def model_deploy_step(best_model_name: str) -> None:
    """
    Registers the best model in the MLflow Model Registry.
    """
    logger.info(f"Starting Model Registry/Deployment Step for model: {best_model_name}")
    
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(MLFLOW_EXPERIMENT_NAME)
    
    tokenizer = AutoTokenizer.from_pretrained(best_model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(best_model_name)
    
    with mlflow.start_run(run_name="zenml_deploy") as run:
        components = {
            "model": model,
            "tokenizer": tokenizer,
        }
        # Log and register best model in model registry
        mlflow.transformers.log_model(
            transformers_model=components,
            artifact_path="best_model",
            task="summarization",
            registered_model_name=MODEL_REGISTRY_NAME,
            model_config={
                "num_beams": 4,
                "length_penalty": 2.0,
                "max_length": 142,
                "min_length": 30,
                "do_sample": False
            }
        )
    logger.info(f"Successfully registered model '{best_model_name}' under registry name '{MODEL_REGISTRY_NAME}'")


# --- ZENML PIPELINE DEFINITION ---
@zenml_pipeline
def text_summarization_pipeline():
    """
    Full ZenML orchestration pipeline:
    ingest -> validate -> transform -> train -> evaluate -> deploy
    """
    data = data_ingest_step()
    validated_data = data_validate_step(df=data)
    transformed_data = data_transform_step(df=validated_data)
    best_model = model_train_step(df=transformed_data)
    evaluation_metrics = model_evaluate_step(df=transformed_data, best_model_name=best_model)
    model_deploy_step(best_model_name=best_model)


if __name__ == "__main__":
    # Ensure ZenML uses local tracking
    os.environ["ZENML_DEBUG"] = "true"
    os.environ["PYTHONIOENCODING"] = "utf-8"
    
    # Run the pipeline
    text_summarization_pipeline()
