import os
import time
import logging
import pandas as pd
from typing import Dict, Any, Tuple
import torch
import mlflow
import optuna
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline as hf_pipeline

from zenml import step, pipeline
from src.config import (
    MLFLOW_TRACKING_URI,
    MLFLOW_EXPERIMENT_NAME,
    MODEL_REGISTRY_NAME,
    DEFAULT_MODEL_NAME,
    DEFAULT_MIN_LENGTH,
    DEFAULT_MAX_LENGTH,
    BENCHMARK_SAMPLE_SIZE,
    TUNE_SAMPLE_SIZE
)
from src.utils import load_data_sample, calculate_rouge_scores, SystemMonitor

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@step(enable_cache=True)
def ingest_data(split: str = "validation", sample_size: int = 10) -> pd.DataFrame:
    """
    Ingest step: Loads a sample of the CNN/DailyMail dataset from HuggingFace.
    """
    logger.info(f"Ingesting dataset split: {split} (sample size: {sample_size})")
    df = load_data_sample(split=split, sample_size=sample_size)
    return df

@step(enable_cache=True)
def validate_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Validation step: Actively checks schema, null values, and class balance (document length representation).
    Raises ValueError to halt the pipeline on failure.
    """
    logger.info("Validating dataset schema, nulls, and length representations...")
    
    # 1. Schema Validation
    required_columns = ["article", "highlights"]
    for col in required_columns:
        if col not in df.columns:
            raise ValueError(f"Schema Validation Failed: Missing required column '{col}'.")
            
    # 2. Null/Empty Value Check
    null_counts = df[required_columns].isnull().sum()
    if null_counts.sum() > 0:
        raise ValueError(f"Null Validation Failed: Found null values in dataset: \n{null_counts}")
        
    empty_strings = ((df["article"].str.strip() == "") | (df["highlights"].str.strip() == "")).sum()
    if empty_strings > 0:
        raise ValueError(f"Empty Cell Validation Failed: Found {empty_strings} empty or whitespace-only text cells.")
        
    # 3. Class Balance / Text Length Bucket Representation Check
    # We define 3 classes of articles based on character length: Short, Medium, Long.
    # To pass, we want to make sure the dataset is not homogeneous (i.e. not all articles belong to a single length class),
    # ensuring length variety.
    def get_length_bucket(text: str) -> str:
        length = len(text)
        if length < 1000:
            return "Short"
        elif length < 3000:
            return "Medium"
        else:
            return "Long"
            
    df_copy = df.copy()
    df_copy["length_bucket"] = df_copy["article"].apply(get_length_bucket)
    bucket_counts = df_copy["length_bucket"].value_counts()
    logger.info(f"Article length class counts: \n{bucket_counts}")
    
    # Gated representation check (only run if sample size is sufficient, e.g. >= 3)
    if len(df) >= 3:
        for bucket, count in bucket_counts.items():
            proportion = count / len(df)
            if proportion >= 1.0:
                raise ValueError(
                    f"Class Balance Validation Failed: 100% of data is in class '{bucket}'. "
                    f"Pipeline halted to ensure representative length diversity."
                )
                
    logger.info("Data Validation Passed successfully.")
    return df

@step(enable_cache=True)
def transform_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Transformation step: Trims whitespaces and limits article size to prevent model token overflow.
    """
    logger.info("Transforming text inputs...")
    df_transformed = df.copy()
    
    # Strip whitespaces
    df_transformed["article"] = df_transformed["article"].str.strip()
    df_transformed["highlights"] = df_transformed["highlights"].str.strip()
    
    # Truncate text strictly to prevent token overflow on default models (max 4000 characters)
    df_transformed["article"] = df_transformed["article"].apply(lambda x: x[:4000])
    
    logger.info("Transformation complete.")
    return df_transformed

@step(enable_cache=False) # Disable cache for train to allow trial executions
def train_model(df: pd.DataFrame, model_name: str = DEFAULT_MODEL_NAME, n_trials: int = 2) -> Dict[str, Any]:
    """
    Train step: Runs hyperparameter tuning using Optuna to select the best num_beams and length_penalty.
    """
    logger.info(f"Tuning generation parameters for model: {model_name} (trials: {n_trials})")
    
    articles = df["article"].tolist()
    references = df["highlights"].tolist()
    
    device = 0 if torch.cuda.is_available() else -1
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    summarizer = hf_pipeline(
        "summarization",
        model=model,
        tokenizer=tokenizer,
        device=device
    )
    
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(MLFLOW_EXPERIMENT_NAME)
    
    def objective(trial):
        num_beams = trial.suggest_int("num_beams", 1, 4)
        length_penalty = trial.suggest_float("length_penalty", 0.5, 2.5)
        
        with mlflow.start_run(run_name=f"zenml_trial_{trial.number}", nested=True):
            mlflow.log_params({
                "num_beams": num_beams,
                "length_penalty": length_penalty,
                "trial_number": trial.number,
                "orchestrator": "zenml"
            })
            
            # Start resource monitoring
            monitor = SystemMonitor()
            monitor.start()
            
            predictions = []
            for article in articles:
                summary = summarizer(
                    article,
                    max_length=DEFAULT_MAX_LENGTH,
                    min_length=DEFAULT_MIN_LENGTH,
                    num_beams=num_beams,
                    length_penalty=length_penalty,
                    do_sample=False
                )[0]["summary_text"]
                predictions.append(summary)
                monitor.sample() # Sample after each inference iteration
                
            # Stop monitoring and collect metrics
            sys_metrics = monitor.stop()
            
            scores = calculate_rouge_scores(predictions, references)
            mlflow.log_metric("rougeL", scores["rougeL"])
            mlflow.log_metrics(sys_metrics)
            return scores["rougeL"]
            
    with mlflow.start_run(run_name=f"zenml_tuning_{model_name.replace('/', '_')}"):
        mlflow.log_params({
            "model_name": model_name,
            "tuning_samples": len(articles),
            "n_trials": n_trials,
            "orchestrator": "zenml"
        })
        
        study = optuna.create_study(direction="maximize")
        study.optimize(objective, n_trials=n_trials)
        
        logger.info(f"Best trial parameters: {study.best_params}")
        logger.info(f"Best ROUGE-L: {study.best_value:.4f}")
        
        mlflow.log_params({f"best_{k}": v for k, v in study.best_params.items()})
        mlflow.log_metric("best_rougeL", study.best_value)
        
        return {
            "model_name": model_name,
            "best_params": study.best_params,
            "best_rougeL": study.best_value
        }

@step(enable_cache=True)
def evaluate_model(train_result: Dict[str, Any], df: pd.DataFrame) -> Dict[str, float]:
    """
    Evaluation step: Evaluates the chosen model with the best parameters and computes metrics.
    """
    model_name = train_result["model_name"]
    best_params = train_result["best_params"]
    
    logger.info(f"Evaluating {model_name} with parameters {best_params}...")
    
    articles = df["article"].tolist()
    references = df["highlights"].tolist()
    
    # Start resource monitoring
    monitor = SystemMonitor()
    monitor.start()
    
    device = 0 if torch.cuda.is_available() else -1
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    monitor.sample() # Sample after loading model in memory
    
    summarizer = hf_pipeline(
        "summarization",
        model=model,
        tokenizer=tokenizer,
        device=device
    )
    monitor.sample() # Sample after pipeline initialization
    
    start_time = time.time()
    predictions = []
    for article in articles:
        summary = summarizer(
            article,
            max_length=DEFAULT_MAX_LENGTH,
            min_length=DEFAULT_MIN_LENGTH,
            num_beams=best_params.get("num_beams", 4),
            length_penalty=best_params.get("length_penalty", 1.0),
            do_sample=False
        )[0]["summary_text"]
        predictions.append(summary)
        monitor.sample() # Sample after each inference iteration
        
    elapsed_time = time.time() - start_time
    avg_latency = elapsed_time / len(articles)
    
    # Stop monitoring and collect metrics
    sys_metrics = monitor.stop()
    
    scores = calculate_rouge_scores(predictions, references)
    logger.info(f"Evaluation scores: {scores} | Avg Latency: {avg_latency:.2f}s")
    
    res = {
        "rouge1": scores["rouge1"],
        "rouge2": scores["rouge2"],
        "rougeL": scores["rougeL"],
        "avg_latency_sec": avg_latency
    }
    res.update(sys_metrics)
    return res

@step(enable_cache=False)
def deploy_model(train_result: Dict[str, Any], eval_metrics: Dict[str, float]) -> bool:
    """
    Deploy step: Registers the best model in the MLflow Model Registry if threshold is met.
    """
    model_name = train_result["model_name"]
    best_params = train_result["best_params"]
    rouge_l = eval_metrics["rougeL"]
    
    threshold = 0.12
    logger.info(f"Deploy check: ROUGE-L score is {rouge_l:.4f} (Threshold: {threshold})")
    
    if rouge_l >= threshold:
        logger.info("Performance threshold met. Registering model in MLflow registry...")
        
        mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
        mlflow.set_experiment(MLFLOW_EXPERIMENT_NAME)
        
        # Load and register model
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
        components = {
            "model": model,
            "tokenizer": tokenizer
        }
        
        with mlflow.start_run(run_name="zenml_deploy"):
            mlflow.log_params(best_params)
            mlflow.log_metrics(eval_metrics)
            
            mlflow.transformers.log_model(
                transformers_model=components,
                artifact_path="best_model",
                task="summarization",
                registered_model_name=MODEL_REGISTRY_NAME,
                model_config={
                    "num_beams": best_params.get("num_beams", 4),
                    "length_penalty": best_params.get("length_penalty", 1.0),
                    "max_length": DEFAULT_MAX_LENGTH,
                    "min_length": DEFAULT_MIN_LENGTH,
                    "do_sample": False
                }
            )
            
        logger.info(f"Model successfully registered in MLflow Model Registry as '{MODEL_REGISTRY_NAME}'")
        return True
    else:
        logger.warning("Performance threshold not met. Model deployment skipped.")
        return False

@pipeline
def text_summarization_pipeline():
    """
    ZenML Pipeline orchestrating the full ML lifecycle.
    """
    # 1. Ingest
    df_raw = ingest_data(sample_size=BENCHMARK_SAMPLE_SIZE)
    # 2. Validate
    df_valid = validate_data(df=df_raw)
    # 3. Transform
    df_trans = transform_data(df=df_valid)
    # 4. Train (Tuning)
    train_res = train_model(df=df_trans)
    # 5. Evaluate
    eval_metrics = evaluate_model(train_result=train_res, df=df_trans)
    # 6. Deploy
    deploy_model(train_result=train_res, eval_metrics=eval_metrics)

if __name__ == "__main__":
    logger.info("Initializing and running the ZenML pipeline...")
    
    # Execute the pipeline
    text_summarization_pipeline()
    
    logger.info("ZenML pipeline execution finished successfully.")
