import os
import mlflow
import optuna
import torch
import logging
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline
from src.config import (
    MLFLOW_TRACKING_URI,
    MLFLOW_EXPERIMENT_NAME,
    MODEL_REGISTRY_NAME,
    TUNE_SAMPLE_SIZE,
    DEFAULT_MODEL_NAME
)
from src.utils import load_data_sample, calculate_rouge_scores, SystemMonitor

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def objective(trial, model_name, articles, references, device):
    # Suggest hyperparameters
    num_beams = trial.suggest_int("num_beams", 1, 6)
    length_penalty = trial.suggest_float("length_penalty", 0.5, 2.5)
    
    # Start nested MLflow run for each trial
    with mlflow.start_run(run_name=f"trial_{trial.number}", nested=True):
        mlflow.log_params({
            "num_beams": num_beams,
            "length_penalty": length_penalty,
            "trial_number": trial.number
        })
        
        # Start resource monitoring
        monitor = SystemMonitor()
        monitor.start()
        
        try:
            predictions = []
            for article in articles:
                truncated_article = article[:4000]
                summary = trial.user_attrs["pipeline"](
                    truncated_article,
                    max_length=142,
                    min_length=30,
                    num_beams=num_beams,
                    length_penalty=length_penalty,
                    do_sample=False
                )[0]['summary_text']
                predictions.append(summary)
                monitor.sample() # Sample after each inference iteration
                
            # Stop monitoring and collect metrics
            sys_metrics = monitor.stop()
            
            scores = calculate_rouge_scores(predictions, references)
            score = scores['rougeL']
            
            # Log metrics
            mlflow.log_metric("rouge1", scores['rouge1'])
            mlflow.log_metric("rouge2", scores['rouge2'])
            mlflow.log_metric("rougeL", score)
            
            # Log system metrics
            mlflow.log_metrics(sys_metrics)
            
            return score

        except Exception as e:
            logger.error(f"Trial {trial.number} failed: {e}")
            mlflow.log_param("status", "failed")
            return 0.0

def tune_hyperparameters(model_name: str = None, n_trials: int = None):
    if n_trials is None:
        n_trials = int(os.getenv("N_TRIALS", "3"))
    if not model_name:
        model_name = os.getenv("BEST_MODEL_NAME", DEFAULT_MODEL_NAME)
        
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(MLFLOW_EXPERIMENT_NAME)
    
    logger.info(f"Tuning generation parameters for model: {model_name}")
    
    df = load_data_sample(split="validation", sample_size=TUNE_SAMPLE_SIZE)
    articles = df['article'].tolist()
    references = df['highlights'].tolist()
    
    device = 0 if torch.cuda.is_available() else -1
    
    # Load tokenizer and model once to avoid reload overhead
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    summarizer = pipeline(
        "summarization",
        model=model,
        tokenizer=tokenizer,
        device=device
    )
    
    # Parent MLflow Run for the study
    with mlflow.start_run(run_name=f"optuna_tuning_{model_name.replace('/', '_')}") as parent_run:
        mlflow.log_params({
            "model_name": model_name,
            "tuning_samples": len(articles),
            "n_trials": n_trials
        })
        
        study = optuna.create_study(direction="maximize")
        
        # Inject the pipeline into study/trials attributes so the objective can access it
        study.enqueue_trial({"num_beams": 4, "length_penalty": 2.0}) # standard default settings
        
        def lambda_obj(trial):
            trial.set_user_attr("pipeline", summarizer)
            return objective(trial, model_name, articles, references, device)
            
        study.optimize(lambda_obj, n_trials=n_trials)
        
        logger.info(f"Best trial parameters: {study.best_params}")
        logger.info(f"Best ROUGE-L: {study.best_value:.4f}")
        
        # Log best parameters and metrics in the parent run
        mlflow.log_params({f"best_{k}": v for k, v in study.best_params.items()})
        mlflow.log_metric("best_rougeL", study.best_value)
        
        # Log and Register the Best Model in registry
        components = {
            "model": model,
            "tokenizer": tokenizer,
        }
        
        # We save model with the best parameters in metadata / config
        model_info = mlflow.transformers.log_model(
            transformers_model=components,
            artifact_path="best_model",
            task="summarization",
            registered_model_name=MODEL_REGISTRY_NAME,
            model_config={
                "num_beams": study.best_params.get("num_beams", 4),
                "length_penalty": study.best_params.get("length_penalty", 1.0),
                "max_length": 142,
                "min_length": 30,
                "do_sample": False
            }
        )
        
        logger.info(f"Best model registered under: {MODEL_REGISTRY_NAME}")
        return study.best_params

if __name__ == "__main__":
    tune_hyperparameters()
