import os
import pandas as pd
import mlflow
from src.config import MLFLOW_TRACKING_URI, MLFLOW_EXPERIMENT_NAME

def export_runs_to_csv(output_path: str = "mlflow_runs.csv"):
    # Set tracking URI
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    
    # Get experiment
    experiment = mlflow.get_experiment_by_name(MLFLOW_EXPERIMENT_NAME)
    if not experiment:
        print(f"Experiment '{MLFLOW_EXPERIMENT_NAME}' not found.")
        return
        
    print(f"Fetching runs for experiment: '{MLFLOW_EXPERIMENT_NAME}' (ID: {experiment.experiment_id})...")
    
    # Search runs
    runs_df = mlflow.search_runs(experiment_ids=[experiment.experiment_id])
    
    if runs_df.empty:
        print("No runs found in this experiment.")
        return
        
    # Export to CSV
    runs_df.to_csv(output_path, index=False)
    print(f"Successfully exported {len(runs_df)} runs to {output_path}")

if __name__ == "__main__":
    export_runs_to_csv()
