import time
import os
import logging
from contextlib import asynccontextmanager
from typing import Optional

from fastapi import FastAPI, HTTPException, Response
from fastapi.responses import PlainTextResponse, RedirectResponse
from pydantic import BaseModel, Field
import mlflow
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline
from prometheus_client import generate_latest, CONTENT_TYPE_LATEST, REGISTRY

from src.config import (
    MLFLOW_TRACKING_URI,
    MODEL_REGISTRY_NAME,
    DEFAULT_MODEL_NAME,
    DEFAULT_MIN_LENGTH,
    DEFAULT_MAX_LENGTH
)
from src.metrics import (
    LATENCY_HISTOGRAM,
    INPUT_TOKEN_HISTOGRAM,
    OUTPUT_TOKEN_HISTOGRAM,
    REQUEST_COUNTER
)

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global variables for model, tokenizer, and configuration
summarizer_pipeline = None
model_tokenizer = None
active_model_name = None
generation_config = {}

class SummarizeRequest(BaseModel):
    text: str = Field(..., description="The text content to be summarized")
    num_beams: Optional[int] = Field(None, ge=1, le=10, description="Beam width for beam search")
    length_penalty: Optional[float] = Field(None, ge=0.0, le=5.0, description="Length penalty for generation")

class SummarizeResponse(BaseModel):
    summary: str
    latency_sec: float
    input_tokens: int
    output_tokens: int
    model_used: str

@asynccontextmanager
async def lifespan(app: FastAPI):
    global summarizer_pipeline, model_tokenizer, active_model_name, generation_config
    
    device = 0 if torch.cuda.is_available() else -1
    logger.info(f"App starting up. Using device: {'GPU' if device == 0 else 'CPU'}")
    
    # Try loading from MLflow Model Registry
    loaded_from_registry = False
    # Reachability check with a short timeout to prevent client blocking when MLflow is offline
    mlflow_reachable = False
    if MLFLOW_TRACKING_URI:
        try:
            import urllib.request
            urllib.request.urlopen(MLFLOW_TRACKING_URI, timeout=1.5)
            mlflow_reachable = True
        except Exception:
            pass

    try:
        if not mlflow_reachable:
            raise ConnectionError("MLflow tracking server is not reachable")
        mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
        logger.info(f"Attempting to load best model '{MODEL_REGISTRY_NAME}' from MLflow registry at {MLFLOW_TRACKING_URI}...")
        # Get the latest version in the registry
        client = mlflow.tracking.MlflowClient()
        latest_versions = client.get_latest_versions(MODEL_REGISTRY_NAME, stages=["None", "Production", "Staging"])
        
        if latest_versions:
            latest_version = latest_versions[0].version
            model_uri = f"models:/{MODEL_REGISTRY_NAME}/{latest_version}"
            logger.info(f"Found model version {latest_version}. Loading from {model_uri}...")
            
            # Ensure destination directory exists
            os.makedirs("./local_mlflow_model", exist_ok=True)
            # Load registered model using mlflow.transformers
            model_info = mlflow.transformers.load_model(model_uri, dst_path="./local_mlflow_model")
            
            # Extract tokenizer, model and configuration
            # mlflow transformers loading gives us a dict or pipeline-like wrapper
            # If it's a pipeline:
            if hasattr(model_info, 'tokenizer') and hasattr(model_info, 'model'):
                model_tokenizer = model_info.tokenizer
                summarizer_pipeline = pipeline(
                    "summarization",
                    model=model_info.model,
                    tokenizer=model_tokenizer,
                    device=device
                )
            else:
                # If model_info is a dictionary containing model/tokenizer components
                model_tokenizer = model_info.get("tokenizer")
                summarizer_pipeline = pipeline(
                    "summarization",
                    model=model_info.get("model"),
                    tokenizer=model_tokenizer,
                    device=device
                )
            
            # Extract pipeline configuration if saved
            if hasattr(model_info, 'model_config'):
                generation_config = model_info.model_config
            
            active_model_name = f"Registry:{MODEL_REGISTRY_NAME}:v{latest_version}"
            loaded_from_registry = True
            logger.info("Successfully loaded model from MLflow registry.")
        else:
            logger.warning(f"No versions found in MLflow registry for model: {MODEL_REGISTRY_NAME}")
            
    except Exception as e:
        logger.warning(f"Could not load model from MLflow Registry: {e}. Falling back to default model: {DEFAULT_MODEL_NAME}")
        
    # Fallback to local default pre-trained HF model if registry failed/empty
    if not loaded_from_registry:
        try:
            logger.info(f"Loading fallback default model '{DEFAULT_MODEL_NAME}' from Hugging Face...")
            import gc
            gc.collect()
            model_tokenizer = AutoTokenizer.from_pretrained(DEFAULT_MODEL_NAME)
            model = AutoModelForSeq2SeqLM.from_pretrained(DEFAULT_MODEL_NAME, low_cpu_mem_usage=True)
            summarizer_pipeline = pipeline(
                "summarization",
                model=model,
                tokenizer=model_tokenizer,
                device=device
            )
            active_model_name = f"HF:{DEFAULT_MODEL_NAME}"
            generation_config = {
                "num_beams": 4,
                "length_penalty": 2.0,
                "max_length": DEFAULT_MAX_LENGTH,
                "min_length": DEFAULT_MIN_LENGTH,
                "do_sample": False
            }
            logger.info("Successfully loaded fallback model.")
        except Exception as fallback_err:
            logger.critical(f"Failed to load fallback model: {fallback_err}")
            raise RuntimeError(f"No model available to serve: {fallback_err}")
            
    yield
    # Shutdown logic
    logger.info("App shutting down.")

app = FastAPI(
    title="Text Summarization API",
    description="Production-grade HF Text Summarization service with MLflow tracking and Prometheus metrics.",
    version="1.0.0",
    lifespan=lifespan
)

@app.get("/")
async def root():
    """
    Redirects the root URL to the API documentation (/docs).
    """
    return RedirectResponse(url="/docs")

@app.get("/health", status_code=200)
async def health_check():
    """
    Health check endpoint for container orchestrators.
    """
    if summarizer_pipeline is None:
        raise HTTPException(status_code=503, detail="Model is not initialized.")
    return {
        "status": "healthy",
        "model": active_model_name,
        "device": str(summarizer_pipeline.device)
    }

@app.post("/summarize", response_model=SummarizeResponse)
async def summarize(request: SummarizeRequest):
    """
    Generates a summary for the provided input text.
    """
    if not request.text.strip():
        REQUEST_COUNTER.labels(status="error").inc()
        raise HTTPException(status_code=400, detail="Input text cannot be empty.")
        
    start_time = time.time()
    
    try:
        # Determine number of tokens in input
        input_tokens = len(model_tokenizer.encode(request.text, truncation=True))
        INPUT_TOKEN_HISTOGRAM.observe(input_tokens)
        
        # Override parameters if provided in request, else use model defaults
        num_beams = request.num_beams if request.num_beams is not None else generation_config.get("num_beams", 4)
        length_penalty = request.length_penalty if request.length_penalty is not None else generation_config.get("length_penalty", 2.0)
        
        # Run summarization
        # Char limit truncation as safety
        truncated_text = request.text[:8000]
        
        summary_result = summarizer_pipeline(
            truncated_text,
            max_length=DEFAULT_MAX_LENGTH,
            min_length=DEFAULT_MIN_LENGTH,
            num_beams=num_beams,
            length_penalty=length_penalty,
            do_sample=False
        )
        
        summary_text = summary_result[0]['summary_text']
        
        # Determine output token length
        output_tokens = len(model_tokenizer.encode(summary_text))
        OUTPUT_TOKEN_HISTOGRAM.observe(output_tokens)
        
        latency = time.time() - start_time
        LATENCY_HISTOGRAM.observe(latency)
        
        # Increment request counter
        REQUEST_COUNTER.labels(status="success").inc()
        
        return SummarizeResponse(
            summary=summary_text,
            latency_sec=latency,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            model_used=active_model_name
        )
        
    except Exception as e:
        logger.error(f"Error during summarization: {e}")
        REQUEST_COUNTER.labels(status="error").inc()
        raise HTTPException(status_code=500, detail=f"Inference error: {str(e)}")

@app.get("/metrics")
async def get_metrics():
    """
    Exposes metrics for Prometheus scraping.
    """
    return Response(content=generate_latest(REGISTRY), media_type=CONTENT_TYPE_LATEST)
