from prometheus_client import Histogram, Counter

# Namespace or prefix for all metrics
NAMESPACE = "summarization_service"

# Request Latency Metric
LATENCY_HISTOGRAM = Histogram(
    "request_latency_seconds",
    "Time taken to process summarization request in seconds",
    namespace=NAMESPACE,
    buckets=[0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0, 30.0, 60.0]
)

# Input token length distribution
INPUT_TOKEN_HISTOGRAM = Histogram(
    "input_token_length",
    "Number of tokens in the input text",
    namespace=NAMESPACE,
    buckets=[10, 50, 100, 250, 500, 750, 1000, 1500, 2000]
)

# Output token length distribution
OUTPUT_TOKEN_HISTOGRAM = Histogram(
    "output_token_length",
    "Number of tokens in the generated summary",
    namespace=NAMESPACE,
    buckets=[5, 10, 20, 30, 50, 70, 100, 120, 150, 200]
)

# Request counter for overall traffic stats
REQUEST_COUNTER = Counter(
    "requests_total",
    "Total number of summarization requests processed",
    namespace=NAMESPACE,
    labelnames=["status"]
)
