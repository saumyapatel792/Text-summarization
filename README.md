# Text Summarization Service (HF + Full Deployment Loop)

A production-grade text summarization API built with Hugging Face transformers (BART/T5), tuned with Optuna, monitored with Prometheus and Grafana, tracked with MLflow, and verified via ROUGE regression tests in CI/CD.

---

## Quick Start (Local Setup)

### 1. Installation
Create a virtual environment and install the dependencies:
```bash
python -m venv venv
venv\Scripts\activate     # On Windows
source venv/bin/activate  # On macOS/Linux

# Install PyTorch CPU first to save download bandwidth
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt
```

### 2. Running Benchmarks & Tuning
Execute the benchmarking script to evaluate Hugging Face candidate models, and then tune the generation parameters:
```bash
# Start MLflow in a separate terminal or let the script run locally (saves runs in ./mlruns)
python src/train.py
python src/tune.py
```

### 3. Spin up the Production Stack (FastAPI, MLflow, Prometheus, Grafana)
Navigate to the `docker/` folder and launch the Docker Compose orchestration:
```bash
cd docker
docker compose up --build -d
```
The services will be available at:
* **FastAPI Service:** [http://localhost:8000](http://localhost:8000)
* **MLflow Tracking Server:** [http://localhost:5000](http://localhost:5000)
* **Prometheus UI:** [http://localhost:9090](http://localhost:9090)
* **Grafana Dashboard:** [http://localhost:3000](http://localhost:3000) (User: `admin`, Password: `admin`)

---

## 1. MLflow Tracking & Experimentation Guide

Once MLflow is running (either locally or via Docker), navigate to `http://localhost:5000` to interact with the tracking server.

### 1) Compare Runs
1. In the left panel, select the experiment: **`Text_Summarization_System`**.
2. Tick the checkboxes next to the runs (e.g. the model benchmarks or Optuna trials) you want to compare.
3. Click the **Compare** button at the top of the table.
4. This opens the comparison page showing side-by-side parameters (e.g., `model_name`, `num_beams`, `length_penalty`) and metrics (`rougeL`, `avg_latency_sec`).

### 2) Filter Runs
In the search bar above the runs list, write SQL-like queries to filter runs:
* **To find only trials with high performance:**
  ```sql
  metrics.rougeL > 0.15
  ```
* **To search for a specific model:**
  ```sql
  params.model_name = 'sshleifer/distilbart-cnn-6-6'
  ```
* **Combine parameters & performance filters:**
  ```sql
  params.num_beams >= 4 and metrics.avg_latency_sec < 2.0
  ```

### 3) Visualize Hyperparameter Effects
1. Select the Optuna tuning run or check multiple nested trials and click **Compare**.
2. Scroll down to the **Visualizations** section.
3. Choose **Parallel Coordinates Plot**:
   * Add **Parameters** (`num_beams`, `length_penalty`) and **Metrics** (`rougeL`).
   * This displays interactive lines mapping hyperparameter paths to performance, revealing sweet spots.
4. Choose **Scatter Plot**:
   * Set X-axis to `length_penalty`, Y-axis to `rougeL` to see correlation.

### 4) Inspect the Saved (Best) Model
1. Go to the **Models** tab in the top navigation bar.
2. Select **`TextSummarizerBest`** (the model registered by `tune.py`).
3. Click on the latest version number to inspect:
   * **Source Run**: Link to the Optuna trial that generated this version.
   * **Schema**: Expected input and output signature.
   * **Artifacts**: File hierarchy containing the saved tokenizer, model weights (`pytorch_model.bin` or `model.safetensors`), and generation config JSON.

### 5) Export Results as CSV
Run the provided exporter utility to generate `mlflow_runs.csv`:
```bash
python mlflow_export.py
```
This queries the MLflow client and outputs a structured CSV for offline Excel/Pandas analysis.

---

## 2. Prometheus Query Language (PromQL) Guide

Access the Prometheus dashboard at `http://localhost:9090` to execute queries.

### 1) Check Target Health
To check if the FastAPI summarization application is actively scraped:
```promql
up{job="summarization_service"}
```
* **Value = 1**: The application is up and running healthy.
* **Value = 0**: The application is unreachable.

### 2) Query a Rate of Change
To calculate the average rate of requests per second received by the API over the last 5 minutes:
```promql
rate(summarization_service_requests_total[5m])
```
To calculate the rate of success vs error responses:
```promql
rate(summarization_service_requests_total{status="success"}[5m])
```

### 3) Average Over Time
To calculate the average input token length over the last 1 hour:
```promql
avg_over_time(summarization_service_input_token_length_sum[1h]) / avg_over_time(summarization_service_input_token_length_count[1h])
```
To calculate the average request latency over the last 30 minutes:
```promql
rate(summarization_service_request_latency_seconds_sum[30m]) / rate(summarization_service_request_latency_seconds_count[30m])
```

### 4) Compare Graph vs Table View
1. Type a query in the Prometheus Expression input box (e.g. `summarization_service_requests_total`).
2. Click **Execute**.
3. Toggle between the **Graph** tab (visual time-series lines) and the **Table** tab (instant vector values at the current timestamp).

### 5) Sample Queries (Runs/Traffic within a timeframe)
* **Number of successful API requests processed in the last 5 minutes:**
  ```promql
  increase(summarization_service_requests_total{status="success"}[5m])
  ```
* **Total request count in the last 1 hour:**
  ```promql
  increase(summarization_service_requests_total[1h])
  ```
* **Total request count in the last day (24h):**
  ```promql
  increase(summarization_service_requests_total[24h])
  ```
* **Estimated request count in the last month (30 days):**
  ```promql
  increase(summarization_service_requests_total[30d])
  ```

---

## 3. Grafana Operations Guide

Access Grafana at `http://localhost:3000` (credentials: `admin`/`admin`).

### 1) Edit an Existing Panel
1. Open the auto-provisioned **Text Summarization Service Performance** dashboard.
2. Hover over the panel you want to edit (e.g., "Request Latency Over Time").
3. Click the three dots in the top-right corner of the panel and select **Edit**.
4. In the editor view:
   * **Query Section (Bottom)**: Modify the PromQL expression.
   * **Panel Options (Right Panel)**: Change panel type (e.g., Time Series to Bar Chart), title, colors, and line configurations.
5. Click **Apply** in the top-right corner to save to the current session dashboard.

### 2) Create a New Panel
1. Click the **Add Panel** button (graph icon with a plus) in the top-right dashboard menu.
2. Select **Add a new panel**.
3. Under the **Query** tab, select the data source as **Prometheus**.
4. In the query field, enter a PromQL statement, e.g. for p99 latency:
   ```promql
   histogram_quantile(0.99, sum(rate(summarization_service_request_latency_seconds_bucket[5m])) by (le))
   ```
5. On the right, choose your visualization style (e.g., Gauge or Time series) and set the title to `p99 Latency (5m)`.
6. Click **Save** in the top-right corner.

### 3) Set an Alert Threshold
1. While editing a panel (or creating a new one), click on the **Alert** tab next to Query/Transform in the bottom panel editor.
2. Click **Create alert rule from this panel**.
3. Define the evaluation criteria:
   * Select the query representation (e.g., latency values).
   * Set condition: `IS ABOVE 5` (alert if average latency exceeds 5 seconds).
4. Configure evaluation interval (e.g. evaluate every `1m` for `5m`).
5. Assign a contact point (e.g. Slack/Email) to route notifications, then save the rule.

### 4) Change Time Range / Refresh Rate
* **Time Range**: In the top-right corner of the dashboard, click the clock icon dropdown. Select a pre-configured time window (e.g., `Last 5 minutes`, `Last 1 hour`, `Last 30 days`) or define a custom absolute range.
* **Refresh Rate**: Next to the time range dropdown, click the refresh rate arrow/dropdown. Choose how frequently Grafana queries the backend Prometheus instance (e.g. `Off`, `5s`, `10s`, `1m`).
