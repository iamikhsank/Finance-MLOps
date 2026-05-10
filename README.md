# Financial MLOps Pipeline

<p align="center">
  <img src="https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white" alt="Python" />
  <img src="https://img.shields.io/badge/Apache_Airflow-017CEE?style=for-the-badge&logo=Apache-Airflow&logoColor=white" alt="Airflow" />
  <img src="https://img.shields.io/badge/PyTorch-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white" alt="PyTorch" />
  <img src="https://img.shields.io/badge/FastAPI-009688?style=for-the-badge&logo=FastAPI&logoColor=white" alt="FastAPI" />
  <img src="https://img.shields.io/badge/MongoDB-4EA94B?style=for-the-badge&logo=mongodb&logoColor=white" alt="MongoDB" />
  <img src="https://img.shields.io/badge/Docker-2496ED?style=for-the-badge&logo=docker&logoColor=white" alt="Docker" />
  <img src="https://img.shields.io/badge/GitHub_Actions-2088FF?style=for-the-badge&logo=github-actions&logoColor=white" alt="GitHub Actions" />
</p>


## Project Overview

This repository contains an automated, end-to-end Machine Learning Operations (MLOps) pipeline designed for stock price prediction and financial market analysis. The system automates the complete lifecycle of a Deep Learning model, ranging from daily data ingestion and feature engineering to continuous model retraining, experiment tracking, and real-time deployment via an API interface.

Built around predicting equity data (specifically focused on BBCA.JK tickers as default), this project serves as a blueprint for enterprise-grade ML orchestration using Docker containerization, workflow managers, and cloud data warehouses.

## Key Architecture Features

```mermaid
flowchart TB
    %% Styling Configuration
    classDef data fill:#e0f7fa,stroke:#0277bd,stroke-width:2px,color:#000
    classDef orchestrator fill:#fce4ec,stroke:#c2185b,stroke-width:2px,color:#000
    classDef ml fill:#fff3e0,stroke:#ef6c00,stroke-width:2px,color:#000
    classDef api fill:#e8f5e9,stroke:#2e7d32,stroke-width:2px,color:#000
    classDef ci fill:#f3e5f5,stroke:#7b1fa2,stroke-width:2px,color:#000
    classDef external fill:#eceff1,stroke:#546e7a,stroke-width:2px,color:#000,stroke-dasharray: 5 5

    %% Nodes
    Yahoo(["fa:fa-globe Yahoo Finance API"])
    
    subgraph Data Pipeline
        Ingest["fab:fa-python Data Ingestion Script"]
        Mongo[("fas:fa-database MongoDB Atlas")]
    end
    
    subgraph Orchestration
        Airflow(("fas:fa-wind Apache Airflow DAGs"))
    end
    
    subgraph Machine Learning Pipeline
        Train["fas:fa-brain PyTorch LSTM Model"]
        MLflow[["fas:fa-chart-line MLflow (DagsHub)"]]
    end
    
    subgraph Deployment & Serving
        FastAPI{"fas:fa-bolt FastAPI Server"}
        Client(["fas:fa-user End User / Client"])
    end
    
    subgraph CI/CD & Infrastructure
        GitHub["fab:fa-github GitHub Actions"]
        Docker["fab:fa-docker Docker & GHCR"]
    end
    
    %% Relationships
    Yahoo -->|"Raw Equity Data"| Ingest
    Ingest -->|"Stores Features"| Mongo
    
    Airflow -.->|"Triggers Schedule"| Ingest
    Airflow -.->|"Conditional Retraining"| Train
    
    Mongo -->|"Historical Data"| Train
    Train -->|"Logs Metrics & Weights"| MLflow
    Train -->|"Deploys Best Model"| FastAPI
    
    FastAPI -->|"JSON Predictions"| Client
    
    GitHub -->|"Automated Testing"| Docker
    Docker -.->|"Containerizes"| Airflow
    Docker -.->|"Containerizes"| FastAPI

    %% Apply Styles
    class Yahoo,Client external
    class Ingest,Mongo data
    class Airflow orchestrator
    class Train,MLflow ml
    class FastAPI api
    class GitHub,Docker ci
```


*   **Automated Ingestion:** Automated fetching of market data via `yfinance` with persistence layer managed in MongoDB Atlas.
*   **Deep Learning Forecasting:** Implements an LSTM (Long Short-Term Memory) neural network architecture built with PyTorch for temporal pattern recognition.
*   **Experiment Tracking & Logging:** Integrated with MLflow via DagsHub to visualize loss metrics, track hyperparameters, and save weight checkpoints remotely.
*   **Workflow Orchestration:** Utilizes Apache Airflow DAGs to conditionalize and run continuous data scraping, feature derivation, and adaptive retraining jobs.
*   **Scalable Inference:** Uses FastAPI with Uvicorn to expose low-latency endpoints for real-time prediction serving.
*   **Robust CI/CD:** Github Actions automated verification testing and direct building to the GitHub Container Registry (GHCR).

## Technical Stack

| Component | Technology Used |
| :--- | :--- |
| **Orchestrator** | Apache Airflow 2.8.1 |
| **Machine Learning** | PyTorch, Scikit-Learn, TA (Technical Analysis Library) |
| **API Layer** | FastAPI, Pydantic, Uvicorn |
| **Database** | MongoDB Atlas (Production), PostgreSQL (Airflow Metadata) |
| **Tracking** | MLflow Hosted via DagsHub |
| **Containerization** | Docker, Docker Compose |
| **CI/CD** | GitHub Actions, GHCR |
| **Language** | Python 3.10 |

## Directory Structure

```text
Finance-MLOps/
├── .github/workflows/       # CI/CD GitHub Actions Pipeline definition
├── api/
│   └── app/main.py          # FastAPI application entrypoint and routes
├── dags/
│   └── mlops_pipeline.py    # Apache Airflow directed acyclic graph definition
├── data_pipeline/
│   └── src/
│       ├── db_helper.py     # MongoDB persistence utility wrappers
│       └── ingest.py        # Data loading logics from financial sources
├── eda/                     # Research notebooks and analysis scripts
├── k8s/                     # Optional Kubernetes manifests for deployment scaling
├── ml_service/
│   └── src/
│       ├── model.py         # PyTorch LSTM Neural Net Architecture
│       ├── train.py         # Fitting execution with MLflow tracking
│       ├── evaluate.py      # Regression evaluation metrics compute
│       └── retrain_loop.py  # Conditional logical flows for retraining
├── models/                  # Local caching directory for artifacts
├── Dockerfile.airflow       # Customized airflow execution environment
├── Dockerfile.api           # Slim container for fast prediction serving
├── docker-compose.yml       # Multi-service container configuration
├── requirements.txt         # Application and framework dependencies
└── .env                     # Environment variable store
```

## Prerequisites

Ensure that the following prerequisites are installed locally before environment orchestration:

1.  Docker and Docker Compose (v3.8+)
2.  Git
3.  A Valid MongoDB Atlas Connection String
4.  A DagsHub account and MLflow token for Remote Tracking (Recommended)

## Configuration Environment

Create a `.env` file located at the root of the working repository with the following mandatory definitions:

```env
MONGO_URI=mongodb+srv://<user>:<pass>@cluster.mongodb.net/
DB_NAME=finance_mlops
COLLECTION_NAME=stock_data

MLFLOW_TRACKING_URI=https://dagshub.com/<username>/Finance-MLOps.mlflow
MLFLOW_TRACKING_USERNAME=<username>
MLFLOW_TRACKING_PASSWORD=<dagshub_token>

TICKER=BBCA.JK
```

## Running with Docker Compose

The entire application infrastructure (PostgreSQL metadata db, Airflow Init, Webserver, Scheduler, and FastAPI server) is encapsulated into a cohesive `docker-compose` stack.

### Step 1: Initialize Stack
Verify connectivity and build relevant components locally:
```bash
docker-compose up airflow-init
```

### Step 2: Execute Application Services
Run the services simultaneously in detached daemon mode:
```bash
docker-compose up -d
```

### Step 3: Verify Accessible Dashboards
Upon successful startup, the system maps default access to following hosts:
*   **Airflow Webserver:** `http://localhost:8080` (Default User/Pass: `airflow` / `airflow`)
*   **FastAPI (Auto-Docs):** `http://localhost:8000/docs`

## Application API Endpoints

Available primary routes provided by the FastAPI interface:

*   `GET /` : Health confirmation and root availability.
*   `POST /predict` : Takes financial lag sequence and executes inferential pass via current PyTorch artifact.
*   `GET /history` : Returns metadata of cached dataset loads.

## Continuous Integration / Continuous Deployment (CI/CD)

Automated software quality assurance is executed via GitHub Actions workflows, enforced upon merges or pull requests targeted at the `main` stable branch.

The pipeline utilizes the following workflow cycle:
1.  **Test Phase:** Validates dependencies integrity and performs module unit instantiation tests in temporary runner runners.
2.  **Container Generation:** On successful tests, rebuilds standard Docker images derived from `Dockerfile.api` and `Dockerfile.airflow`.
3.  **Registry Push:** Securely authenticates and pushes the container images directly onto the GitHub Container Registry (GHCR), hosted at `ghcr.io`.

## License

Distributed under the terms located within the included [LICENSE](file:///d:/Antigravity/Machine%20Learning%20Project/Finance-MLOps/LICENSE) file.