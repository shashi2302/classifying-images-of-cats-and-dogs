Cat vs. Dog Image Classification

This project implements a complete machine learning pipeline for classifying images of cats and dogs, from data acquisition to deployment.

## Project Structure


.
├── .github/            # CI/CD workflows
├── configs/            # Configuration files
├── data/               # Data storage
│   ├── raw/            # Raw downloaded data
│   ├── processed/      # Processed data ready for training
│   └── database/       # SQLite database
├── docker/             # Docker configuration
├── kubernetes/         # Kubernetes deployment files
├── mlflow/             # MLflow tracking
├── notebooks/          # Jupyter notebooks for exploration
├── src/                # Source code
│   ├── api/            # Flask API
│   ├── data/           # Data acquisition and processing
│   ├── models/         # Model training and evaluation
│   └── utils/          # Utility functions
└── tests/              # Unit and integration tests
```

## Features

- Data acquisition and cleaning using Pandas, 
- Data storage in SQLite database with SQLAlchemy
- Model development with Scikit-learn
- Feature engineering and model evaluation
- Visualizations with Seaborn and Matplotlib
- MLOps logging with MLflow
- Comprehensive testing with pytest
- Containerization with Docker
- CI/CD pipeline with GitHub Actions
- Kubernetes deployment
- Monitoring and health checks

## Getting Started

### Prerequisites

- Python 3.8+
- Docker
- Kubernetes (for deployment)

### Installation

1. Clone the repository
2. Install dependencies:
   ```
   pip install -r requirements.txt
   ```

3. Run the data acquisition script:
   ```
   python src/data/acquire_data.py
   ```

4. Train the model:
   ```
   python src/models/train_model.py
   ```

5. Run the API locally:
   ```
   python src/api/app.py
   ```

### Docker

Build and run the Docker container:

```
docker build -t cat-dog-classifier .
docker run -p 5000:5000 cat-dog-classifier
```

### Kubernetes Deployment

Apply the Kubernetes configuration:

```
kubectl apply -f kubernetes/deployment.yaml
kubectl apply -f kubernetes/service.yaml
```

## Success Metrics

- Model accuracy: >90% on test set
- API response time: <200ms
- System uptime: >99.9%
- Resource utilization: <70% CPU, <80% memory
