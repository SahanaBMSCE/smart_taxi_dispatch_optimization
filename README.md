# Smart Taxi Dispatch Optimization using Reinforcement Learning and MLOps

## Project Overview

Smart Taxi Dispatch Optimization is a Reinforcement Learning and MLOps project designed to optimize taxi dispatch decisions in a simulated urban transportation environment.

The project compares:

1. Baseline Nearest-Taxi Strategy  
2. Q-Learning  
3. Proximal Policy Optimization (PPO)  

The objective is to minimize passenger waiting time while demonstrating a complete end-to-end MLOps workflow including:

- Experiment Tracking using MLflow
- Data and Model Versioning using DVC
- API Deployment using FastAPI
- Containerization using Docker
- Workflow Orchestration using Apache Airflow
- CI/CD Automation using GitHub Actions
- Reproducible Training Pipelines

---

# SDG Alignment

This project contributes to the following United Nations Sustainable Development Goals (SDGs):

## SDG 9 — Industry, Innovation and Infrastructure
- Intelligent transportation optimization
- AI-driven infrastructure enhancement
- Scalable ML deployment pipelines

## SDG 11 — Sustainable Cities and Communities
- Reduced passenger waiting time
- Efficient urban mobility management
- Improved transportation resource utilization

## SDG 13 — Climate Action
- Reduced unnecessary taxi movement
- Optimized fleet allocation
- Potential reduction in fuel consumption and emissions

---

# Problem Statement

Traditional taxi dispatch systems commonly rely on heuristic-based nearest-vehicle selection strategies.

Such approaches:
- do not adapt dynamically
- fail to learn from demand patterns
- often lead to inefficient fleet allocation

This project explores whether Reinforcement Learning can learn better dispatch policies that reduce average passenger waiting time compared to static heuristic methods.

---

# Objectives

The primary objectives of the project are:

- Simulate a taxi dispatch environment
- Implement a baseline dispatch strategy
- Train a Q-Learning agent
- Train a PPO agent
- Compare dispatch performance
- Track experiments using MLflow
- Version models and pipelines using DVC
- Deploy APIs using FastAPI
- Containerize the application using Docker
- Automate workflows using CI/CD pipelines

---

# Technologies Used

| Category | Technologies |
|---|---|
| Programming Language | Python 3.10 |
| Reinforcement Learning | Q-Learning, PPO |
| RL Library | Stable-Baselines3 |
| API Framework | FastAPI |
| Experiment Tracking | MLflow |
| Data & Model Versioning | DVC |
| Containerization | Docker, Docker Compose |
| Workflow Orchestration | Apache Airflow |
| CI/CD | GitHub Actions |
| Visualization | Matplotlib |
| Model Serialization | Joblib |

---

# Project Structure

```text
smart_taxi_dispatch_optimization/
│
├── api/
│   └── app.py
│
├── configs/
│   ├── qlearning_v1.yaml
│   └── qlearning_v2_explored.yaml
│
├── dags/
│   └── taxi_pipeline.py
│
├── docs/
│   └── methodology.md
│
├── experiments/
│   ├── evaluate.py
│   ├── plot_results.py
│   └── train.py
│
├── features/
│   └── demand_hotspots.py
│
├── models/
│   └── ppo/
│
├── plots/
│   ├── reward_curve_v2.png
│   ├── reward_curve_ppo.png
│   └── wait_time_comparison.png
│
├── policies/
│   └── policy_v2_explored.pkl
│
├── results/
│   ├── results_qlearning_v2_explored.json
│   └── results_ppo.json
│
├── sim/
│   ├── ppo_agent.py
│   ├── ppo_env.py
│   └── taxi_env.py
│
├── Dockerfile
├── docker-compose.yml
├── dvc.yaml
├── requirements.txt
└── README.md
```

---

# Reinforcement Learning Approaches

## 1. Baseline Strategy

The baseline strategy dispatches the nearest available taxi to the passenger pickup location.

### Characteristics
- Rule-based heuristic
- No learning capability
- Static dispatch behavior
- Used as benchmark

---

## 2. Q-Learning

Q-Learning is a value-based Reinforcement Learning algorithm that learns an optimal dispatch policy using a Q-table.

### Features
- Epsilon-greedy exploration
- Temporal Difference learning
- State-action value estimation
- Tabular policy learning

### Reward Function

```text
Reward = - Waiting Time
```

The agent learns to minimize waiting time by maximizing cumulative reward.

---

## 3. Proximal Policy Optimization (PPO)

PPO is a policy-gradient Reinforcement Learning algorithm implemented using Stable-Baselines3.

### Features
- Neural network policy learning
- Stable policy optimization
- Improved convergence
- Better scalability

### PPO Configuration

| Parameter | Value |
|---|---|
| Learning Rate | 0.0003 |
| Gamma | 0.99 |
| Batch Size | 64 |
| Timesteps | 50000 |

---

# Performance Comparison

| Model | Average Waiting Time |
|---|---|
| Baseline | 2.14 |
| Q-Learning | 1.13 |
| PPO | 0.86 |

## Observations

- Both RL approaches outperform the baseline strategy.
- PPO achieves the lowest average waiting time.
- PPO demonstrates superior policy optimization capability.

---

# Results

## Generated Outputs

### Reward Curves
- `plots/reward_curve_v2.png`
- `plots/reward_curve_ppo.png`

### Model Comparison Plot
- `plots/wait_time_comparison.png`

### Experiment Results
- `results/results_qlearning_v2_explored.json`
- `results/results_ppo.json`

### MLflow Artifacts
- Reward plots
- Metrics
- Hyperparameters
- Model artifacts

---

# MLOps Pipeline

## MLflow Experiment Tracking

MLflow is used for:
- Parameter tracking
- Metric logging
- Artifact management
- Reward curve visualization
- Experiment comparison

---

## DVC Pipeline

DVC is used for:
- Data versioning
- Model versioning
- Pipeline reproducibility
- Experiment reproducibility

---

## FastAPI Deployment

FastAPI serves trained models through REST APIs.

### Available Endpoints

| Endpoint | Description |
|---|---|
| `/predict_qlearning` | Q-Learning prediction |
| `/predict_ppo` | PPO prediction |
| `/compare_models` | Model comparison |

---

## Docker Containerization

Docker is used for:
- Portable deployment
- Dependency isolation
- Reproducible execution

---

## GitHub Actions CI/CD

CI/CD automates:
- Dependency installation
- Script verification
- DVC pipeline validation
- Docker image build validation

---

## Apache Airflow DAG

Airflow is used for:
- Pipeline orchestration
- Task dependency management
- Automated workflow execution

---

# Steps to Run the Project

## 1. Clone Repository

```bash
git clone https://github.com/shinchana1011/smart_taxi_dispatch_optimization.git
cd smart_taxi_dispatch_optimization
```

---

## 2. Create Virtual Environment

```bash
python3 -m venv venv
source venv/bin/activate
```

---

## 3. Install Dependencies

```bash
pip install -r requirements.txt
```

---

## 4. Run Q-Learning Training

```bash
python -m experiments.train --config configs/qlearning_v2_explored.yaml
```

---

## 5. Run PPO Training

```bash
python -m sim.ppo_agent
```

---

## 6. Generate Evaluation Plots

```bash
python experiments/plot_results.py
```

---

## 7. Run FastAPI Application

```bash
uvicorn api.app:app --reload
```

Open:

```text
http://127.0.0.1:8000/docs
```

---

## 8. Run MLflow UI

```bash
mlflow ui --backend-store-uri file:./mlruns
```

Open:

```text
http://127.0.0.1:5000
```

---

## 9. Run DVC Pipeline

```bash
dvc repro
```

---

## 10. Build Docker Image

```bash
docker build -t smart-taxi-api .
```

---

## 11. Run Docker Compose

```bash
docker compose up --build
```

---

## 12. Run Airflow DAG

```bash
airflow standalone
```

---

# Contributors

- Shinchana R Reddy
- Sinchana K
- Sahana K Sonni
- Shreyashree C R
- Varsha Tolani

---

# License

This project is intended for academic and educational purposes.
