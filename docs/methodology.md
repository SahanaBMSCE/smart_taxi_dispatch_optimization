
---

# 2. methodology.md

```md
# Methodology

# Introduction

This project focuses on optimizing taxi dispatch operations using Reinforcement Learning and modern MLOps practices.

The methodology combines:
- Simulation-based RL training
- Experiment tracking
- Reproducible pipelines
- Containerized deployment
- Automated CI/CD workflows

The primary goal is to minimize passenger waiting time while demonstrating a complete end-to-end machine learning lifecycle.

---

# SDG Alignment

The project supports the following Sustainable Development Goals (SDGs):

## SDG 9 — Industry, Innovation and Infrastructure
- AI-enabled intelligent transportation systems
- Scalable ML deployment pipelines
- Smart infrastructure optimization

## SDG 11 — Sustainable Cities and Communities
- Efficient urban transportation
- Reduced passenger waiting time
- Improved mobility systems

## SDG 13 — Climate Action
- Reduced unnecessary taxi movement
- Optimized fleet dispatch
- Lower fuel consumption potential

---

# System Workflow

The workflow consists of the following stages:

1. Taxi dispatch simulation
2. Baseline evaluation
3. Q-Learning training
4. PPO training
5. Model evaluation
6. MLflow experiment tracking
7. DVC versioning
8. FastAPI deployment
9. Docker containerization
10. GitHub Actions CI/CD automation

---

# Taxi Dispatch Environment

A custom grid-based taxi simulation environment was developed.

## Environment Components

- Taxi fleet
- Passenger pickup locations
- Waiting time computation
- Dispatch actions
- Reward generation

The environment simulates dispatch decisions and evaluates the efficiency of selected policies.

---

# Baseline Strategy

The baseline strategy dispatches the nearest available taxi to the passenger.

## Characteristics

- Deterministic heuristic
- No learning capability
- Used as benchmark

The baseline provides a reference waiting time against which RL performance is evaluated.

---

# Q-Learning Methodology

Q-Learning was implemented as the first Reinforcement Learning approach.

## Q-Learning Equation

```text
Q(s,a) = Q(s,a) + α[r + γ max Q(s',a') − Q(s,a)]