from airflow import DAG
from airflow.operators.bash import BashOperator
from datetime import datetime

default_args = {
    "owner": "airflow",
    "depends_on_past": False,
    "start_date": datetime(2025, 1, 1),
    "retries": 1,
}

with DAG(
    dag_id="smart_taxi_dispatch_pipeline",
    default_args=default_args,
    description="RL + MLOps pipeline for Smart Taxi Dispatch Optimization",
    schedule=None,          # Manual trigger
    catchup=False,
    tags=["RL", "MLOps", "Taxi"],
) as dag:

    # ---------------------------
    # Preprocessing
    # ---------------------------

    preprocess_data = BashOperator(
        task_id="preprocess_data",
        bash_command="""
        cd ~/projects/smart_taxi_dispatch_optimization &&
        source venv/bin/activate &&
        python preprocessing/preprocess.py
        """
    )

    # ---------------------------
    # Train Q-Learning
    # ---------------------------

    train_qlearning = BashOperator(
        task_id="train_qlearning",
        bash_command="""
        cd ~/projects/smart_taxi_dispatch_optimization &&
        source venv/bin/activate &&
        python experiments/train.py \
        --config configs/qlearning_v2_explored.yaml
        """
    )

    # ---------------------------
    # Train PPO
    # ---------------------------

    train_ppo = BashOperator(
        task_id="train_ppo",
        bash_command="""
        cd ~/projects/smart_taxi_dispatch_optimization &&
        source venv/bin/activate &&
        python -m sim.ppo_agent
        """
    )

    # ---------------------------
    # Evaluate / Compare models
    # ---------------------------

    compare_models = BashOperator(
        task_id="compare_models",
        bash_command="""
        cd ~/projects/smart_taxi_dispatch_optimization &&
        source venv/bin/activate &&
        python experiments/evaluate.py
        """
    )

    # ---------------------------
    # Plot graphs
    # ---------------------------

    plot_results = BashOperator(
        task_id="plot_results",
        bash_command="""
        cd ~/projects/smart_taxi_dispatch_optimization &&
        source venv/bin/activate &&
        python experiments/plot_results.py
        """
    )

    # ---------------------------
    # Workflow order
    # ---------------------------

    (
        preprocess_data
        >> train_qlearning
        >> train_ppo
        >> compare_models
        >> plot_results
    )