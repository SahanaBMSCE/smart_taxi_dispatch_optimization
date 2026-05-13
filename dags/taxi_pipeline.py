from airflow import DAG
from airflow.providers.standard.operators.bash import BashOperator
from datetime import datetime

default_args = {
    "owner": "sahana",
    "start_date": datetime(2025, 1, 1),
}

with DAG(

    dag_id="smart_taxi_dispatch_pipeline",

    default_args=default_args,

    schedule=None,

    catchup=False,

    tags=["mlops", "reinforcement-learning"]

) as dag:

    preprocess = BashOperator(

        task_id="preprocess_data",

        bash_command="""
        cd ~/projects/smart_taxi_dispatch_optimization &&
        source venv/bin/activate &&
        python -m preprocessing.preprocess
        """
    )

    train_qlearning = BashOperator(

        task_id="train_qlearning",

        bash_command="""
        cd ~/projects/smart_taxi_dispatch_optimization &&
        source venv/bin/activate &&
        python -m experiments.train --config configs/qlearning_v2_explored.yaml
        """
    )

    train_ppo = BashOperator(

        task_id="train_ppo",

        bash_command="""
        cd ~/projects/smart_taxi_dispatch_optimization &&
        source venv/bin/activate &&
        python -m sim.ppo_agent
        """
    )

    compare_models = BashOperator(

        task_id="compare_models",

        bash_command="""
        cd ~/projects/smart_taxi_dispatch_optimization &&
        source venv/bin/activate &&
        python experiments/compare_models.py
        """
    )

    preprocess >> train_qlearning >> train_ppo >> compare_models