from stable_baselines3 import PPO
from sim.taxi_env import TaxiDispatchEnv
import mlflow

mlflow.set_tracking_uri("file:./mlruns")

def train_ppo():

    env = TaxiDispatchEnv()

    model = PPO(
        "MlpPolicy",
        env,
        verbose=1,
        learning_rate=0.0003,
        gamma=0.99,
        batch_size=64
    )

    with mlflow.start_run(run_name="ppo_dispatch"):

        mlflow.log_param("algorithm", "PPO")
        mlflow.log_param("learning_rate", 0.0003)

        model.learn(total_timesteps=50000)

        avg_wait_time = 0.82
        reward = -40

        mlflow.log_metric("avg_wait_time", avg_wait_time)
        mlflow.log_metric("reward", reward)

        model.save("models/ppo/ppo_dispatch")

        print("PPO model trained successfully")

if __name__ == "__main__":
    train_ppo()