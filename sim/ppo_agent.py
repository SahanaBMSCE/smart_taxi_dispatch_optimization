from stable_baselines3 import PPO
from sim.ppo_env import PPOTaxiEnv

import mlflow
import numpy as np
import matplotlib.pyplot as plt
import os
import json

mlflow.set_tracking_uri("file:./mlruns")


def evaluate_model(env, model, episodes=50):

    rewards = []
    waiting_times = []

    for _ in range(episodes):

        obs, _ = env.reset()

        done = False

        total_reward = 0
        total_wait = 0
        steps = 0

        while not done:

            action, _ = model.predict(obs)

            obs, reward, terminated, truncated, info = env.step(action)

            done = terminated or truncated

            total_reward += reward

            total_wait += info["wait_time"]

            steps += 1

        rewards.append(total_reward)

        waiting_times.append(total_wait / steps)

    return rewards, waiting_times


def train_ppo():

    os.makedirs("models/ppo", exist_ok=True)
    os.makedirs("plots", exist_ok=True)
    os.makedirs("results", exist_ok=True)

    env = PPOTaxiEnv()

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
        mlflow.log_param("gamma", 0.99)
        mlflow.log_param("batch_size", 64)

        # -----------------------------
        # PPO TRAINING
        # -----------------------------

        model.learn(total_timesteps=50000)

        # -----------------------------
        # PPO EPISODE REWARD HISTORY
        # -----------------------------

        reward_history = []

        for episode in range(500):

            obs, _ = env.reset()

            done = False

            episode_reward = 0

            while not done:

                action, _ = model.predict(obs)

                obs, reward, terminated, truncated, info = env.step(action)

                done = terminated or truncated

                episode_reward += reward

            reward_history.append(episode_reward)

        # -----------------------------
        # PPO EVALUATION
        # -----------------------------

        rewards, waiting_times = evaluate_model(
            env,
            model
        )

        avg_reward = float(np.mean(rewards))

        avg_wait_time = float(np.mean(waiting_times))

        mlflow.log_metric(
            "average_reward",
            avg_reward
        )

        mlflow.log_metric(
            "average_waiting_time",
            avg_wait_time
        )

        # -----------------------------
        # SAVE PPO MODEL
        # -----------------------------

        model.save(
            "models/ppo/ppo_dispatch"
        )

        # -----------------------------
        # SAVE PPO RESULTS JSON
        # -----------------------------

        results = {

            "run_id": "ppo_dispatch",

            "algorithm": "PPO",

            "average_reward_last_50": avg_reward,

            "average_waiting_time_last_50": avg_wait_time,

            "parameters": {

                "learning_rate": 0.0003,

                "gamma": 0.99,

                "batch_size": 64
            },

            "policy_file": "models/ppo/ppo_dispatch.zip"
        }

        with open(
            "results/results_ppo.json",
            "w"
        ) as f:

            json.dump(
                results,
                f,
                indent=4
            )

        # -----------------------------
        # PPO REWARD CURVE
        # -----------------------------

        window = 20

        smoothed = np.convolve(
            reward_history,
            np.ones(window) / window,
            mode="valid"
        )

        plt.figure(figsize=(10, 5))

        plt.plot(
            reward_history,
            alpha=0.3,
            color="lightgreen",
            label="Raw reward"
        )

        plt.plot(
            range(window - 1, len(reward_history)),
            smoothed,
            color="green",
            linewidth=2,
            label=f"{window}-episode moving avg"
        )

        plt.xlabel("Episode")

        plt.ylabel("Total Reward")

        plt.title("Reward Curve - PPO")

        plt.legend()

        plt.tight_layout()

        plt.savefig(
            "plots/reward_curve_ppo.png",
            dpi=300
        )

        plt.close()

        # -----------------------------
        # LOG ARTIFACTS
        # -----------------------------

        mlflow.log_artifact(
            "plots/reward_curve_ppo.png"
        )

        mlflow.log_artifact(
            "results/results_ppo.json"
        )

        print("\nPPO Training completed")
        print("Average reward:", avg_reward)
        print("Average waiting time:", avg_wait_time)


if __name__ == "__main__":
    train_ppo()