import argparse
import numpy as np
import os
from data_generation import generate_dataset, save_dataset, plot_sample_trajectory
from train import train_model
from test import evaluate_dataset, predict_single_trajectory

CSV_PATH = "dataset.csv"
TIME_STEPS = 200
DURATION = 10.0


def main():
    parser = argparse.ArgumentParser(description="Trajectory Classification Pipeline")
    parser.add_argument("--generate", action="store_true", help="Generate new dataset")
    parser.add_argument("--append", action="store_true", help="Append new data to existing dataset")
    parser.add_argument("--train", action="store_true", help="Train the model")
    parser.add_argument("--test", action="store_true", help="Evaluate model on test dataset")
    parser.add_argument("--predict", action="store_true", help="Predict a single new trajectory")
    args = parser.parse_args()

    t = np.linspace(0, DURATION, TIME_STEPS)

    if args.generate:
        print("📦 Generating new dataset...")
        df = generate_dataset(num_samples=1000, t=t)
        save_dataset(df, CSV_PATH)
        plot_sample_trajectory(t, np.array(df.iloc[0]["x"]), "sample_plot.png")

    elif args.append:
        print("➕ Appending to existing dataset...")
        if not os.path.exists(CSV_PATH):
            raise FileNotFoundError("No existing dataset to append to.")
        df_old = pd.read_csv(CSV_PATH)
        df_new = generate_dataset(num_samples=500, t=t)
        df_all = pd.concat([df_old, df_new], ignore_index=True)
        save_dataset(df_all, CSV_PATH)

    if args.train:
        print("🧠 Starting training...")
        train_model(CSV_PATH, time_steps=TIME_STEPS)

    if args.test:
        print("🧪 Running evaluation...")
        evaluate_dataset(CSV_PATH, time_steps=TIME_STEPS)

    if args.predict:
        print("🔮 Predicting a new trajectory...")
        # Example: user supplies their own x, v, a arrays here
        # Replace with real values when needed
        x = np.sin(2 * np.pi * np.linspace(0, 1, TIME_STEPS))
        v = np.gradient(x, t[1] - t[0])
        a = np.gradient(v, t[1] - t[0])
        predict_single_trajectory(x, v, a, time_steps=TIME_STEPS)


if __name__ == "__main__":
    main()
