import pandas as pd
import matplotlib.pyplot as plt
import argparse
import os
import numpy as np
import sys
sys.path.append(".")
from libs.tools import expanded_join

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--files", type=str, nargs='+', default=None, required=True)
    parser.add_argument("--dest_folder", type=str, default="results")
    parser.add_argument("--saving_name", type=str, required=True)
    args = parser.parse_args()

    if args.files is not None:
        if isinstance(args.files, str):
            args.files = [args.files]
        
        measurements = ["Val Seen", "Val Unseen", "Val Joint", "Val Harmonic Mean"]
        models_stats = {}
        files_names = []

        for file in args.files:
            # check if file exists
            if not os.path.exists(file):
                print("Could not find file.")
            else:
                # Load CSV
                df = pd.read_csv(file)
                file_n = file.split('/')[1] # get only file name
                name = file_n[:-4]
                files_names.append(name)

                models_stats[name] = {
                    "Val Seen": df["val_seen_acc"].max(),
                    "Val Unseen": df["val_unseen_acc"].max(),
                    "Val Joint": df["val_joint_acc"].max(),
                    "Val Harmonic Mean": df["harmonic_mean_acc"].max()
                }

        # Create figure with 3 rows, 1 column
        fig, ax = plt.subplots(figsize=(12, 8))
        width = 0.25
        x = np.arange(len(measurements))   # measurements = list of metric names

        for i, file_name in enumerate(files_names):
            values = [models_stats[file_name][m] for m in measurements]
            rects = ax.bar(x + i * width, values, width, label=file_name)
            ax.bar_label(rects, padding=3, fontsize=12, fmt="%.2f")
        
        ax.set_ylabel('Accuracies', fontsize=14)
        ax.set_title('Accuracies per model', fontsize=16)

        ax.set_xticks(x + width, measurements)
        ax.tick_params(axis='x', labelsize=12)
        ax.tick_params(axis='y', labelsize=12)

        ax.legend(loc='upper left', fontsize=12)
        ax.set_ylim(0, 1)

        plt.tight_layout()
        fig.savefig(expanded_join(args.dest_folder, f"{args.saving_name}.png"))
        plt.close(fig)

        print(f"Saved image: {args.saving_name}.png")