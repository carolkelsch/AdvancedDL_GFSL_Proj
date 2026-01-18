import pandas as pd
import matplotlib.pyplot as plt
import argparse
import os
import sys
sys.path.append(".")
from libs.tools import expanded_join

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--file", type=str, default=None)
    parser.add_argument("--plot_best", type=bool, default=True)
    args = parser.parse_args()

    if args.file is not None:
        # check if file exists
        if not os.path.exists(args.file):
            print("Could not find file.")
        else:
            # Load CSV
            df = pd.read_csv(args.file)

            epochs = df["Epoch"]

            # Create figure with 3 rows, 1 column
            fig, axes = plt.subplots(3, 1, figsize=(10, 12), sharex=True)

            # -------------------------
            # 1) Loss + Pos/Neg samples
            # -------------------------
            ax1 = axes[0]
            ax1.plot(epochs, df["tr_loss"], label="Train Loss")
            ax1.set_ylabel("Loss")
            ax1.set_title("Training Loss and Samples")

            ax1_right = ax1.twinx()
            ax1_right.plot(epochs, df["tr_pos"], linestyle="--", label="Positive Samples")
            ax1_right.plot(epochs, df["tr_neg"], linestyle="--", label="Negative Samples")
            ax1_right.set_ylabel("Samples")

            # Combine legends
            lines_1, labels_1 = ax1.get_legend_handles_labels()
            lines_2, labels_2 = ax1_right.get_legend_handles_labels()
            ax1.legend(lines_1 + lines_2, labels_1 + labels_2, loc="best")

            # -------------------------
            # 2) Accuracies
            # -------------------------
            ax2 = axes[1]
            ax2.plot(epochs, df["val_seen_acc"], label="Seen Acc")
            ax2.plot(epochs, df["val_unseen_acc"], label="Unseen Acc")
            ax2.plot(epochs, df["val_joint_acc"], label="Joint Acc")
            if args.plot_best:
                best_epoch_seen = df["val_seen_acc"].idxmax() + 1
                best_seen = df["val_seen_acc"].max()
                ax2.plot(best_epoch_seen, best_seen, 'ro', label="Best Seen Acc")
                best_epoch_unseen = df["val_unseen_acc"].idxmax() + 1
                best_unseen = df["val_unseen_acc"].max()
                ax2.plot(best_epoch_unseen, best_unseen, 'ro', label="Best Unseen Acc")
                best_epoch_joint = df["val_joint_acc"].idxmax() + 1
                best_joint = df["val_joint_acc"].max()
                ax2.plot(best_epoch_joint, best_joint, 'ro', label="Best Joint Acc")
            ax2.set_ylim(0, 1)
            ax2.set_ylabel("Accuracy")
            ax2.set_title("Validation Accuracies")
            ax2.legend()
            ax2.grid(True)

            # -------------------------
            # 3) Harmonic Mean
            # -------------------------
            ax3 = axes[2]
            ax3.plot(epochs, df["harmonic_mean_acc"], label="Harmonic Mean Acc")
            if args.plot_best:
                best_epoch = df["harmonic_mean_acc"].idxmax() + 1
                best_value = df["harmonic_mean_acc"].max()
                ax3.plot(best_epoch, best_value, 'ro', label="Best Harmonic Mean")
            ax3.set_ylim(0, 1)
            ax3.set_xlabel("Epoch")
            ax3.set_ylabel("Accuracy")
            ax3.set_title("Harmonic Mean Accuracy")
            ax3.legend()
            ax3.grid(True)

            plt.tight_layout()
            fig.savefig(f"{args.file[:-3]}png")
            plt.close(fig)

            print(f"Saved image: {args.file[:-3]}png")
