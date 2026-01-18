import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt

import sys
sys.path.append(".")
from libs.cub import Cub200, Cub200Dataset
from libs.tools import expanded_join


def plot_dist(data: list, unique_labels, fig_name, saving_path):
    count = np.zeros_like(unique_labels)

    for i, clas in enumerate(unique_labels):
        count[i] = data.count(clas)

    print(count[:15])
    fig, ax1 = plt.subplots(figsize=(8,6))
    fig.suptitle(f'Dataset {fig_name} classes distribution')

    ax1.set_xlabel('classes')
    ax1.set_ylabel('Instances')
    ax1.bar(unique_labels, count)
    ax1.grid(True)
    fig.tight_layout() 

    # Save figure
    plt.savefig(expanded_join(saving_path, f"{fig_name}.png"))
    print('Saved plot to', expanded_join(saving_path, f"{fig_name}.png"))
    plt.close(fig)

cub_dataset = Cub200()
_, train_labels = cub_dataset.get_training_set()
_, val_labels = cub_dataset.get_validation_set()
_, test_labels = cub_dataset.get_training_set()

unique_train = np.unique(train_labels)
plot_dist(list(train_labels), unique_train, "train", "results")
