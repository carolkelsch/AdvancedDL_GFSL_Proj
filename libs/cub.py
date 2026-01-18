
import os
import numpy as np
from PIL import Image
import sys
sys.path.append("../")

from libs.tools import expanded_join
from torch.utils.data import Dataset

class Cub200:
    """ Caltech-UCSD Birds 200 (Cub-200-2011) dataset wrapper for retrieval task. Refs:
    http://www.vision.caltech.edu/visipedia/CUB-200-2011.html

    :return A Dataset instance for Cub-200-2011 dataset (retrieval task).
    """

    LOCAL_ROOT = "/home/pjacob/RESOURCES/CUB_200_2011/CUB_200_2011" # adapt to your usage
    CREMI_ROOT = "/net/ens/atcv/datasets/CUB_200_2011"

    def __init__(self):
        np.random.seed(42)

        if os.path.exists(self.LOCAL_ROOT):
            self.root_path = self.LOCAL_ROOT
        elif os.path.exists(self.CREMI_ROOT):
            self.root_path = self.CREMI_ROOT
        else:
            raise FileNotFoundError("Cannot access to CUB dataset. Check if given paths are correct."
                                    f"Got '{self.root_path}'")

        x1, x2, y1, y2, self.labels, self.images = self._parse_annotations()

        # Split train/val/test as follows:
        #   1. split seen/unseen classes: 130 / 20 / 50 classes
        #   2. Samples 10% (resp. 25%) of seen classes for the validation set (resp. the testing set)
        unseen_val_ind = np.where(np.logical_and(self.labels > 130, self.labels <= 150))[0]
        unseen_test_ind = np.where(self.labels > 150)[0]

        # Seen class split:
        train_idx = []
        val_idx = []
        test_idx = []
        for lbl in range(1, 131):
            cls_idx = np.where(self.labels == lbl)[0]
            np.random.shuffle(cls_idx)
            start_val = int(0.65 * len(cls_idx))
            start_test = int(0.75 * len(cls_idx))
            train_idx.append(cls_idx[0:start_val])
            val_idx.append(cls_idx[start_val:start_test])
            test_idx.append(cls_idx[start_test:])

        val_idx.append(unseen_val_ind)
        test_idx.append(unseen_test_ind)

        self.train_idx = np.concatenate(train_idx)
        self.val_idx = np.concatenate(val_idx)
        self.test_idx = np.concatenate(test_idx)


    def get_training_set(self):
        return self.images[self.train_idx], self.labels[self.train_idx]

    def get_validation_set(self):
        return self.images[self.val_idx], self.labels[self.val_idx]

    def get_testing_set(self):
        return self.images[self.test_idx], self.labels[self.test_idx]

    def _parse_annotations(self):
        images = []
        labels = []
        x1 = []
        x2 = []
        y1 = []
        y2 = []

        with open(expanded_join(self.root_path, 'images.txt')) as f:
            for line in f:
                split_l = line.split(' ')
                images.append(expanded_join(self.root_path, 'images', split_l[1][:-1]))

        with open(expanded_join(self.root_path, 'image_class_labels.txt')) as f:
            for line in f:
                labels.append(int(line.split(' ')[1]))

        with open(expanded_join(self.root_path, 'bounding_boxes.txt')) as f:
            for line in f:
                split_l = line.split(' ')
                x1.append(float(split_l[1]))
                y1.append(float(split_l[2]))
                x2.append(float(split_l[3]) + x1[-1])
                y2.append(float(split_l[4][:-1]) + y1[-1])

        images = np.array(images)
        labels = np.array(labels, dtype=np.int32)
        x1 = np.array(x1, dtype=np.int32)
        y1 = np.array(y1, dtype=np.int32)
        x2 = np.array(x2, dtype=np.int32)
        y2 = np.array(y2, dtype=np.int32)

        return x1, x2, y1, y2, labels, images

class Cub200Dataset(Dataset):
    def __init__(self, images, labels, transform=None, target_transform=None):
        super().__init__()
        self.images = images
        self.labels = labels
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, item):
        with Image.open(self.images[item]) as im:
            img = im.convert("RGB")
        if self.transform is not None:
            img = self.transform(img)

        label = self.labels[item]
        if self.target_transform is not None:
            label = self.target_transform(label)
        return img, label
