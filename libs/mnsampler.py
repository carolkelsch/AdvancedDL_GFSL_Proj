
import numpy as np
import torch
from typing import List, Iterator, Union
from torch.utils.data import Sampler
import random

class MNSampler(Sampler[List[int]]):
    """Sampler class to generate batch indexes suitable for distance-based learning with small batches. Its goal is to
    build batches composed of M classes (uniformly sampled) and N samples (uniformly sampled) per class.

    :param n_iter: Number of iteration per epoch
    :param n_class_per_batch: Number of sampled classes per batch.
    :param n_samples_per_class: Number of samples per class.
    """
    def __init__(self, labels, n_iter: int, n_class_per_batch: int, n_samples_per_class: int) -> None:
        super().__init__()
        self.n_iter = n_iter
        self.labels = labels
        self.n_class_per_batch = n_class_per_batch
        self.n_samples_per_class = n_samples_per_class

        self.class_labels = list(set(labels))

    def __len__(self) -> int:
        return self.n_iter

    def __iter__(self) -> Iterator[List[int]]:
        for _ in range(self.n_iter):
            sampled_cls = np.random.choice(self.class_labels, size=self.n_class_per_batch, replace=False)
            batch_idx = []
            for i in sampled_cls:
                idx = np.where(self.labels == i)[0]
                np.random.shuffle(idx)
                batch_idx.append(idx[:self.n_samples_per_class])

            batch_idx = np.concatenate(batch_idx)
            yield batch_idx


class KMNSampler(Sampler[List[int]]):
    """Sampler class to generate batch indexes suitable for distance-based learning with small batches. Its goal is to
    build batches composed of M classes (uniformly sampled), K-shot support samples and N samples (uniformly sampled) per class.

    :param n_iter: Number of iteration per epoch
    :param n_class_per_batch: Number of sampled classes per batch.
    :param n_ksupport_per_batch: Number of sampled classes per batch.
    :param n_samples_per_class: Number of samples per class.
    """
    def __init__(self, labels, n_iter: int, n_class_per_batch: int, n_ksupport_per_batch: int, n_samples_per_class: int) -> None:
        super().__init__()
        self.n_iter = n_iter
        self.labels = labels
        self.n_class_per_batch = n_class_per_batch
        self.n_ksupport_per_batch = n_ksupport_per_batch
        self.n_samples_per_class = n_samples_per_class

        self.class_labels = list(set(labels))

    def __len__(self) -> int:
        return self.n_iter

    def __iter__(self) -> Iterator[List[int]]:
        for _ in range(self.n_iter):
            sampled_cls = np.random.choice(self.class_labels, size=self.n_class_per_batch, replace=False)
            batch_idx = []
            for i in sampled_cls:
                idx = np.where(self.labels == i)[0]
                np.random.shuffle(idx)
                '''if len(idx) < (self.n_ksupport_per_batch + self.n_samples_per_class):
                    print(f"Something is not right, class {i} does not have enouth data!")'''
                batch_idx.append(idx[:(self.n_ksupport_per_batch + self.n_samples_per_class)])

            batch_idx = np.concatenate(batch_idx)
            yield batch_idx


class KMNJointSampler(Sampler[List[int]]):
    """Sampler class to generate batch indexes suitable for distance-based learning with small batches. Its goal is to
    build batches composed of M classes (uniformly sampled) and N samples (uniformly sampled) per class.

    :param n_iter: Number of iteration per epoch
    :param n_class_per_batch: Number of sampled classes per batch.
    :param n_samples_per_class: Number of samples per class.
    """
    def __init__(self, labels, n_iter: int, known_ids: int, n_ksupport_per_batch: int, n_unseen_classes_per_batch: int, n_seen_classes_per_batch: int, n_samples_per_class: int) -> None:
        super().__init__()
        self.n_iter = n_iter
        self.labels = labels
        self.known_ids = known_ids
        self.n_ksupport_per_batch = n_ksupport_per_batch
        self.n_unseen_classes_per_batch = n_unseen_classes_per_batch
        self.n_seen_classes_per_batch = n_seen_classes_per_batch
        self.n_samples_per_class = n_samples_per_class

        self.unseen_class_labels = list(set(labels) - set(self.known_ids))

    def __len__(self) -> int:
        return self.n_iter

    def __iter__(self) -> Iterator[List[int]]:
        for _ in range(self.n_iter):
            sampled_cls = np.concatenate(
                [np.random.choice(self.known_ids,
                size=self.n_seen_classes_per_batch,
                replace=False),
                np.random.choice(self.unseen_class_labels,
                size=self.n_unseen_classes_per_batch,
                replace=False)]
            )
            batch_idx = []
            for i in sampled_cls:
                idx = np.where(self.labels == i)[0]
                np.random.shuffle(idx)
                if i not in self.known_ids:
                    batch_idx.append(idx[:(self.n_samples_per_class + self.n_ksupport_per_batch)])
                else:
                    batch_idx.append(idx[:self.n_samples_per_class])

            batch_idx = np.concatenate(batch_idx)
            yield batch_idx


def split_support_query(
    images,
    labels,
    n_way:int,
    k_shot:int,
    q_queries:int,
    seen_class_ids:Union[List[int], None]=None
):
    """
    Split a batch into support and query sets.

    Joint evaluation:
        - Support set contains ONLY unseen classes
        - Query set may contain seen + unseen classes

    Unseen-only evaluation:
        - All classes are treated as unseen

    images: Tensor (N, ...)
    labels: Tensor (N,)
    n_way: number of unseen classes sampled
    k_shot: number of support instances for each unseen class
    q_queries: number of instances to classify
    seen_class_ids: set or list of seen class ids, or None
    """

    support_images = []
    support_labels = []
    query_images = []
    query_labels = []

    images = list(images)
    labels = list(labels)

    # Identify unseen classes
    labels = [lbl.item() for lbl in labels]
    unique_labels = list(set(labels))

    if seen_class_ids is None:
        unseen_classes = unique_labels
    else:
        unseen_classes = [c for c in unique_labels if c not in seen_class_ids]

    # Group indices by class
    class_to_indices = {}
    for idx, lbl in enumerate(labels):
        class_to_indices.setdefault(lbl, []).append(idx)
    
    # Build support set (ONLY unseen classes)
    for clas in unseen_classes:
        indices = class_to_indices[clas]
        assert len(indices) >= k_shot, f"Not enough samples for class {clas}, only found {len(indices)}"

        chosen = random.sample(indices, k_shot)
        for idx in chosen:
            support_images.append(images[idx])
            support_labels.append(labels[idx])

        # Remove used indices so they are not reused in query
        class_to_indices[clas] = list(set(indices) - set(chosen))

    # Build query set
    for clas in unseen_classes:
        indices = class_to_indices[clas]
        chosen = random.sample(indices, min(q_queries, len(indices)))

        for idx in chosen:
            query_images.append(images[idx])
            query_labels.append(labels[idx])

    # Add seen-class samples to query set (joint evaluation)
    if seen_class_ids is not None:
        for clas in seen_class_ids:
            if clas not in class_to_indices:
                continue
            indices = class_to_indices[clas]
            chosen = random.sample(indices, min(q_queries, len(indices)))

            for idx in chosen:
                query_images.append(images[idx])
                query_labels.append(labels[idx])

    # Stack results
    # if len(support_images) > 0:
    support_images = torch.stack(support_images)
    support_labels = torch.tensor(support_labels)
    # if len(query_images) > 0:
    query_images = torch.stack(query_images)
    query_labels = torch.tensor(query_labels)

    return support_images, support_labels, query_images, query_labels