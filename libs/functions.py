import torch
import torch.nn as nn
import torchvision.models as models
import pandas as pd
import numpy as np
import sys
sys.path.append("../")

from libs.mnsampler import split_support_query

@torch.no_grad()
def compute_prototypes(embeddings, labels):
    prototypes = {}
    for clas in torch.unique(labels):
        cls_emb = embeddings[labels == clas]
        proto = cls_emb.mean(dim=0)
        proto = torch.nn.functional.normalize(proto, dim=0)
        prototypes[int(clas.item())] = proto
    return prototypes


@torch.no_grad()
def compute_seen_class_prototypes(model, dataloader, device):
    """
    Compute class prototypes for seen classes using the entire training set.

    dataloader: standard DataLoader yielding (x, y)
    returns: dict {class_id: prototype_embedding}
    """
    model.eval()

    embeddings_per_class = {}

    for x, y in dataloader:
        x = x.to(device)
        y = y.to(device)

        emb = model(x)

        for clas in torch.unique(y):
            clas = int(clas.item())
            cls_emb = emb[y == clas]

            if clas not in embeddings_per_class:
                embeddings_per_class[clas] = []
            embeddings_per_class[clas].append(cls_emb)

    # Average embeddings per class
    prototypes = {}
    for clas, emb_list in embeddings_per_class.items():
        all_embs = torch.cat(emb_list, dim=0)
        proto = all_embs.mean(dim=0)
        proto = torch.nn.functional.normalize(proto, dim=0)
        prototypes[clas] = proto

    return prototypes

@torch.no_grad()
def compute_seen_class_prototypes_quantized(model, dataloader, device, model_dtype=None):
    """
    Compute class prototypes for seen classes using the entire training set.

    dataloader: standard DataLoader yielding (x, y)
    returns: dict {class_id: prototype_embedding}
    """
    model.eval()

    embeddings_per_class = {}

    for x, y in dataloader:
        if model_dtype == None:
            x = x.to(device)
        else:
            x = x.to(dtype = model_dtype, device = device)
        y = y.to(device)

        torch.compiler.cudagraph_mark_step_begin()
        emb = model(x).clone()
        if model_dtype != None:
            emb = emb.to(dtype = torch.float32)

        for clas in torch.unique(y):
            clas = int(clas.item())
            cls_emb = emb[y == clas]

            if clas not in embeddings_per_class:
                embeddings_per_class[clas] = []
            embeddings_per_class[clas].append(cls_emb)

    # Average embeddings per class
    prototypes = {}
    for clas, emb_list in embeddings_per_class.items():
        all_embs = torch.cat(emb_list, dim=0)
        proto = all_embs.mean(dim=0)
        proto = torch.nn.functional.normalize(proto, dim=0)
        prototypes[clas] = proto

    return prototypes

def cosine_logits(query_embeddings, prototypes):
    proto_keys = list(prototypes.keys())
    proto_tensor = torch.stack([prototypes[k] for k in proto_keys])  # (C, D)

    # cosine similarity = dot product because everything is normalized
    logits = query_embeddings @ proto_tensor.T  # (Q, C)
    return logits, proto_keys


def train_few_shot(
    model,
    optimizer,
    episodic_loader,
    device,
    epochs=10
):
    model.train()
    criterion = nn.CrossEntropyLoss()

    for epoch in range(epochs):
        total_loss = 0.0

        for episode in episodic_loader:
            support_x, support_y, query_x, query_y = episode

            support_x = support_x.to(device)
            support_y = support_y.to(device)
            query_x = query_x.to(device)
            query_y = query_y.to(device)

            optimizer.zero_grad()
            support_emb = model(support_x)
            query_emb = model(query_x)

            prototypes = compute_prototypes(support_emb, support_y)
            logits, proto_keys = distance_logits(query_emb, prototypes)

            # remap labels to prototype indices
            label_map = {clas: i for i, clas in enumerate(proto_keys)}
            mapped_query_y = torch.tensor(
                [label_map[int(y.item())] for y in query_y],
                device=device
            )

            loss = criterion(logits, mapped_query_y)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        print(f"Epoch {epoch+1}: Loss = {total_loss / len(episodic_loader):.4f}")


def compute_accuracy(logits, labels):
    preds = logits.argmax(dim=1)
    return (preds == labels).float().mean().item()

def compute_harmonic_mean_accuracy(seen_acc, unseen_acc):
    return 2 * seen_acc * unseen_acc / (seen_acc + unseen_acc) if seen_acc > 0 and unseen_acc > 0 else 0


@torch.no_grad()
def evaluate_seen(model, dataloader, class_prototypes, device):
    model.eval()
    accs = []
    norm_factor = 1/len(class_prototypes)

    # labels_list = {"Batch": [], "labels": [], "mapped_y": [], "predictions": []}

    for batch, (x, y) in enumerate(dataloader):
        lb = [i.item() for i in y]

        x, y = x.to(device), y.to(device)
        emb = model(x)

        logits, proto_keys = cosine_logits(emb, class_prototypes)
        label_map = {clas: i for i, clas in enumerate(proto_keys)}
        mapped_y = torch.tensor(
            [label_map[int(lbl.item())] for lbl in y],
            device=device
        )
        '''batch_list = [batch for i in mapped_y]
        mapped = [i.item() for i in mapped_y]
        predictions = logits.argmax(dim=1)

        for i in range(len(mapped)):
            labels_list["Batch"].append(batch_list[i])
            labels_list["labels"].append(lb[i])
            labels_list["mapped_y"].append(mapped[i])
            labels_list["predictions"].append(predictions[i].item())'''
        
        accs.append(compute_accuracy(logits, mapped_y))

    '''df = pd.DataFrame(labels_list)
    df.to_csv("debug/seen_batches.csv", index=False)'''
    return sum(accs) / len(accs), (sum(accs) / len(accs)) / norm_factor
@torch.no_grad()
def evaluate_unseen(model, episodic_loader, n_way, k_shot, n_query, device):
    model.eval()
    accs = []

    # labels_list = {"Episode": [], "labels": [], "mapped_y": [], "predictions": []}

    for episode, (x, y) in enumerate(episodic_loader):
        
        support_x, support_y, query_x, query_y = split_support_query(x, y, n_way, k_shot, n_query)
        lb = [i.item() for i in query_y]
        norm_factor = 1/len(np.unique(lb))

        support_x, support_y = support_x.to(device), support_y.to(device)
        query_x, query_y = query_x.to(device), query_y.to(device)

        support_emb = model(support_x)
        query_emb = model(query_x)

        prototypes = compute_prototypes(support_emb, support_y)
        # print(f"unseen prototypes: {prototypes}")
        logits, proto_keys = cosine_logits(query_emb, prototypes)

        label_map = {clas: i for i, clas in enumerate(proto_keys)}
        mapped_query_y = torch.tensor(
            [label_map[int(y.item())] for y in query_y],
            device=device
        )
        '''batch_list = [episode for i in mapped_query_y]
        mapped = [i.item() for i in mapped_query_y]
        predictions = logits.argmax(dim=1)

        for i in range(len(mapped)):
            labels_list["Episode"].append(batch_list[i])
            labels_list["labels"].append(lb[i])
            labels_list["mapped_y"].append(mapped[i])
            labels_list["predictions"].append(predictions[i].item())'''
        
        accs.append(compute_accuracy(logits, mapped_query_y))
    
    '''df = pd.DataFrame(labels_list)
    df.to_csv("debug/unseen_batches.csv", index=False)'''
    return sum(accs) / len(accs), (sum(accs) / len(accs)) / norm_factor


@torch.no_grad()
def evaluate_joint(
    model,
    seen_prototypes,
    n_way,
    k_shot,
    n_query,
    seen_class_ids,
    episodic_loader_joint,
    device,
    gamma=0.1
):
    model.eval()
    accs = []

    # labels_list = {"Episode": [], "labels": [], "mapped_y": [], "predictions": []}

    for episode, (x, y) in enumerate(episodic_loader_joint):
        
        support_x, support_y, query_x, query_y = split_support_query(x, y, n_way, k_shot, n_query, seen_class_ids)
        lb = [i.item() for i in query_y]
        norm_factor = 1/len(np.unique(lb))

        query_x, query_y = query_x.to(device), query_y.to(device)

        if len(support_x) > 0:
            support_x, support_y = support_x.to(device), support_y.to(device)
            # Compute unseen prototypes
            support_emb = model(support_x)
            unseen_prototypes = compute_prototypes(support_emb, support_y)

            # Merge prototypes
            joint_prototypes = {**seen_prototypes, **unseen_prototypes}
        
        else:
            joint_prototypes = {**seen_prototypes}

        query_emb = model(query_x)
        logits, proto_keys = cosine_logits(query_emb, joint_prototypes)

        # apply calibration
        for i, clas in enumerate(proto_keys):
            if clas in seen_class_ids:
                logits[:, i] -= gamma

        label_map = {clas: i for i, clas in enumerate(proto_keys)}
        mapped_query_y = torch.tensor(
            [label_map[int(y.item())] for y in query_y],
            device=device
        )
        '''batch_list = [episode for i in mapped_query_y]
        mapped = [i.item() for i in mapped_query_y]
        predictions = logits.argmax(dim=1)

        for i in range(len(mapped)):
            labels_list["Episode"].append(batch_list[i])
            labels_list["labels"].append(lb[i])
            labels_list["mapped_y"].append(mapped[i])
            labels_list["predictions"].append(predictions[i].item())'''
        
        accs.append(compute_accuracy(logits, mapped_query_y))
    
    '''df = pd.DataFrame(labels_list)
    df.to_csv("debug/joint_batches.csv", index=False)'''
    return sum(accs) / len(accs), (sum(accs) / len(accs)) / norm_factor

def evaluate_model(model,
    seen_loader,
    unseen_episodic_loader,
    joint_episodic_loader,
    seen_prototypes,
    n_way,
    k_shot,
    n_query,
    seen_class_ids,
    device,
    gamma=0.1
):

    # Evaluate
    seen_acc, norm_seen = evaluate_seen(model, seen_loader, seen_prototypes, device)
    unseen_acc, norm_unseen = evaluate_unseen(model, unseen_episodic_loader, n_way, k_shot, n_query, device)
    joint_acc, norm_joint = evaluate_joint(model, seen_prototypes, n_way, k_shot, n_query, seen_class_ids, joint_episodic_loader, device, gamma)
    harm_mean_acc = compute_harmonic_mean_accuracy(seen_acc, unseen_acc)

    return seen_acc, unseen_acc, joint_acc, norm_seen, norm_unseen, norm_joint, harm_mean_acc



@torch.no_grad()
def evaluate_seen_quantized(model, dataloader, class_prototypes, device, quantization_type=None):
    model.eval()
    accs = []
    norm_factor = 1/len(class_prototypes)

    # labels_list = {"Batch": [], "labels": [], "mapped_y": [], "predictions": []}

    for batch, (x, y) in enumerate(dataloader):
        lb = [i.item() for i in y]

        if quantization_type == None:
            x = x.to(device)
        else:
            x = x.to(dtype = quantization_type, device = device)
        
        y = y.to(device)
        emb = model(x)
        if quantization_type != None:
            emb = emb.to(dtype=torch.float32)

        logits, proto_keys = cosine_logits(emb, class_prototypes)
        label_map = {clas: i for i, clas in enumerate(proto_keys)}
        mapped_y = torch.tensor(
            [label_map[int(lbl.item())] for lbl in y],
            device=device
        )
        '''batch_list = [batch for i in mapped_y]
        mapped = [i.item() for i in mapped_y]
        predictions = logits.argmax(dim=1)

        for i in range(len(mapped)):
            labels_list["Batch"].append(batch_list[i])
            labels_list["labels"].append(lb[i])
            labels_list["mapped_y"].append(mapped[i])
            labels_list["predictions"].append(predictions[i].item())'''
        
        accs.append(compute_accuracy(logits, mapped_y))

    '''df = pd.DataFrame(labels_list)
    df.to_csv("debug/seen_batches.csv", index=False)'''
    return sum(accs) / len(accs), (sum(accs) / len(accs)) / norm_factor

@torch.no_grad()
def evaluate_unseen_quantized(model, episodic_loader, n_way, k_shot, n_query, device, quantization_type=None):
    model.eval()
    accs = []

    # labels_list = {"Episode": [], "labels": [], "mapped_y": [], "predictions": []}

    for episode, (x, y) in enumerate(episodic_loader):
        
        support_x, support_y, query_x, query_y = split_support_query(x, y, n_way, k_shot, n_query)
        lb = [i.item() for i in query_y]
        norm_factor = 1/len(np.unique(lb))

        if quantization_type == None:
            support_x = support_x.to(device)
            query_x = query_x.to(device)
        else:
            support_x = support_x.to(dtype = quantization_type, device = device)
            query_x = query_x.to(dtype = quantization_type, device = device)
        
        support_y = support_y.to(device)
        query_y = query_y.to(device)

        torch.compiler.cudagraph_mark_step_begin()
        support_emb = model(support_x).clone()

        torch.compiler.cudagraph_mark_step_begin()
        query_emb = model(query_x).clone()

        if quantization_type != None:
            support_emb = support_emb.to(dtype=torch.float32)
            query_emb = query_emb.to(dtype=torch.float32)

        prototypes = compute_prototypes(support_emb, support_y)
        # print(f"unseen prototypes: {prototypes}")
        logits, proto_keys = cosine_logits(query_emb, prototypes)

        label_map = {clas: i for i, clas in enumerate(proto_keys)}
        mapped_query_y = torch.tensor(
            [label_map[int(y.item())] for y in query_y],
            device=device
        )
        '''batch_list = [episode for i in mapped_query_y]
        mapped = [i.item() for i in mapped_query_y]
        predictions = logits.argmax(dim=1)

        for i in range(len(mapped)):
            labels_list["Episode"].append(batch_list[i])
            labels_list["labels"].append(lb[i])
            labels_list["mapped_y"].append(mapped[i])
            labels_list["predictions"].append(predictions[i].item())'''
        
        accs.append(compute_accuracy(logits, mapped_query_y))
    
    '''df = pd.DataFrame(labels_list)
    df.to_csv("debug/unseen_batches.csv", index=False)'''
    return sum(accs) / len(accs), (sum(accs) / len(accs)) / norm_factor


@torch.no_grad()
def evaluate_joint_quantized(
    model,
    seen_prototypes,
    n_way,
    k_shot,
    n_query,
    seen_class_ids,
    episodic_loader_joint,
    device,
    gamma=0.1,
    quantization_type=None
):
    model.eval()
    accs = []

    # labels_list = {"Episode": [], "labels": [], "mapped_y": [], "predictions": []}

    for episode, (x, y) in enumerate(episodic_loader_joint):
        
        support_x, support_y, query_x, query_y = split_support_query(x, y, n_way, k_shot, n_query, seen_class_ids)
        lb = [i.item() for i in query_y]
        norm_factor = 1/len(np.unique(lb))

        if quantization_type == None:
            query_x = query_x.to(device)
        else:
            query_x = query_x.to(dtype = quantization_type, device = device)

        query_y = query_y.to(device)

        if len(support_x) > 0:

            if quantization_type == None:
                support_x = support_x.to(device)
            else:
                support_x = support_x.to(dtype = quantization_type, device = device)

            support_y = support_y.to(device)
            # Compute unseen prototypes
            torch.compiler.cudagraph_mark_step_begin()
            support_emb = model(support_x).clone()

            if quantization_type != None:
                support_emb = support_emb.to(dtype=torch.float32)

            unseen_prototypes = compute_prototypes(support_emb, support_y)

            # Merge prototypes
            joint_prototypes = {**seen_prototypes, **unseen_prototypes}
        
        else:
            joint_prototypes = {**seen_prototypes}
        
        torch.compiler.cudagraph_mark_step_begin()
        query_emb = model(query_x).clone()

        if quantization_type != None:
            query_emb = query_emb.to(dtype=torch.float32)

        logits, proto_keys = cosine_logits(query_emb, joint_prototypes)

        # apply calibration
        for i, clas in enumerate(proto_keys):
            if clas in seen_class_ids:
                logits[:, i] -= gamma

        label_map = {clas: i for i, clas in enumerate(proto_keys)}
        mapped_query_y = torch.tensor(
            [label_map[int(y.item())] for y in query_y],
            device=device
        )
        '''batch_list = [episode for i in mapped_query_y]
        mapped = [i.item() for i in mapped_query_y]
        predictions = logits.argmax(dim=1)

        for i in range(len(mapped)):
            labels_list["Episode"].append(batch_list[i])
            labels_list["labels"].append(lb[i])
            labels_list["mapped_y"].append(mapped[i])
            labels_list["predictions"].append(predictions[i].item())'''
        
        accs.append(compute_accuracy(logits, mapped_query_y))
    
    '''df = pd.DataFrame(labels_list)
    df.to_csv("debug/joint_batches.csv", index=False)'''
    return sum(accs) / len(accs), (sum(accs) / len(accs)) / norm_factor

def evaluate_model_quantized(model,
    seen_loader,
    unseen_episodic_loader,
    joint_episodic_loader,
    seen_prototypes,
    n_way,
    k_shot,
    n_query,
    seen_class_ids,
    device,
    gamma=0.1,
    quantization_type=None
):

    # Evaluate
    seen_acc, norm_seen = evaluate_seen_quantized(model, seen_loader, seen_prototypes, device, quantization_type=quantization_type)
    unseen_acc, norm_unseen = evaluate_unseen_quantized(model, unseen_episodic_loader, n_way, k_shot, n_query, device, quantization_type=quantization_type)
    joint_acc, norm_joint = evaluate_joint_quantized(model, seen_prototypes, n_way, k_shot, n_query, seen_class_ids, joint_episodic_loader, device, gamma, quantization_type=quantization_type)
    harm_mean_acc = compute_harmonic_mean_accuracy(seen_acc, unseen_acc)

    return seen_acc, unseen_acc, joint_acc, norm_seen, norm_unseen, norm_joint, harm_mean_acc
