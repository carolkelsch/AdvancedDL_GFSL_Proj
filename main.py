

import torch
from torch.nn import Linear
from torch.utils.data import DataLoader
from torchvision.transforms.v2 import Compose, ToImage, RandomCrop, Resize, ToDtype, CenterCrop
import numpy as np
import gc
from torch.nn.utils import prune
import torch.utils.benchmark as benchmark
import pandas as pd
import matplotlib.pyplot as plt
import os
import argparse
from torchvision.models.resnet import resnet50
import sys
sys.path.append(".")

from libs.cub import Cub200, Cub200Dataset
from libs.loss import ContrastiveLossV2
from libs.mnsampler import MNSampler, KMNSampler, KMNJointSampler
from libs.metrics import active_samples
from libs.tools import expanded_join
from libs.functions import evaluate_model, evaluate_model_quantized, compute_seen_class_prototypes, compute_seen_class_prototypes_quantized, evaluate_joint
from libs.model import EmbeddingModel
from libs.utils import set_seed

 # get sparsiti ratio for each layer
def compute_sparsity_after_prune(model):
    sparsity_info = {}

    for name, module in model.named_modules():
        if isinstance(module, (torch.nn.Conv2d, torch.nn.Linear)):
            weight = module.weight.data
            num_zeros = torch.sum(weight == 0).item()
            num_elements = weight.numel()
            sparsity_ratio = num_zeros / num_elements

            sparsity_info[name] = {
                'sparsity_ratio': sparsity_ratio,
                'num_zeros': num_zeros,
                'total_elements': num_elements
            }

    return sparsity_info


# model pruning function
def prune_model_globally(args, model, sparsity):
    
    model.eval()
    print(f"\nPruning model with sparsity {sparsity}")

    parameters_to_prune = []
    for i, layer in enumerate(model.modules()):
        if isinstance(layer, (torch.nn.Conv2d, torch.nn.Linear)):
            parameters_to_prune.append((layer, 'weight'))
            print(f"layer {i}: {type(layer)} will be pruned")
            if layer.bias is not None:
                parameters_to_prune.append((layer, 'bias'))
                print(f"biases too")
    print()

    prune.global_unstructured(
        parameters_to_prune,
        pruning_method=prune.L1Unstructured,
        amount=sparsity
    )

    # compress the model weight file.
    for name, module in model.named_modules():
        if isinstance(module, (torch.nn.Conv2d, torch.nn.Linear)):
            try:
                prune.remove(module, 'weight')
                if module.bias is not None:
                    prune.remove(module, 'bias')
            except Exception as e:
                print(f"Could not remove pruning reparameterization from {name}: {e}")
                pass

    print("\nPruning complete and made permanent.")
    # Save the pruned model weights
    torch.save(model.state_dict(), expanded_join(args.models_dir, f"{args.save_name}_gpruned{int(sparsity*100)}.pth"))
    return model


def evaluate_inference_model(args,
    model,
    model_dtype,
    device,
    val_seen_loader,
    val_unseen_loader,
    val_joint_loader,
    seen_prototypes,
    seen_labels,
    print_freq=50,
    best_gamma=0.1
):

    # Evaluate accuracy on validation set
    val_loss = 0.
    val_acc = 0.
    val_ma_loss = 0.
    val_ma_acc = 0.
    
    model.eval()

    print()
    print("Validation starts...")
    
    with torch.no_grad():
        # Validation loop
        seen_acc, unseen_acc, joint_acc, seen_norm, unseen_norm, joint_norm, harm_mean_acc = evaluate_model_quantized(
            model,
            val_seen_loader,
            val_unseen_loader,
            val_joint_loader,
            seen_prototypes,
            args.n_way,
            args.k_shot,
            args.n_query,
            seen_labels,
            device,
            best_gamma,
            model_dtype
        )
        sample_batch = next(iter(val_joint_loader))
        input_sample = sample_batch[0].to(dtype=model_dtype, device=device)

        # Warmup (important for GPU + compile)
        for _ in range(5):
            _ = model(input_sample)
            torch.cuda.synchronize() if device == "cuda" else None

        # --- Measure latency using torch.utils.benchmark.Timer ---
        t = benchmark.Timer(
            stmt='model(inp)',
            setup='torch.cuda.synchronize()',  # Ensure previous CUDA ops complete
            globals={'model': model, 'inp': input_sample},
            label='Model Inference',
            sub_label='Forward pass',
            description='Latency ResNet50 forward pass'
        )

        # Run the timing
        measurement = t.timeit(10)
        avg_latency = measurement.mean * 1000  # convert to milliseconds
        median_latency = measurement.median * 1000

    # get model size in MB
    # Take any parameter to determine element size
    bytes_per_elem = next(model.parameters()).element_size()
    num_params = sum(p.numel() for p in model.parameters())
    total_size_mb = num_params * bytes_per_elem / (1024 ** 2)
    
    print(f"Model size: {total_size_mb:.2f} MB")
    print()

    return seen_acc, unseen_acc, joint_acc, seen_norm, unseen_norm, joint_norm, harm_mean_acc, avg_latency, median_latency, total_size_mb


def train_model(model,
    optimizer,
    loss_fn,
    train_loader,
    val_seen_loader,
    val_unseen_loader,
    val_joint_loader,
    epochs,
    n_way,
    k_shot,
    n_query,
    seen_labels,
    device,
    models_dir = "models",
    results_dir = "results",
    save_name = "resnet50_base"
):
    best_val_acc = 0.0
    training_hist = {"Epoch": [], "tr_loss": [], "tr_pos": [], "tr_neg": [], "val_seen_acc": [], "val_unseen_acc": [], "val_joint_acc": [], "harmonic_mean_acc": []}
    for epoch in range(epochs):
        mean_loss = 0.0
        mean_pos = 0.0
        mean_neg = 0.0
        val_mean_loss = 0.0
        val_mean_pos = 0.0
        val_mean_neg = 0.0

        model.train()
        for step, (imgs, labels) in enumerate(train_loader):
            imgs, labels = imgs.to(device), labels.to(device)

            optimizer.zero_grad()

            emb = model(imgs)

            loss = loss_fn(emb, labels)

            loss.backward()
            optimizer.step()

            mean_loss += loss.item()
            pos, neg = active_samples(emb, labels, pos_margin=0.8, neg_margin=0.3)
            mean_pos += pos.item()
            mean_neg += neg.item()       

            print(f"\rloss = {loss.item():.3f}  | pos = {pos:.0f}  | neg = {neg:.0f}", flush=True, end=" ")

        mean_loss = mean_loss / len(train_loader)
        mean_pos = mean_pos / len(train_loader)
        mean_neg = mean_neg / len(train_loader)
        print(f"\nEpoch{epoch+1}/{epochs}: loss = {mean_loss:.3f}  | pos = {mean_pos:.0f}  | neg = {mean_neg:.0f}")

        training_hist["Epoch"].append(epoch + 1)
        training_hist["tr_loss"].append(mean_loss)
        training_hist["tr_pos"].append(mean_pos)
        training_hist["tr_neg"].append(mean_neg)

        model.eval()

        # Compute seen prototypes (once)
        seen_prototypes = compute_seen_class_prototypes(model, train_loader, device)

        # if epoch == 0:
            # print(f"seen prototypes: {seen_prototypes}")

        seen_acc, unseen_acc, joint_acc, seen_norm, unseen_norm, joint_norm, harm_mean_acc = evaluate_model(
            model,
            val_seen_loader,
            val_unseen_loader,
            val_joint_loader,
            seen_prototypes,
            n_way,
            k_shot,
            n_query,
            seen_labels,
            device,
            0.1
        )

        print(f"Validation: Seen = {seen_acc:.3f}  | Unseen = {unseen_acc:.3f}  | Joint = {joint_acc:.3f}")
        print(f"Normalized: Seen = {seen_norm:.3f}  | Unseen = {unseen_norm:.3f}  | Joint = {joint_norm:.3f}")
        print(f"Harmonic Mean Accuracy: {harm_mean_acc:.3f}")

        training_hist["val_seen_acc"].append(seen_acc)
        training_hist["val_unseen_acc"].append(unseen_acc)
        training_hist["val_joint_acc"].append(joint_acc)
        training_hist["harmonic_mean_acc"].append(harm_mean_acc)
        
        if best_val_acc < joint_acc:
            # save model
            best_val_acc = joint_acc
            torch.save(model.state_dict(), expanded_join(models_dir, f"{save_name}.pth"))
        
        # save model and data every epoch
        torch.save(model.state_dict(), expanded_join(models_dir, f"{save_name}_epoch{epoch+1}.pth"))
        df = pd.DataFrame(training_hist)
        df.to_csv(expanded_join(results_dir, f"{save_name}_training.csv"), index=False)

        # delete model from previous epoch to save space
        if epoch > 0:
            prev_model_path = expanded_join(models_dir, f"{save_name}_epoch{epoch}.pth")
            if os.path.exists(prev_model_path):
                os.remove(prev_model_path)

    df = pd.DataFrame(training_hist)
    df.to_csv(expanded_join(results_dir, f"{save_name}_training.csv"), index=False)

    return training_hist, best_val_acc


if __name__ == '__main__':

    argparser = argparse.ArgumentParser()
    argparser.add_argument('--dim', type=float, default=128)
    argparser.add_argument('--weight_decay', type=float, default=1e-4)
    argparser.add_argument('--lr', type=float, default=0.01)
    argparser.add_argument('--epochs', type=int, default=50)
    argparser.add_argument('--batch_size', type=int, default=64)
    argparser.add_argument('--val_episodes', type=int, default=100)
    argparser.add_argument('--episodes', type=int, default=1000)
    argparser.add_argument('--k_shot', type=int, default=5)
    argparser.add_argument('--n_way', type=int, default=6)
    argparser.add_argument('--n_query', type=int, default=5)
    argparser.add_argument('--n_query_seen', type=int, default=2)
    argparser.add_argument('--models_dir', type=str, default="models")
    argparser.add_argument('--results_dir', type=str, default="results")
    argparser.add_argument('--save_name', type=str, required=True)
    argparser.add_argument('--g_vals', type=float, nargs='+', default=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6])
    argparser.add_argument('--quantization_type', type=str, default='fp16', choices=["fp16", "bf16"], help='Quantization type to quantize the model')
    argparser.add_argument('--best_gamma', type=float, default=0.2)
    
    argparser.add_argument('--quantize', action='store_true', help='Quantize the model')
    argparser.add_argument('--prune', action='store_true', help='Prune the model')
    argparser.add_argument('--sparsity_ratios', type=float, nargs='+', default=[0.2, 0.4, 0.5, 0.6, 0.7, 0.8])
    argparser.add_argument('--train', action='store_true', help='Train the model')
    argparser.add_argument('--eval', action='store_true', help='Evaluate the model')
    argparser.add_argument('--test_gamma', action='store_true', help='Test gamma values')
    args = argparser.parse_args()

    # set seeds
    set_seed(42)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('Using device:', device)

    train_transform = Compose([Resize((224, 224)),ToImage(), ToDtype(dtype=torch.float32)])

    print("Loading dataset")
    cub_dataset = Cub200()
    train_img, train_labels = cub_dataset.get_training_set()
    
    print("Creating model")
    model = EmbeddingModel(backbone_name="resnet50", embed_dim=int(args.dim))
    model.to(device)

    train_dataset = Cub200Dataset(train_img, train_labels, transform=train_transform)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)

    # create different validation sets
    val_img, val_labels = cub_dataset.get_validation_set()
    val_img_seen = []
    val_labels_seen = []
    val_img_unseen = []
    val_labels_unseen = []
    seen_labels = np.unique(train_labels)
    # print(f"Seen classes: {seen_labels}")

    # split validation sets into seen and unseen
    for i, label in enumerate(val_labels):
        if label in seen_labels:
            val_img_seen.append(val_img[i])
            val_labels_seen.append(label)
        else:
            val_img_unseen.append(val_img[i])
            val_labels_unseen.append(label)

    print("Splitting validation data")
    # seen validation data
    val_seen_dataset = Cub200Dataset(val_img_seen, val_labels_seen, transform=train_transform)
    val_seen_sampler = MNSampler(val_labels_seen, n_iter=args.val_episodes, n_class_per_batch=len(seen_labels), n_samples_per_class=args.n_query_seen)
    val_seen_loader = DataLoader(val_seen_dataset, batch_sampler=val_seen_sampler)

    # unseen validation data
    val_unseen_dataset = Cub200Dataset(val_img_unseen, val_labels_unseen, transform=train_transform)
    val_unseen_sampler = KMNSampler(val_labels_unseen, n_iter=args.val_episodes, n_class_per_batch=args.n_way, n_ksupport_per_batch=args.k_shot, n_samples_per_class=args.n_query)
    val_unseen_loader = DataLoader(val_unseen_dataset, batch_sampler=val_unseen_sampler)

    # joint validation data
    val_joint_dataset = Cub200Dataset(val_img, val_labels, transform=train_transform)
    val_joint_sampler = KMNJointSampler(val_labels, n_iter=args.val_episodes, known_ids=seen_labels, n_ksupport_per_batch=args.k_shot, n_unseen_classes_per_batch=args.n_way, n_seen_classes_per_batch=args.n_way, n_samples_per_class=args.n_query)
    val_joint_loader = DataLoader(val_joint_dataset, batch_sampler=val_joint_sampler)

    # create different test sets
    test_img, test_labels = cub_dataset.get_testing_set()
    test_img_seen = []
    test_labels_seen = []
    test_img_unseen = []
    test_labels_unseen = []

    # split teest sets into seen and unseen
    for i, label in enumerate(test_labels):
        if label in seen_labels:
            test_img_seen.append(test_img[i])
            test_labels_seen.append(label)
        else:
            test_img_unseen.append(test_img[i])
            test_labels_unseen.append(label)

    print("Splitting test data")
    # seen validation data
    test_seen_dataset = Cub200Dataset(test_img_seen, test_labels_seen, transform=train_transform)
    test_seen_sampler = MNSampler(test_labels_seen, n_iter=args.val_episodes, n_class_per_batch=len(seen_labels), n_samples_per_class=args.n_query_seen)
    test_seen_loader = DataLoader(test_seen_dataset, batch_sampler=test_seen_sampler)

    # unseen validation data
    test_unseen_dataset = Cub200Dataset(test_img_unseen, test_labels_unseen, transform=train_transform)
    test_unseen_sampler = KMNSampler(test_labels_unseen, n_iter=args.val_episodes, n_class_per_batch=args.n_way, n_ksupport_per_batch=args.k_shot, n_samples_per_class=args.n_query)
    test_unseen_loader = DataLoader(test_unseen_dataset, batch_sampler=test_unseen_sampler)

    # joint validation data
    test_joint_dataset = Cub200Dataset(test_img, test_labels, transform=train_transform)
    test_joint_sampler = KMNJointSampler(test_labels, n_iter=args.val_episodes, known_ids=seen_labels, n_ksupport_per_batch=args.k_shot, n_unseen_classes_per_batch=args.n_way, n_seen_classes_per_batch=args.n_way, n_samples_per_class=args.n_query)
    test_joint_loader = DataLoader(test_joint_dataset, batch_sampler=test_joint_sampler)

    if args.train:
        loss_fn = ContrastiveLossV2(pos_margin=0.8, neg_margin=0.3)
        optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=args.weight_decay)

        print("Training model")
        train_model(model,
            optimizer,
            loss_fn,
            train_loader,
            val_seen_loader,
            val_unseen_loader,
            val_joint_loader,
            args.epochs,
            args.n_way,
            args.k_shot,
            args.n_query,
            seen_labels,
            device,
            save_name=args.save_name,
            models_dir=args.models_dir,
            results_dir=args.results_dir
        )
    
    if args.test_gamma:
        # validate existing model
        print("Testing gamma values")

        # load model
        model_path = expanded_join(args.models_dir, f"{args.save_name}.pth")
        if os.path.exists(model_path):
            print(f"Loading model from {model_path}")
            model.load_state_dict(torch.load(model_path))
        
        model.eval()
        print("Computing seen prototypes")
        # Compute seen prototypes (once)
        seen_prototypes = compute_seen_class_prototypes(model, train_loader, device)
            
        best_joint_acc = 0.0
        print("\nTesting gamma in joint accuracy")
        
        for g in args.g_vals:
            print(f"Testing gamma = {g}")
            joint_acc, norm_joint = evaluate_joint(model, seen_prototypes, args.n_way, args.k_shot, args.n_query, seen_labels, val_joint_loader, device, gamma=g)
            print(f"\tValidation: Joint = {joint_acc:.3f}")
            print(f"\tNormalized: Joint = {norm_joint:.3f}")
            if joint_acc > best_joint_acc:
                best_joint_acc = joint_acc
                best_gamma = g
        
        print(f"\nBest gamma = {best_gamma} in validation")

    if args.eval:
        # validate existing model
        print("Testing model")

        if "float" in args.save_name or "float" in args.save_name:
            print("Note: Evaluating a quantized model.")

            # load model
            model_path = expanded_join(args.models_dir, f"{args.save_name}.pth")
            if os.path.exists(model_path):
                print(f"Loading model from {model_path}")
                model.load_state_dict(torch.load(model_path))
            
            print("Computing seen prototypes")

            if args.save_name.split('_')[-1] == "float16":
                model_dtype = torch.float16
            elif args.save_name.split('_')[-1] == "bfloat16":
                model_dtype = torch.bfloat16
            else:
                model_dtype = torch.float32
            
            model.to(dtype=model_dtype, device=device)
            model.eval()
            
            # Compute seen prototypes (once)
            seen_prototypes = compute_seen_class_prototypes_quantized(model, train_loader, device, model_dtype)

            seen_acc, unseen_acc, joint_acc, seen_norm, unseen_norm, joint_norm, harm_mean_acc = evaluate_model_quantized(
                model,
                test_seen_loader,
                test_unseen_loader,
                test_joint_loader,
                seen_prototypes,
                args.n_way,
                args.k_shot,
                args.n_query,
                seen_labels,
                device,
                gamma=args.best_gamma,
                quantization_type=model_dtype
            )

            print(f"\nUsing best gamma = {args.best_gamma} from validation")
            print(f"Test: Seen = {seen_acc:.3f}  | Unseen = {unseen_acc:.3f}  | Joint = {joint_acc:.3f}")
            print(f"Normalized: Seen = {seen_norm:.3f}  | Unseen = {unseen_norm:.3f}  | Joint = {joint_norm:.3f}")
            print(f"Harmonic Mean Accuracy: {harm_mean_acc:.3f}")

        else:

            # load model
            model_path = expanded_join(args.models_dir, f"{args.save_name}.pth")
            if os.path.exists(model_path):
                print(f"Loading model from {model_path}")
                model.load_state_dict(torch.load(model_path))
            
            model.eval()
            print("Computing seen prototypes")
            # Compute seen prototypes (once)
            seen_prototypes = compute_seen_class_prototypes(model, train_loader, device)

            seen_acc, unseen_acc, joint_acc, seen_norm, unseen_norm, joint_norm, harm_mean_acc = evaluate_model(
                model,
                test_seen_loader,
                test_unseen_loader,
                test_joint_loader,
                seen_prototypes,
                args.n_way,
                args.k_shot,
                args.n_query,
                seen_labels,
                device,
                args.best_gamma
            )

            print(f"\nUsing best gamma = {args.best_gamma} from validation")
            print(f"Test: Seen = {seen_acc:.3f}  | Unseen = {unseen_acc:.3f}  | Joint = {joint_acc:.3f}")
            print(f"Normalized: Seen = {seen_norm:.3f}  | Unseen = {unseen_norm:.3f}  | Joint = {joint_norm:.3f}")
            print(f"Harmonic Mean Accuracy: {harm_mean_acc:.3f}")
        
        df = pd.DataFrame({"Model" : [args.save_name], "Gamma": [args.best_gamma], "SeenAcc": [seen_acc],
            "UnseenAcc": [unseen_acc], "JointAcc": [joint_acc], "HarmonicMeanAcc": [harm_mean_acc]})
        df.to_csv(expanded_join(args.results_dir, f"{args.save_name}_test_results.csv"), index=False)
        print("Saved test results file.")

    if args.prune:
        print('Pruning model')
        results = {"Sparsity": [], "val_seen_acc": [], "val_unseen_acc": [], "val_joint_acc": [], "harmonic_mean_acc": []}
        best_acc_sparsity = 0.
        best_global_sparsity = 0

        # compute prototypes once
        # load the-pretrained model
        model_path = expanded_join(args.models_dir, f"{args.save_name}.pth")
        if os.path.exists(model_path):
            print(f"Loading model from {model_path}")
            model.load_state_dict(torch.load(model_path))
        model.to(device)
        seen_prototypes = compute_seen_class_prototypes(model, train_loader, device)

        del model
        gc.collect()
        torch.cuda.empty_cache()

        for sparsity_ratio in args.sparsity_ratios:

            model = EmbeddingModel(backbone_name="resnet50", embed_dim=int(args.dim))
            
            # load the-pretrained model
            model_path = expanded_join(args.models_dir, f"{args.save_name}.pth")
            if os.path.exists(model_path):
                print(f"Loading model from {model_path}")
                model.load_state_dict(torch.load(model_path))
            model.to(device)

            prune_model_globally(args, model, sparsity_ratio)

            # Evaluate accuracy on validation set
            print("Validating...")
            model.eval()

            seen_acc, unseen_acc, joint_acc, seen_norm, unseen_norm, joint_norm, harm_mean_acc = evaluate_model(
                model,
                val_seen_loader,
                val_unseen_loader,
                val_joint_loader,
                seen_prototypes,
                args.n_way,
                args.k_shot,
                args.n_query,
                seen_labels,
                device,
                args.best_gamma
            )
            print(f"Sparsity = {sparsity_ratio*100}")
            print(f"Validation: Seen = {seen_acc:.3f}  | Unseen = {unseen_acc:.3f}  | Joint = {joint_acc:.3f}")
            print(f"Normalized: Seen = {seen_norm:.3f}  | Unseen = {unseen_norm:.3f}  | Joint = {joint_norm:.3f}")
            print(f"Harmonic Mean Accuracy: {harm_mean_acc:.3f}")

            results["val_seen_acc"].append(seen_acc)
            results["val_unseen_acc"].append(unseen_acc)
            results["val_joint_acc"].append(joint_acc)
            results["harmonic_mean_acc"].append(harm_mean_acc)
            results["Sparsity"].append(sparsity_ratio)
            
            del model
            gc.collect()
            torch.cuda.empty_cache()
        
        # Load CSV
        df = pd.DataFrame(results)
        df.to_csv(expanded_join(args.results_dir, f"{args.save_name}_pruning_results.csv"), index=False)

        # Create figure with 3 rows, 1 column
        fig, axes = plt.subplots(2, 1, figsize=(7, 12), sharex=True)

        sparsities = df["Sparsity"]
        # -------------------------
        # 1) Accuracies
        # -------------------------
        ax1 = axes[0]
        ax1.plot(sparsities, df["val_seen_acc"], label="Seen Acc")
        ax1.plot(sparsities, df["val_unseen_acc"], label="Unseen Acc")
        ax1.plot(sparsities, df["val_joint_acc"], label="Joint Acc")

        ax1.set_ylim(0, 1)
        ax1.set_ylabel("Accuracy")
        ax1.set_title("Validation Accuracies")
        ax1.legend()
        ax1.grid(True)

        # -------------------------
        # 3) Harmonic Mean
        # -------------------------
        ax2 = axes[1]
        ax2.plot(sparsities, df["harmonic_mean_acc"], label="Harmonic Mean Acc")
        
        ax2.set_ylim(0, 1)
        ax2.set_xlabel("Sparsity")
        ax2.set_ylabel("Accuracy")
        ax2.set_title("Harmonic Mean Accuracy")
        ax2.legend()
        ax2.grid(True)

        plt.tight_layout()
        fig.savefig(expanded_join(args.results_dir, f"{args.save_name}_pruning.png"))
        plt.close(fig)

        print(f"Saved image: {args.save_name}_pruning.png")
    
    if args.quantize:
        # load model
        model_path = expanded_join(args.models_dir, f"{args.save_name}.pth")
        if os.path.exists(model_path):
            print(f"Loading model from {model_path}")
            model.load_state_dict(torch.load(model_path))
        
        model.eval()
        print("Computing seen prototypes")
        
        del model
        # Quantization types
        if args.quantization_type == "bf16":
            quant_type = torch.bfloat16
        else:
            quant_type = torch.float16
            
        quantization_results = {"Weights_type": [], "SeenAccuracy": [], "UnseenAccuracy": [], "JointAccuracy": [], "HarmonicMeanAccuracy": [], "MedianLatency": [], "Model_size": []}
        
        for qtype in [torch.float32, quant_type]:
            print(f"Quantizing the model in {args.quantization_type} and evaluating on {device}")
            print()
            # load the-pretrained model
            model = EmbeddingModel(backbone_name="resnet50", embed_dim=int(args.dim))
            model_path = expanded_join(args.models_dir, f"{args.save_name}.pth")
            if os.path.exists(model_path):
                print(f"Loading model from {model_path}")
                model.load_state_dict(torch.load(model_path))
            
            # quantize model
            print(f"Quantizing model to type: {qtype}")
            model = model.to(dtype=qtype)
            print(f"Running model on {next(model.parameters()).device}")
            # Compile the model
            model = torch.compile(model, mode="max-autotune", fullgraph=False)
            model.to(device)

            # Compute seen prototypes (once)
            seen_prototypes = compute_seen_class_prototypes_quantized(model, train_loader, device, qtype)

            # save model
            print()
            newname = f"{args.save_name}_{str(qtype).split('.')[1]}.pth"
            model_dict = model._orig_mod  # unwrap once
            torch.save(model_dict.state_dict(), expanded_join(args.models_dir, newname))
            print("Quantized model saved.")

            print(f"Evaluating model with type: {qtype}")
            # evaluate model
            model.eval()
            
            seen_acc, unseen_acc, joint_acc, seen_norm, unseen_norm, joint_norm, harm_mean_acc, avg_latency, median_latency, total_size_mb = evaluate_inference_model(args, model, qtype, device, val_seen_loader, val_unseen_loader, val_joint_loader, seen_prototypes, seen_labels, best_gamma=args.best_gamma)
            print(f"Validation: Seen = {seen_acc:.3f}  | Unseen = {unseen_acc:.3f}  | Joint = {joint_acc:.3f}")
            print(f"Normalized: Seen = {seen_norm:.3f}  | Unseen = {unseen_norm:.3f}  | Joint = {joint_norm:.3f}")
            print(f"Harmonic Mean Accuracy: {harm_mean_acc:.3f}")

            quantization_results["Weights_type"].append(str(qtype).split('.')[1])
            quantization_results["SeenAccuracy"].append(seen_acc)
            quantization_results["UnseenAccuracy"].append(unseen_acc)
            quantization_results["JointAccuracy"].append(joint_acc)
            quantization_results["HarmonicMeanAccuracy"].append(harm_mean_acc)
            quantization_results["MedianLatency"].append(median_latency)
            quantization_results["Model_size"].append(total_size_mb)

            print("Model evaluation:")
            print(f"Validation joint accuracy: {(joint_acc*100):.2f}")
            print(f"Latency: {median_latency:.2f} ms")
            print(f"Size: {total_size_mb:.2f} MB")
            print()

        df = pd.DataFrame(quantization_results)
        df.to_csv(expanded_join(args.results_dir, f'quantization_metrics_{device}.csv'), index=False)

        #quantization_results = {'Weights_type': ["float32", "float16"], "SeenAccuracy": [0.9, 0.6], "UnseenAccuracy": [0.9, 0.6], "JointAccuracy": [0.9, 0.6], "HarmonicMeanAccuracy": [0.9, 0.6], "MedianLatency": [300, 94], "Model_size": [300000, 800000]}
        metrics = list(quantization_results.keys())[1:]

        label = ["", "", "", "", "ms", "MB"]

        # plot collected metrics
        fig, ax = plt.subplots(2, 3, figsize=(12, 8), sharey=True)

        weights_y = quantization_results["Weights_type"]

        for i, metric in enumerate(metrics):
            if i < 3:
                row = 0
                column = i
            else:
                row = i // 3
                column = i % 3
            values = quantization_results[metric]
            ax[row, column].barh(weights_y, values, height=0.7)
            ax[row, column].set_xlabel(label[i])
            if column == 0:
                ax[row, column].set_ylabel("Model weights' type")
            ax[row, column].set_title(f'{metric.replace("_", " ")}')

        plt.tight_layout()
        plt.savefig(expanded_join(args.results_dir, f"quantization_metrics_{args.quantization_type}.png"))

        # compact plot
        fig, ax = plt.subplots(1, 3, figsize=(12, 4), sharey=True)

        metrics = ["JointAccuracy", "MedianLatency", "Model_size"]
        label = ["", "ms", "MB"]

        for i, metric in enumerate(metrics):
            values = quantization_results[metric]
            ax[i].barh(weights_y, values, height=0.7)
            ax[i].set_xlabel(label[i])
            if i == 0:
                ax[i].set_ylabel("Model weights' type")
            ax[i].set_title(f'{metric.replace("_", " ")}')

        plt.tight_layout()
        plt.savefig(expanded_join(args.results_dir, f"quantization_metrics_{args.quantization_type}_compact.png"))