import argparse
import csv
import gc
import json
import math
import random
import tempfile
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import sklearn
import torch
import torch.nn.functional as F
import torch_geometric
import torch_sparse
from sklearn.cluster import KMeans
from torch_geometric.datasets import Amazon, Reddit
from torch_geometric.transforms import NormalizeFeatures
from torch_geometric.utils import is_undirected, to_undirected
from torch_sparse import SparseTensor

from models import GCN_mamba_Net_pro_max


OGB_NAMES = {"arxiv": "ogbn-arxiv", "products": "ogbn-products"}


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def set_split(data, seed, official=None):
    if official is None:
        order = torch.randperm(
            data.num_nodes, generator=torch.Generator().manual_seed(seed)
        )
        count = int(0.1 * data.num_nodes)
        indices = (order[:count], order[count:2 * count], order[2 * count:])
    else:
        indices = (official["train"], official["valid"], official["test"])
    for name, index in zip(("train_mask", "val_mask", "test_mask"), indices):
        mask = torch.zeros(data.num_nodes, dtype=torch.bool, device=data.y.device)
        mask[index.view(-1).long().to(mask.device)] = True
        if not mask.any():
            raise ValueError(f"{name} must contain at least one node")
        setattr(data, name, mask)


def load_graph(args):
    official = None
    if args.dataset in OGB_NAMES:
        from ogb.nodeproppred import PygNodePropPredDataset

        dataset = PygNodePropPredDataset(name=OGB_NAMES[args.dataset], root=args.root)
        if args.split == "official":
            official = dataset.get_idx_split()
    elif args.dataset == "reddit":
        dataset = Reddit(root=str(Path(args.root) / "reddit"))
    else:
        dataset = Amazon(
            root=str(Path(args.root) / args.dataset), name=args.dataset,
            transform=NormalizeFeatures(),
        )
    data = dataset[0]
    data.y = data.y.view(-1).long()
    if getattr(data, "edge_index", None) is None:
        row, col, _ = data.adj_t.coo()
        data.edge_index = torch.stack((col, row), dim=0)
    # Reddit is already undirected; avoid sorting its large edge tensor.
    if args.dataset != "reddit" and not is_undirected(
        data.edge_index, num_nodes=data.num_nodes
    ):
        data.edge_index = to_undirected(data.edge_index, num_nodes=data.num_nodes)
    return data, int(dataset.num_classes), official


def normalized_adjacency(edge_index, num_nodes):
    """D^(-1/2) (A + I) D^(-1/2), stored as a sparse tensor."""
    adj = SparseTensor(
        row=edge_index[0], col=edge_index[1], sparse_sizes=(num_nodes, num_nodes)
    ) + SparseTensor.eye(num_nodes)
    inv_sqrt = adj.sum(dim=1).float().pow(-0.5)
    inv_sqrt[torch.isinf(inv_sqrt)] = 0
    return inv_sqrt.view(-1, 1) * adj * inv_sqrt.view(1, -1)


def kmeans_clusters(features, num_clusters):
    features = F.normalize(features.detach().cpu().float(), p=2, dim=1).numpy()
    assignments = KMeans(
        n_clusters=num_clusters, random_state=42, n_init=10
    ).fit_predict(features)
    return torch.as_tensor(assignments, dtype=torch.long)


def cluster_targets(data, num_clusters, num_classes):
    """Training-only class distributions for the fixed feature clusters."""
    targets = data.cluster_id[data.train_mask]
    counts = torch.bincount(
        targets * num_classes + data.y[data.train_mask],
        minlength=num_clusters * num_classes,
    ).reshape(num_clusters, num_classes).float()
    totals = counts.sum(dim=1, keepdim=True)
    return targets, counts / totals.clamp_min(1), totals.squeeze(1) > 0


def accuracy(logits, labels, mask):
    return (logits[mask].argmax(-1) == labels[mask]).float().mean().item()


def train(model, data, num_classes, args, run, seed, history_writer):
    optimizer = torch.optim.Adam(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )
    use_contrastive = args.contrastive_weight > 0
    if use_contrastive:
        targets, prototypes, active = cluster_targets(data, args.cluster, num_classes)
    best_epoch, best_val, best_test = 0, -1.0, 0.0

    for epoch in range(1, args.epochs + 1):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        layer_log_probs = model(model.args, data)
        train_log_probs = layer_log_probs[data.train_mask]
        loss = F.nll_loss(train_log_probs[:, -1, :], data.y[data.train_mask])
        if use_contrastive:
            log_probs = train_log_probs[:, min(1, args.layers - 1), :]
            cluster_logits = (log_probs @ prototypes.t()) / args.tau
            # Ignore clusters without training nodes in this particular run.
            cluster_logits = cluster_logits.masked_fill(~active.unsqueeze(0), -torch.inf)
            loss = loss + args.contrastive_weight * F.cross_entropy(cluster_logits, targets)
        loss.backward()
        optimizer.step()

        if epoch != 1 and epoch % args.eval_interval != 0 and epoch != args.epochs:
            continue
        model.eval()
        with torch.inference_mode():
            eval_logits = model(model.args, data)[:, -1, :]
            train_acc, val_acc, test_acc = (
                accuracy(eval_logits, data.y, mask)
                for mask in (data.train_mask, data.val_mask, data.test_mask)
            )
        history_writer.writerow((run, seed, epoch, loss.item(), train_acc, val_acc, test_acc))
        if val_acc > best_val:
            best_epoch, best_val, best_test = epoch, val_acc, test_acc
        print(
            f"run={run} epoch={epoch} loss={loss.item():.4f} "
            f"train={train_acc:.4f} val={val_acc:.4f} test={test_acc:.4f}",
            flush=True,
        )
        if args.early_stopping > 0 and epoch - best_epoch >= args.early_stopping:
            break
    return dict(best_epoch=best_epoch, best_val=best_val, best_test=best_test, final_epoch=epoch)


def smoothing_metrics(hidden, logits, data, args, seed):
    generator = torch.Generator(device=hidden.device).manual_seed(seed)
    nodes = torch.randperm(data.num_nodes, generator=generator, device=hidden.device)
    nodes = nodes[:args.metric_nodes]
    edges = torch.randperm(
        data.edge_index.size(1), generator=generator, device=hidden.device
    )[:args.metric_edges]
    row, col = data.edge_index[:, edges]
    rows = []
    for layer in range(hidden.size(1)):
        features = hidden[:, layer, :].float()
        left, right = features[row], features[col]
        rows.append(dict(
            layer=layer + 1,
            feature_variance=features[nodes].var(dim=0, unbiased=False).mean().item(),
            edge_cosine=F.cosine_similarity(left, right, dim=-1).mean().item(),
            dirichlet_energy=(left - right).pow(2).sum(-1).mean().item(),
            test_accuracy=(accuracy(logits, data.y, data.test_mask)
                           if layer == hidden.size(1) - 1 else ""),
        ))
    return rows


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dataset", type=str.lower,
                        choices=["arxiv","reddit"],
                        default="arxiv")
    parser.add_argument("--root", default="./data", help="dataset download/cache directory")
    parser.add_argument("--split", choices=["random", "official"], default="random")
    parser.add_argument("--layers", type=int, default=9)
    parser.add_argument("--d_model", type=int, default=128)
    parser.add_argument("--d_inner", type=int, default=128)
    parser.add_argument("--dt_rank", type=int, default=8)
    parser.add_argument("--d_state", type=int, default=8)
    parser.add_argument("--dropout", type=float, default=0.6)
    parser.add_argument("--mamba_dropout", type=float, default=0.05)
    parser.add_argument("--bias", action="store_true")
    parser.add_argument("--cluster", type=int, default=5)
    parser.add_argument("--tau", type=float, default=1.0)
    parser.add_argument("--contrastive_weight", type=float, default=0.1)
    parser.add_argument("--lr", type=float, default=0.02)
    parser.add_argument("--weight_decay", type=float, default=1e-3)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--eval_interval", type=int, default=10)
    parser.add_argument("--early_stopping", type=int, default=200,
                        help="epochs without validation improvement; 0 disables")
    parser.add_argument("--runs", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=int, default=0, help="CUDA index; -1 selects CPU")
    parser.add_argument("--no_metrics", action="store_true",
                        help="skip layer features and final smoothing evaluation")
    parser.add_argument("--metric_nodes", type=int, default=10000)
    parser.add_argument("--metric_edges", type=int, default=100000)
    parser.add_argument("--output_dir", default="./results/large_scale_main")
    args = parser.parse_args()
    for name in ("layers", "d_model", "d_inner", "dt_rank", "d_state", "cluster",
                 "epochs", "eval_interval", "runs", "metric_nodes", "metric_edges"):
        if getattr(args, name) < 1:
            parser.error(f"--{name} must be positive")
    if args.d_model != args.d_inner:
        parser.error("--d_model and --d_inner must be equal for S3TM")
    if not all(math.isfinite(value) for value in (
        args.dropout, args.mamba_dropout, args.tau, args.contrastive_weight,
        args.lr, args.weight_decay,
    )):
        parser.error("floating-point arguments must be finite")
    if not (0 <= args.dropout < 1 and 0 <= args.mamba_dropout < 1):
        parser.error("dropout probabilities must be in [0, 1)")
    if args.tau <= 0 or args.lr <= 0 or args.contrastive_weight < 0 or args.weight_decay < 0:
        parser.error("tau/lr must be positive; loss weight/weight decay must be non-negative")
    if args.early_stopping < 0 or args.device < -1 or not 0 <= args.seed <= 2**32 - args.runs:
        parser.error("invalid early_stopping, device, or run seed range")
    if args.split == "official" and args.dataset not in OGB_NAMES:
        parser.error("official splits are available only for arxiv and products")
    return args


def main():
    args = parse_args()
    seed_everything(args.seed)
    device = torch.device(
        f"cuda:{args.device}" if args.device >= 0 and torch.cuda.is_available() else "cpu"
    )
    data, num_classes, official = load_graph(args)
    if args.contrastive_weight > 0:
        if args.cluster > data.num_nodes:
            raise ValueError("--cluster cannot exceed the number of nodes")
        print(f"KMeans: clusters={args.cluster} random_state=42 n_init=10", flush=True)
        data.cluster_id = kmeans_clusters(data.x, args.cluster)
    data.adj_t = normalized_adjacency(data.edge_index, data.num_nodes)
    data = data.to(device)
    dataset_info = SimpleNamespace(num_features=data.num_features, num_classes=num_classes)
    model_args = SimpleNamespace(
        **vars(args), layer_num=args.layers, layer_fusion="s3tm", readout="last"
    )
    print(
        f"dataset={args.dataset} nodes={data.num_nodes} edges={data.edge_index.size(1)} "
        f"features={data.num_features} classes={num_classes} split={args.split} device={device}",
        flush=True,
    )

    output_root = Path(args.output_dir)
    output_root.mkdir(parents=True, exist_ok=True)
    output = Path(tempfile.mkdtemp(prefix=args.dataset + "_", dir=str(output_root)))
    versions = dict(torch=torch.__version__, numpy=np.__version__, sklearn=sklearn.__version__,
                    torch_geometric=torch_geometric.__version__, torch_sparse=torch_sparse.__version__)
    if args.dataset in OGB_NAMES:
        import ogb

        versions["ogb"] = ogb.__version__
    with (output / "config.json").open("w") as handle:
        json.dump(dict(arguments=vars(args), device=str(device), versions=versions,
                       model=dict(source="models.py", name="GCN_mamba_Net_pro_max",
                                  layer_fusion="s3tm", readout="last"),
                       kmeans=dict(random_state=42, n_init=10, normalization="l2")),
                  handle, indent=2)
        handle.write("\n")
    print(f"Output directory: {output}", flush=True)

    fields = ["run", "seed", "best_epoch", "best_val", "best_test", "final_epoch",
              "metrics_epoch", "layer", "feature_variance", "edge_cosine",
              "dirichlet_energy", "test_accuracy"]
    best_tests = []
    with (output / "history.csv").open("w", newline="", buffering=1) as history_file, \
            (output / "results.csv").open("w", newline="", buffering=1) as results_file:
        history_writer = csv.writer(history_file)
        history_writer.writerow(("run", "seed", "epoch", "train_loss", "train_accuracy",
                                 "val_accuracy", "test_accuracy"))
        results_writer = csv.DictWriter(results_file, fieldnames=fields)
        results_writer.writeheader()
        for run in range(1, args.runs + 1):
            seed = args.seed + run - 1
            seed_everything(seed)
            set_split(data, seed, official)
            model = GCN_mamba_Net_pro_max(dataset_info, model_args).to(device)
            result = train(model, data, num_classes, args, run, seed, history_writer)
            best_tests.append(result["best_test"])
            rows = [{}]
            if not args.no_metrics:
                model.eval()
                with torch.inference_mode():
                    layer_log_probs, hidden = model(model_args, data, return_hidden=True)
                    rows = smoothing_metrics(hidden, layer_log_probs[:, -1, :], data, args, seed)
                del layer_log_probs, hidden
            for row in rows:
                results_writer.writerow(dict(
                    run=run, seed=seed, **result,
                    metrics_epoch="" if args.no_metrics else result["final_epoch"], **row,
                ))
            print(
                f"run={run}/{args.runs} best_epoch={result['best_epoch']} "
                f"best_val={result['best_val']:.4f} best_test={result['best_test']:.4f}",
                flush=True,
            )
            del model
            gc.collect()
            if device.type == "cuda":
                torch.cuda.empty_cache()
    std = float(np.std(best_tests, ddof=1)) if args.runs > 1 else 0.0
    print(f"test_mean={np.mean(best_tests):.4f} test_std={std:.4f} output={output}", flush=True)


if __name__ == "__main__":
    main()
