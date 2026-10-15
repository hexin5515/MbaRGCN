"""Large-graph oversmoothing experiment.

This script is intentionally separate from training_over_smoothing.py. It keeps
the graph sparse, trains Mamba with classification and KMeans cluster contrastive
losses, and records layer-wise smoothing statistics for Mamba and a sparse GCN
reference. KMeans follows training_non_targeted.py; --contrastive_weight 0
disables clustering and the auxiliary loss.

Examples:
  python training_large_scale_oversmoothing.py --dataset computers --model both
  python training_large_scale_oversmoothing.py --dataset arxiv --model mamba \
      --layers 5 --d_model 128 --d_inner 128 --d_state 4
  python training_large_scale_oversmoothing.py --dataset reddit --model mamba \
      --layers 2 --d_model 64 --d_inner 64 --no_metrics
"""

import argparse
import csv
import gc
import os
import random
from types import SimpleNamespace

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.cluster import KMeans
from torch import nn
from torch_geometric.nn import GCNConv
from torch_geometric.utils import is_undirected, to_undirected
from torch_sparse import SparseTensor

from dataset_loader import DataLoader
from models import GCN_mamba_Net_pro_max
from training_curves import CurveRecorder, prepare_curve_directory


OGB_NAMES = {
    "arxiv": "ogbn-arxiv",
    "products": "ogbn-products",
}


class DatasetView:
    """Small adapter exposing the attributes used by the existing Mamba model."""

    def __init__(self, data, num_classes):
        self.data = data
        self.num_features = data.num_features
        self.num_classes = int(num_classes)

    def __getitem__(self, index):
        return self.data


class SparseGCN(nn.Module):
    def __init__(self, num_features, num_classes, hidden, layers, dropout):
        super().__init__()
        if layers < 2:
            raise ValueError("--layers must be at least 2 for the GCN baseline")
        self.dropout = dropout
        self.convs = nn.ModuleList()
        # The graph is normalized once up front and kept as a SparseTensor.
        # Avoiding GCNConv's per-layer COO normalization matters on large graphs.
        conv_kwargs = {"add_self_loops": False, "normalize": False}
        self.convs.append(GCNConv(num_features, hidden, **conv_kwargs))
        for _ in range(layers - 1):
            self.convs.append(GCNConv(hidden, hidden, **conv_kwargs))
        self.classifier = nn.Linear(hidden, num_classes)

    def forward(self, data, return_hidden=True):
        x = data.x
        hidden = []
        for conv in self.convs:
            x = conv(x, data.adj_t)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
            if return_hidden:
                hidden.append(x)
        return (hidden if return_hidden else None), self.classifier(x)


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def random_split(num_nodes, seed):
    """Create a deterministic 1:1:8 node split from a random permutation."""
    generator = torch.Generator().manual_seed(seed)
    permutation = torch.randperm(num_nodes, generator=generator)
    train_end = int(0.1 * num_nodes)
    val_end = train_end + int(0.1 * num_nodes)
    return (
        permutation[:train_end],
        permutation[train_end:val_end],
        permutation[val_end:],
    )


def apply_split_masks(data, split_indices):
    for mask_name, indices in zip(
        ("train_mask", "val_mask", "test_mask"), split_indices
    ):
        mask = torch.zeros(
            data.num_nodes, dtype=torch.bool, device=data.y.device
        )
        mask[indices.view(-1).long().to(mask.device)] = True
        setattr(data, mask_name, mask)


def kmeans_node_clusters(features, num_clusters):
    """Match the feature normalization and KMeans in training_non_targeted.py."""
    if not 1 <= num_clusters <= features.size(0):
        raise ValueError("--cluster must be between 1 and the number of nodes")
    cluster_features = F.normalize(
        features.detach().cpu().float(), p=2, dim=1
    ).numpy()
    assignments = KMeans(
        n_clusters=num_clusters, random_state=42, n_init=10
    ).fit_predict(cluster_features)
    return torch.as_tensor(assignments, dtype=torch.long)


def prepare_cluster_targets(data, num_clusters, num_classes):
    """Build cluster label distributions using only the current training split."""
    cluster_ids = data.index[data.train_mask]
    labels = data.y[data.train_mask]
    counts = torch.bincount(
        cluster_ids * num_classes + labels,
        minlength=num_clusters * num_classes,
    ).reshape(num_clusters, num_classes).float()
    totals = counts.sum(dim=1, keepdim=True)
    # Integer cross-entropy targets are equivalent to the reference's one-hot
    # targets without allocating a train_nodes x clusters matrix on large graphs.
    data.cluster_train_id = cluster_ids
    data.cluster_train_label = counts / totals.clamp_min(1)
    data.cluster_train_active = totals.squeeze(1) > 0


def load_graph(name, root, split_mode="random", seed=42):
    name = name.lower()
    if split_mode == "official" and name not in OGB_NAMES:
        raise ValueError("--split official is only available for arxiv and products")
    if name in OGB_NAMES:
        from ogb.nodeproppred import PygNodePropPredDataset

        dataset = PygNodePropPredDataset(name=OGB_NAMES[name], root=root)
        data = dataset[0]
        num_classes = int(dataset.num_classes)
        if split_mode == "official":
            split = dataset.get_idx_split()
            split_indices = (split["train"], split["valid"], split["test"])
        else:
            split_indices = random_split(data.num_nodes, seed)
    elif name == "reddit":
        from torch_geometric.datasets import Reddit

        # PyG downloads and caches the public Reddit node-classification
        # dataset under ``<root>/reddit``.  Its bundled masks are intentionally
        # ignored so every dataset in this script uses the same seeded 1:1:8
        # split controlled by ``--seed``.
        dataset = Reddit(root=os.path.join(root, "reddit"))
        data = dataset[0]
        num_classes = int(dataset.num_classes)
        split_indices = random_split(data.num_nodes, seed)
    else:
        dataset = DataLoader(name)
        data = dataset[0]
        num_classes = int(data.y.view(-1).max().item()) + 1
        split_indices = random_split(data.num_nodes, seed)

    data.y = data.y.view(-1).long()

    apply_split_masks(data, split_indices)

    if getattr(data, "edge_index", None) is None:
        if getattr(data, "adj_t", None) is None:
            raise ValueError("The dataset has neither edge_index nor adj_t")
        row, col, _ = data.adj_t.coo()
        data.edge_index = torch.stack((row, col), dim=0)

    # Reddit is already stored as an undirected COO graph by PyG.  Avoid both
    # the conversion and the full-graph symmetry check, which would create a
    # costly sort/temporary tensor at Reddit scale.
    if name != "reddit" and not is_undirected(
        data.edge_index, num_nodes=data.num_nodes
    ):
        data.edge_index = to_undirected(data.edge_index, num_nodes=data.num_nodes)
    return dataset, data, num_classes


def normalized_sparse_adjacency(edge_index, num_nodes):
    adj = SparseTensor(
        row=edge_index[0],
        col=edge_index[1],
        sparse_sizes=(num_nodes, num_nodes),
    )
    adj = adj + SparseTensor.eye(num_nodes)
    degree = adj.sum(dim=1).to(torch.float)
    degree_inv_sqrt = degree.pow(-0.5)
    degree_inv_sqrt[torch.isinf(degree_inv_sqrt)] = 0
    return (
        degree_inv_sqrt.view(-1, 1)
        * adj
        * degree_inv_sqrt.view(1, -1)
    )


def mamba_args(args, dataset_name):
    return SimpleNamespace(
        dataset=dataset_name,
        d_model=args.d_model,
        d_inner=args.d_inner,
        dt_rank=args.dt_rank,
        d_state=args.d_state,
        layer_num=args.layers,
        dropout=args.dropout,
        mamba_dropout=args.mamba_dropout,
        bias=args.bias,
        readout="last",
        gpr_alpha=0.1,
        layer_fusion=args.layer_fusion,
        fusion_heads=args.fusion_heads,
        fusion_dropout=args.fusion_dropout,
        fusion_hidden=args.fusion_hidden,
    )


def mamba_hidden(model, data, model_args, return_hidden=True,
                 contrastive_mask=None):
    x = model.lin1(data.x)
    x = model.norm_1(x)
    x = F.relu(x)
    x = F.dropout(x, p=model.dropout, training=model.training)

    # The original block materializes deltaA and deltaB_u for every layer at
    # once.  Since its input is the same feature matrix at each scan step, the
    # input projection can be computed once and the state can be advanced one
    # layer at a time.  This is algebraically equivalent and keeps peak memory
    # proportional to one layer's state instead of all layers' states.
    block = model.mamba
    d_inner, d_state = block.A_log.shape
    projected = block.x_proj(x)
    delta, B, C = projected.split(
        [model_args.dt_rank, d_state, d_state], dim=-1
    )
    delta = F.softplus(block.dt_proj(delta))
    A = -torch.exp(block.A_log.float())
    D = block.D.float() if block.D is not None else None
    state = torch.zeros(
        (x.size(0), d_inner, d_state), device=x.device, dtype=delta.dtype
    )

    reinject_source = model_args.layer_fusion == "s3tm"
    if reinject_source:
        # These tensors depend only on the shared input x and learned
        # parameters, so S3TM can reuse one copy across all scan steps.
        delta_a = torch.exp(delta.unsqueeze(-1) * A.unsqueeze(0))
        delta_b_u = (
            delta.unsqueeze(-1)
            * B.unsqueeze(1)
            * x.unsqueeze(-1)
        )
    else:
        # Ablations receive the source only at the first step. Reusing x at
        # every step would leave a shallow bypass after removing +x/Du.
        zero_input = torch.zeros_like(x)
        zero_projected = block.x_proj(zero_input)
        zero_delta, zero_B, zero_C = zero_projected.split(
            [model_args.dt_rank, d_state, d_state], dim=-1
        )
        zero_delta = F.softplus(block.dt_proj(zero_delta))

    collect_layers = return_hidden or model_args.layer_fusion != "s3tm"
    outputs = [] if collect_layers else None
    last_output = None
    contrastive_output = None
    contrastive_layer = min(1, model_args.layer_num - 1)
    for layer in range(model_args.layer_num):
        if reinject_source:
            state = delta_a * state + delta_b_u
            y = (state * C.unsqueeze(1)).sum(dim=-1)
        else:
            if layer == 0:
                step_delta, step_B, step_C, step_input = delta, B, C, x
            else:
                step_delta, step_B, step_C, step_input = (
                    zero_delta, zero_B, zero_C, zero_input
                )
            delta_a_step = torch.exp(
                step_delta.unsqueeze(-1) * A.unsqueeze(0)
            )
            delta_b_u_step = (
                step_delta.unsqueeze(-1)
                * step_B.unsqueeze(1)
                * step_input.unsqueeze(-1)
            )
            state = delta_a_step * state + delta_b_u_step
            y = (state * step_C.unsqueeze(1)).sum(dim=-1)
        state = data.adj_t @ state.reshape(x.size(0), d_inner * d_state)
        state = state.reshape(x.size(0), d_inner, d_state)
        state = F.relu(state)
        state = F.dropout(state, p=block.mamba_dropout, training=model.training)
        current_output = block.out_proj(y + x * D) if D is not None else block.out_proj(y)
        if collect_layers:
            outputs.append(current_output)
        if not collect_layers:
            last_output = current_output
            if contrastive_mask is not None and layer == contrastive_layer:
                # Retain only the training nodes at the contrastive layer.
                contrastive_output = (
                    current_output[contrastive_mask] + x[contrastive_mask]
                )
    if collect_layers:
        hidden = torch.stack(outputs, dim=1)
        if model_args.layer_fusion == "s3tm":
            hidden = hidden + x.unsqueeze(1)
        hidden = F.relu(hidden)
        hidden = F.dropout(hidden, p=model.dropout, training=model.training)
        if contrastive_mask is not None:
            return hidden, hidden[contrastive_mask, contrastive_layer, :]
        return hidden
    last_output = last_output + x
    last_output = F.relu(last_output)
    last_output = F.dropout(last_output, p=model.dropout, training=model.training)
    if contrastive_mask is not None:
        if contrastive_layer == model_args.layer_num - 1:
            contrastive_output = last_output[contrastive_mask]
        else:
            contrastive_output = F.dropout(
                F.relu(contrastive_output), p=model.dropout, training=model.training
            )
        return last_output, contrastive_output
    return last_output


def forward_model(model_name, model, data, model_args=None, return_hidden=True):
    if model_name == "mamba":
        hidden = mamba_hidden(model, data, model_args, return_hidden=return_hidden)
        if return_hidden:
            return hidden, model.fused_logits(hidden)
        if model_args.layer_fusion != "s3tm":
            return None, model.fused_logits(hidden)
        return None, model.lin2(hidden)
    hidden, logits = model(data, return_hidden=return_hidden)
    return (torch.stack(hidden, dim=1) if return_hidden else None), logits


def mamba_training_outputs(model, data, model_args):
    """Return final logits and the reference's contrastive-layer log probabilities."""
    hidden, contrastive_hidden = mamba_hidden(
        model, data, model_args, return_hidden=False,
        contrastive_mask=data.train_mask,
    )
    if model_args.layer_fusion == "s3tm":
        logits = model.lin2(hidden)
    else:
        logits = model.fused_logits(hidden)
    if model_args.layer_num <= 2:
        # The reference replaces the final layer's slot with the fused output.
        contrastive_logits = logits[data.train_mask]
    else:
        contrastive_logits = model.lin2(contrastive_hidden)
    return logits, F.log_softmax(contrastive_logits, dim=-1)


def accuracy(logits, labels, mask):
    prediction = logits[mask].argmax(dim=-1)
    return prediction.eq(labels[mask]).float().mean().item()


def train_model(model_name, model, data, model_args, args,
                curve_recorder=None, run=1, seed=None):
    optimizer = torch.optim.Adam(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )
    best_val = -1.0
    best_test = 0.0
    best_epoch = 0
    history = []
    use_contrastive = model_name == "mamba" and args.contrastive_weight > 0
    if use_contrastive:
        prepare_cluster_targets(data, args.cluster, model.lin2.out_features)

    for epoch in range(1, args.epochs + 1):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        if use_contrastive:
            logits, contrastive_log_probs = mamba_training_outputs(
                model, data, model_args
            )
        else:
            _, logits = forward_model(
                model_name, model, data, model_args, return_hidden=False
            )
        loss = F.cross_entropy(logits[data.train_mask], data.y[data.train_mask])
        if use_contrastive:
            logits_contrast = (
                contrastive_log_probs @ data.cluster_train_label.t()
            ) / args.tau
            # Some KMeans clusters may have no nodes in this training split.
            logits_contrast = logits_contrast.masked_fill(
                ~data.cluster_train_active.unsqueeze(0), -torch.inf
            )
            loss = loss + args.contrastive_weight * F.cross_entropy(
                logits_contrast, data.cluster_train_id
            )
        loss.backward()
        optimizer.step()

        should_eval = epoch == 1 or epoch % args.eval_interval == 0 or epoch == args.epochs
        if not should_eval:
            continue

        model.eval()
        with torch.inference_mode():
            _, eval_logits = forward_model(
                model_name, model, data, model_args, return_hidden=False
            )
            train_acc = accuracy(eval_logits, data.y, data.train_mask)
            val_acc = accuracy(eval_logits, data.y, data.val_mask)
            test_acc = accuracy(eval_logits, data.y, data.test_mask)

        history.append((epoch, loss.item(), train_acc, val_acc, test_acc))
        if curve_recorder is not None:
            curve_recorder.record(
                run, seed, epoch, loss.item(), train_acc, val_acc, test_acc
            )
        if val_acc > best_val:
            best_val = val_acc
            best_test = test_acc
            best_epoch = epoch

        print(
            f"{model_name} epoch={epoch} loss={loss.item():.4f} "
            f"train={train_acc:.4f} val={val_acc:.4f} test={test_acc:.4f}",
            flush=True,
        )

        if (
            args.early_stopping > 0
            and epoch - best_epoch >= args.early_stopping
        ):
            print(
                f"{model_name} early_stop epoch={epoch} "
                f"best_epoch={best_epoch}",
                flush=True,
            )
            break

    return history, best_epoch, best_val, best_test


def layer_metrics(hidden, logits, data, args, generator):
    num_nodes, num_layers = hidden.shape[:2]
    node_count = min(args.metric_nodes, num_nodes)
    node_sample = torch.randperm(num_nodes, generator=generator, device=hidden.device)[:node_count]

    row, col = data.edge_index
    edge_count = min(args.metric_edges, row.numel())
    edge_sample = torch.randperm(row.numel(), generator=generator, device=hidden.device)[:edge_count]
    row = row[edge_sample]
    col = col[edge_sample]

    rows = []
    for layer in range(num_layers):
        current = hidden[:, layer, :]
        sampled = current[node_sample].float()
        edge_left = current[row].float()
        edge_right = current[col].float()
        variance = sampled.var(dim=0, unbiased=False).mean().item()
        cosine = F.cosine_similarity(edge_left, edge_right, dim=-1).mean().item()
        energy = (edge_left - edge_right).pow(2).sum(dim=-1).mean().item()
        # The current model exposes only the final classifier output.  Keep
        # earlier-layer accuracy empty instead of allocating another full-graph
        # logits tensor merely to write NaN.
        layer_acc = accuracy(logits, data.y, data.test_mask) if layer == num_layers - 1 else float("nan")
        rows.append(
            {
                "layer": layer + 1,
                "feature_variance": variance,
                "edge_cosine": cosine,
                "dirichlet_energy": energy,
                "test_accuracy": layer_acc,
            }
        )
    return rows


def write_rows(
    path,
    dataset,
    split_mode,
    model_name,
    layer_fusion,
    run,
    seed,
    rows,
    best_epoch,
    best_val,
    best_test,
):
    fields = [
        "dataset",
        "split",
        "model",
        "layer_fusion",
        "run",
        "seed",
        "best_epoch",
        "best_val",
        "best_test",
        "layer",
        "feature_variance",
        "edge_cosine",
        "dirichlet_energy",
        "test_accuracy",
    ]
    exists = os.path.exists(path)
    with open(path, "a", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        if not exists:
            writer.writeheader()
        # Keep a summary row when layer metrics are disabled.  This makes the
        # CSV useful for accuracy-only large-graph runs while leaving metric
        # columns empty rather than fabricating values.
        rows_to_write = rows or [
            {
                "layer": "",
                "feature_variance": "",
                "edge_cosine": "",
                "dirichlet_energy": "",
                "test_accuracy": "",
            }
        ]
        for row in rows_to_write:
            writer.writerow(
                {
                    "dataset": dataset,
                    "split": split_mode,
                    "model": model_name,
                    "layer_fusion": layer_fusion if model_name == "mamba" else "n/a",
                    "run": run,
                    "seed": seed,
                    "best_epoch": best_epoch,
                    "best_val": best_val,
                    "best_test": best_test,
                    **row,
                }
            )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset",
        type=lambda value: value.lower(),
        choices=["computers", "photo", "arxiv", "products", "reddit"],
        default="computers",
    )
    parser.add_argument("--model", choices=["mamba", "gcn", "both"], default="both")
    parser.add_argument(
        "--split",
        choices=["random", "official"],
        default="random",
        help="random uses a seeded 1:1:8 split; official is available for OGB datasets",
    )
    parser.add_argument("--root", default="./data")
    parser.add_argument("--layers", type=int, default=9)
    parser.add_argument("--d_model", type=int, default=128)
    parser.add_argument("--d_inner", type=int, default=128)
    parser.add_argument("--dt_rank", type=int, default=8)
    parser.add_argument("--d_state", type=int, default=8)
    parser.add_argument("--hidden", type=int, default=128)
    parser.add_argument("--dropout", type=float, default=0.6)
    parser.add_argument("--mamba_dropout", type=float, default=0.05)
    parser.add_argument("--cluster", type=int, default=5, help="number of KMeans clusters")
    parser.add_argument("--tau", type=float, default=1.0, help="cluster contrastive temperature")
    parser.add_argument(
        "--contrastive_weight", type=float, default=0.1,
        help="weight of the Mamba cluster contrastive loss; 0 disables clustering",
    )
    parser.add_argument(
        "--layer_fusion",
        choices=["s3tm", "order_attention", "transformer", "gating", "mlp_filter"],
        default="s3tm",
        help="cross-propagation-layer fusion for the MbaRGCN model",
    )
    parser.add_argument("--fusion_heads", type=int, default=4)
    parser.add_argument("--fusion_dropout", type=float, default=0.1)
    parser.add_argument("--fusion_hidden", type=int, default=0)
    parser.add_argument("--bias", action="store_true")
    parser.add_argument("--lr", type=float, default=0.02)
    parser.add_argument("--weight_decay", type=float, default=1e-3)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--eval_interval", type=int, default=10)
    parser.add_argument(
        "--runs",
        type=int,
        default=1,
        help="independent runs; seed, seed+1, ... control model initialization and random splits",
    )
    parser.add_argument(
        "--early_stopping",
        type=int,
        default=200,
        help="stop after this many epochs without validation improvement; 0 disables",
    )
    parser.add_argument("--metric_nodes", type=int, default=10000)
    parser.add_argument("--metric_edges", type=int, default=100000)
    parser.add_argument(
        "--no_metrics",
        action="store_true",
        help="skip layer-wise oversmoothing metrics and their extra final forward pass",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--output", default="large_scale_oversmoothing.csv")
    parser.add_argument(
        "--curve_dir",
        default="",
        help="enable convergence CSV/PNG/PDF output under this directory",
    )
    args = parser.parse_args()

    if args.runs < 1:
        raise ValueError("--runs must be at least 1")
    if args.epochs < 1 or args.eval_interval < 1:
        raise ValueError("--epochs and --eval_interval must be positive")
    if args.fusion_heads < 1 or not 0.0 <= args.fusion_dropout < 1.0:
        raise ValueError("fusion_heads must be positive and fusion_dropout in [0, 1)")
    if args.fusion_hidden < 0:
        raise ValueError("fusion_hidden must be non-negative")
    if args.cluster < 1 or args.tau <= 0 or args.contrastive_weight < 0:
        raise ValueError("cluster and tau must be positive; contrastive_weight must be non-negative")

    if args.model in ("mamba", "both") and args.d_model != args.d_inner:
        raise ValueError(
            "The current GCN_mamba_Net_pro_max implementation requires "
            "--d_model and --d_inner to be equal."
        )

    seed_everything(args.seed)
    curve_directory = prepare_curve_directory(args, __file__)
    device = torch.device(f"cuda:{args.device}" if torch.cuda.is_available() else "cpu")
    dataset, data, num_classes = load_graph(
        args.dataset, args.root, split_mode=args.split, seed=args.seed
    )
    if args.model in ("mamba", "both") and args.contrastive_weight > 0:
        print(f"KMeans clustering: clusters={args.cluster} random_state=42 n_init=10", flush=True)
        data.index = kmeans_node_clusters(data.x, args.cluster)
    data.adj_t = normalized_sparse_adjacency(data.edge_index, data.num_nodes)
    data = data.to(device)
    dataset_view = DatasetView(data, num_classes)
    model_names = [args.model] if args.model != "both" else ["gcn", "mamba"]

    print(
        f"dataset={args.dataset} nodes={data.num_nodes} edges={data.edge_index.size(1)} "
        f"features={data.num_features} classes={num_classes} "
        f"split={args.split} device={device}",
        flush=True,
    )

    for model_name in model_names:
        best_tests = []
        curve_recorder = (
            CurveRecorder(curve_directory, model_name)
            if curve_directory is not None else None
        )
        for run in range(1, args.runs + 1):
            run_seed = args.seed + run - 1
            seed_everything(run_seed)
            if args.split == "random":
                apply_split_masks(data, random_split(data.num_nodes, run_seed))
            print(f"{model_name} run={run}/{args.runs} seed={run_seed}", flush=True)
            if model_name == "mamba":
                model_args = mamba_args(args, args.dataset)
                model = GCN_mamba_Net_pro_max(dataset_view, model_args).to(device)
            else:
                model_args = None
                model = SparseGCN(
                    data.num_features,
                    num_classes,
                    args.hidden,
                    args.layers,
                    args.dropout,
                ).to(device)

            _, best_epoch, best_val, best_test = train_model(
                model_name, model, data, model_args, args,
                curve_recorder=curve_recorder, run=run, seed=run_seed,
            )
            best_tests.append(best_test)
            if args.no_metrics:
                rows = []
            else:
                model.eval()
                with torch.inference_mode():
                    hidden, logits = forward_model(model_name, model, data, model_args)
                    generator = torch.Generator(device=device).manual_seed(run_seed)
                    rows = layer_metrics(hidden, logits, data, args, generator)
            write_rows(
                args.output,
                args.dataset,
                args.split,
                model_name,
                args.layer_fusion,
                run,
                run_seed,
                rows,
                best_epoch,
                best_val,
                best_test,
            )
            print(
                f"{model_name} run={run}/{args.runs} best_epoch={best_epoch} "
                f"best_val={best_val:.4f} best_test={best_test:.4f} "
                f"output={args.output}",
                flush=True,
            )
            del model
            if not args.no_metrics:
                del hidden, logits
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        if curve_recorder is not None:
            curve_recorder.plot(f"{args.dataset} / {model_name} / full graph")
        mean_test = float(np.mean(best_tests))
        std_test = float(np.std(best_tests, ddof=1)) if args.runs > 1 else 0.0
        print(
            f"{model_name} runs={args.runs} test_mean={mean_test:.4f} "
            f"test_std={std_test:.4f}",
            flush=True,
        )


if __name__ == "__main__":
    main()
