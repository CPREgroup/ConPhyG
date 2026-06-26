"""Generate synthetic gamut data and train gamut_distance_network_39d.pth."""

import argparse
import json
import multiprocessing as mp
import os
from concurrent.futures import ProcessPoolExecutor

import gurobipy as gp
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from gurobipy import GRB
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm


class ConditionalGamutNetwork(nn.Module):
    """Network architecture used by gamut_distance_network.py."""

    def __init__(self, input_dim=39, p_vector_dim=12):
        super().__init__()
        assert input_dim == 39, f"Expected input_dim=39, got {input_dim}"
        assert p_vector_dim == 12, f"Expected p_vector_dim=12, got {p_vector_dim}"

        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.BatchNorm1d(256),
            nn.Linear(256, 512),
            nn.ReLU(),
            nn.BatchNorm1d(512),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.BatchNorm1d(256),
            nn.Dropout(0.2),
        )

        self.gamut_head = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 1),
        )

        self.p_vector_head = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, p_vector_dim),
            nn.Sigmoid(),
        )

        self.distance_head = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 1),
            nn.Softplus(),
        )

    def forward(self, x):
        features = self.encoder(x)
        gamut_score = self.gamut_head(features)
        p_vector = self.p_vector_head(features)
        distance = self.distance_head(features)
        return gamut_score, p_vector, distance


class ConditionalGamutDataset(Dataset):
    def __init__(self, x, y_gamut, y_p_vector, y_distance):
        self.x = torch.from_numpy(x).float()
        self.y_gamut = torch.from_numpy(y_gamut).float()
        self.y_p_vector = torch.from_numpy(y_p_vector).float()
        self.y_distance = torch.from_numpy(y_distance).float()

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        return self.x[idx], self.y_gamut[idx], self.y_p_vector[idx], self.y_distance[idx]


def solve_color_mixing_gurobi(c_target, t_matrix_12, timeout=10):
    """Solve min ||T p - C||^2 with p in [0, 1]^12."""
    try:
        with gp.Env(empty=True) as env:
            env.setParam("OutputFlag", 0)
            env.setParam("TimeLimit", timeout)
            env.start()

            with gp.Model(env=env) as model:
                p = model.addVars(12, lb=0.0, ub=1.0, name="p")
                c_prime = model.addVars(3, lb=-GRB.INFINITY, name="c_prime")
                d = model.addVars(3, lb=-GRB.INFINITY, name="d")

                model.setObjective(d[0] * d[0] + d[1] * d[1] + d[2] * d[2], GRB.MINIMIZE)

                for i in range(3):
                    model.addConstr(
                        gp.quicksum(t_matrix_12[i, j] * p[j] for j in range(12)) == c_prime[i]
                    )
                    model.addConstr(d[i] == c_prime[i] - float(c_target[i]))

                model.optimize()

                if model.Status == GRB.Status.OPTIMAL:
                    p_solution = np.array([p[j].X for j in range(12)], dtype=np.float32)
                    distance = float(np.sqrt(max(model.ObjVal, 0.0)))
                    is_feasible = distance < 1e-5
                    return is_feasible, p_solution, distance
    except Exception:
        pass

    return False, np.zeros(12, dtype=np.float32), float("inf")


def generate_t_matrix_12(gamut_size, base_seed):
    """Generate one synthetic 3x12 camera-projector color mixing matrix."""
    rng = np.random.RandomState(base_seed)

    if gamut_size == "large":
        main_range = (0.65, 0.95)
        cross_range = (0.01, 0.12)
        global_scale = rng.uniform(0.88, 1.0)
        projector_balance = np.ones(4, dtype=np.float32)
    elif gamut_size == "medium":
        main_range = (0.35, 0.65)
        cross_range = (0.05, 0.22)
        global_scale = rng.uniform(0.55, 0.80)
        projector_balance = rng.uniform(0.7, 1.0, size=4)
    else:
        main_range = (0.08, 0.25)
        cross_range = (0.05, 0.22)
        global_scale = rng.uniform(0.20, 0.50)
        projector_balance = rng.uniform(0.4, 1.0, size=4)
        weak_projectors = rng.choice(4, size=rng.randint(1, 3), replace=False)
        projector_balance[weak_projectors] *= rng.uniform(0.2, 0.5)

    t_matrix = np.zeros((3, 12), dtype=np.float32)
    for proj_id in range(4):
        for proj_ch in range(3):
            col = proj_id * 3 + proj_ch
            t_matrix[proj_ch, col] = rng.uniform(*main_range)
            for cam_ch in range(3):
                if cam_ch != proj_ch:
                    t_matrix[cam_ch, col] = rng.uniform(*cross_range)

    for proj_id in range(4):
        t_matrix[:, proj_id * 3:(proj_id + 1) * 3] *= projector_balance[proj_id]

    t_matrix *= global_scale

    if gamut_size == "small":
        weak_cols = rng.choice(12, rng.randint(4, 7), replace=False)
        for col in weak_cols:
            t_matrix[:, col] *= rng.uniform(0.25, 0.65)
        if rng.random() < 0.5:
            t_matrix[rng.choice(3), :] *= rng.uniform(0.3, 0.7)

    return t_matrix.astype(np.float32)


def solve_single_sample(args):
    c_target, t_matrix_12 = args
    is_feasible, p_solution, distance = solve_color_mixing_gurobi(c_target, t_matrix_12)

    x = np.concatenate([c_target, t_matrix_12.flatten()]).astype(np.float32)
    y_gamut = np.float32(0.0 if is_feasible else 1.0)
    y_distance = np.float32(distance)
    return x, y_gamut, p_solution.astype(np.float32), y_distance


def generate_dataset(args):
    os.makedirs(args.dataset_dir, exist_ok=True)
    dataset_path = os.path.join(args.dataset_dir, "dataset.npz")
    t_path = os.path.join(args.dataset_dir, "T_matrices.npy")
    stats_path = os.path.join(args.dataset_dir, "dataset_stats.json")

    if os.path.exists(dataset_path) and not args.force_dataset:
        print(f"Using existing dataset: {dataset_path}")
        return dataset_path

    np.random.seed(args.seed)
    samples_per_t = args.num_samples // args.num_t_matrices
    actual_samples = samples_per_t * args.num_t_matrices
    if actual_samples != args.num_samples:
        print(f"num_samples rounded down to {actual_samples} to divide evenly by num_t_matrices.")

    num_large = int(args.num_t_matrices * 0.4)
    num_medium = int(args.num_t_matrices * 0.4)
    num_small = args.num_t_matrices - num_large - num_medium

    t_matrices = []
    gamut_labels = []
    for i in range(num_large):
        t_matrices.append(generate_t_matrix_12("large", args.seed + i * 17))
        gamut_labels.append("large")
    for i in range(num_medium):
        t_matrices.append(generate_t_matrix_12("medium", args.seed + 10000 + i * 17))
        gamut_labels.append("medium")
    for i in range(num_small):
        t_matrices.append(generate_t_matrix_12("small", args.seed + 20000 + i * 17))
        gamut_labels.append("small")

    np.save(t_path, np.array(t_matrices, dtype=np.float32))

    all_x, all_y_gamut, all_y_p, all_y_distance = [], [], [], []
    coverage_by_type = {"small": [], "medium": [], "large": []}

    workers = args.num_workers if args.num_workers is not None else mp.cpu_count()
    print(f"Generating {actual_samples:,} samples with {args.num_t_matrices:,} T matrices.")
    print(f"Gurobi workers: {workers}")

    for t_matrix, label in tqdm(list(zip(t_matrices, gamut_labels)), desc="T matrices"):
        c_targets = np.random.uniform(0.0, 4.0, size=(samples_per_t, 3)).astype(np.float32)
        tasks = [(c_targets[i], t_matrix) for i in range(samples_per_t)]

        with ProcessPoolExecutor(max_workers=workers) as executor:
            results = list(executor.map(solve_single_sample, tasks))

        y_for_t = np.array([item[1] for item in results], dtype=np.float32)
        coverage_by_type[label].append(float((y_for_t == 0).mean()))

        for x, y_gamut, y_p, y_distance in results:
            all_x.append(x)
            all_y_gamut.append(y_gamut)
            all_y_p.append(y_p)
            all_y_distance.append(y_distance)

    x = np.array(all_x, dtype=np.float32)
    y_gamut = np.array(all_y_gamut, dtype=np.float32).reshape(-1, 1)
    y_p = np.array(all_y_p, dtype=np.float32)
    y_distance = np.array(all_y_distance, dtype=np.float32).reshape(-1, 1)

    np.savez(dataset_path, X=x, Y_gamut=y_gamut, Y_p_vector=y_p, Y_distance=y_distance)

    in_gamut_count = int((y_gamut == 0).sum())
    out_gamut_count = int((y_gamut == 1).sum())
    stats = {
        "num_samples": int(len(x)),
        "num_T_matrices": int(args.num_t_matrices),
        "samples_per_T": int(samples_per_t),
        "in_gamut_count": in_gamut_count,
        "out_gamut_count": out_gamut_count,
        "in_gamut_ratio": float(in_gamut_count / len(x)),
        "input_dim": 39,
        "p_vector_dim": 12,
        "gamut_distribution": {
            "small": int(num_small * samples_per_t),
            "medium": int(num_medium * samples_per_t),
            "large": int(num_large * samples_per_t),
        },
        "coverage_by_type": {
            key: float(np.mean(value)) if value else 0.0 for key, value in coverage_by_type.items()
        },
        "distance_stats": {
            "mean": float(y_distance.mean()),
            "median": float(np.median(y_distance)),
            "max": float(y_distance.max()),
            "min": float(y_distance.min()),
        },
    }
    with open(stats_path, "w", encoding="utf-8") as file:
        json.dump(stats, file, indent=2)

    print(f"Dataset saved to: {dataset_path}")
    print(f"T matrices saved to: {t_path}")
    print(f"Stats saved to: {stats_path}")
    return dataset_path


def compute_loss(model, batch, device):
    x, y_gamut, y_p, y_distance = [item.to(device) for item in batch]
    gamut_score, p_pred, distance_pred = model(x)

    loss_gamut = nn.functional.binary_cross_entropy_with_logits(gamut_score, y_gamut)

    in_mask = (y_gamut == 0).squeeze()
    if in_mask.sum() > 0:
        loss_p = nn.functional.mse_loss(p_pred[in_mask], y_p[in_mask])
    else:
        loss_p = torch.tensor(0.0, device=device)

    out_mask = (y_gamut == 1).squeeze()
    if out_mask.sum() > 0:
        loss_distance = nn.functional.mse_loss(distance_pred[out_mask], y_distance[out_mask])
    else:
        loss_distance = torch.tensor(0.0, device=device)

    return loss_gamut + loss_p + loss_distance, gamut_score, y_gamut


def train_model(args, dataset_path):
    data = np.load(dataset_path)
    x = data["X"]
    y_gamut = data["Y_gamut"]
    y_p = data["Y_p_vector"]
    y_distance = data["Y_distance"]

    # Shuffle the dataset before splitting
    indices = np.random.permutation(len(x))
    x = x[indices]
    y_gamut = y_gamut[indices]
    y_p = y_p[indices]
    y_distance = y_distance[indices]

    split_idx = int(len(x) * 0.9)
    train_dataset = ConditionalGamutDataset(
        x[:split_idx], y_gamut[:split_idx], y_p[:split_idx], y_distance[:split_idx]
    )
    val_dataset = ConditionalGamutDataset(
        x[split_idx:], y_gamut[split_idx:], y_p[split_idx:], y_distance[split_idx:]
    )

    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.loader_workers
    )
    val_loader = DataLoader(
        val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.loader_workers
    )

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    model = ConditionalGamutNetwork(input_dim=39, p_vector_dim=12).to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=5, verbose=True
    )

    best_val_loss = float("inf")
    for epoch in range(args.num_epochs):
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_count = 0

        for batch in tqdm(train_loader, desc=f"Epoch {epoch + 1}/{args.num_epochs} train", leave=False):
            optimizer.zero_grad()
            loss, gamut_score, target_gamut = compute_loss(model, batch, device)
            loss.backward()
            optimizer.step()

            train_loss += float(loss.item())
            train_correct += ((gamut_score > 0).float() == target_gamut).sum().item()
            train_count += len(target_gamut)

        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_count = 0
        with torch.no_grad():
            for batch in tqdm(val_loader, desc=f"Epoch {epoch + 1}/{args.num_epochs} val", leave=False):
                loss, gamut_score, target_gamut = compute_loss(model, batch, device)
                val_loss += float(loss.item())
                val_correct += ((gamut_score > 0).float() == target_gamut).sum().item()
                val_count += len(target_gamut)

        avg_train_loss = train_loss / max(1, len(train_loader))
        avg_val_loss = val_loss / max(1, len(val_loader))
        train_acc = train_correct / max(1, train_count)
        val_acc = val_correct / max(1, val_count)
        scheduler.step(avg_val_loss)

        print(
            f"Epoch {epoch + 1}/{args.num_epochs} | "
            f"train loss {avg_train_loss:.6f}, acc {train_acc:.4f} | "
            f"val loss {avg_val_loss:.6f}, acc {val_acc:.4f}"
        )

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), args.output_model)
            print(f"Saved best model to {args.output_model} (val loss {best_val_loss:.6f})")

    print(f"Training completed. Best validation loss: {best_val_loss:.6f}")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Reproduce gamut_distance_network_39d.pth from synthetic Gurobi labels."
    )
    parser.add_argument("--dataset_dir", type=str, default="dataset_gamut_distance_39d")
    parser.add_argument("--output_model", type=str, default="gamut_distance_network_39d.pth")
    parser.add_argument("--num_samples", type=int, default=1_000_000)
    parser.add_argument("--num_t_matrices", type=int, default=1_000)
    parser.add_argument("--num_workers", type=int, default=None)
    parser.add_argument("--loader_workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num_epochs", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=2048)
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--force_dataset", action="store_true")
    parser.add_argument("--skip_training", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    dataset_path = generate_dataset(args)
    if not args.skip_training:
        train_model(args, dataset_path)


if __name__ == "__main__":
    main()
