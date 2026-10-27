# -*- coding: utf-8 -*-
"""
Fashion-MNIST 优化器比较实验
==============================

在 Fashion-MNIST 数据集上训练一个简单的 MLP 模型，比较 4 种优化器
（SGD、Momentum、RMSProp、Adam）的表现。

包含三组实验：
  Experiment 1 —— 统一学习率（lr = 0.001）
  Experiment 2 —— 各优化器经验合理学习率
  Experiment 3 —— 多随机种子稳健性实验（mean ± std）

输出文件（全部由实际训练生成，未硬编码任何结果数字）：
  results/experiment1_same_lr.csv
  results/experiment2_tuned_lr.csv
  results/experiment3_multiseed_raw.csv
  results/experiment3_multiseed_summary.csv
  results/training_history.csv
  results/environment.txt
  figures/*.png

用法：
  python main.py --experiment all        # 运行全部实验（默认）
  python main.py --experiment exp2       # 仅运行 Experiment 2
  python main.py --experiment multiseed  # 仅运行 Experiment 3
  python main.py --seed 42               # 覆盖 Experiment 3 的随机种子
"""

import argparse
import os
import platform
import random
import sys
import time

import matplotlib
matplotlib.use("Agg")  # 无界面环境下也能保存图片

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

import torchvision  # noqa: F401  (记录版本)

# =========================
# 1. 基础设置
# =========================

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 64
EPOCHS = 10

# Experiment 1：统一学习率
SAME_LR = 0.001

# Experiment 2 / 3：各优化器经验合理学习率
TUNED_LR = {
    "SGD": 0.1,
    "Momentum": 0.01,
    "RMSProp": 0.001,
    "Adam": 0.001,
}

# Experiment 3：多随机种子
MULTISEED_LIST = [42, 123, 2026]

OPTIMIZER_NAMES = ["SGD", "Momentum", "RMSProp", "Adam"]

RESULTS_DIR = "results"
FIGURES_DIR = "figures"


# =========================
# 2. 随机种子控制
# =========================

def set_seed(seed):
    """统一设置 Python/NumPy/torch/CUDA 随机种子。"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# =========================
# 3. 数据集
# =========================

def build_loaders(seed):
    """
    构建训练/测试 DataLoader。
    使用 torch.Generator 显式控制 shuffle 顺序，确保相同 seed 下
    各优化器得到完全相同的 batch 顺序。
    """
    transform = transforms.ToTensor()

    train_dataset = datasets.FashionMNIST(
        root="./data", train=True, download=True, transform=transform
    )
    test_dataset = datasets.FashionMNIST(
        root="./data", train=False, download=True, transform=transform
    )

    # 训练集 shuffle 受 seed 控制
    g = torch.Generator()
    g.manual_seed(seed)

    train_loader = DataLoader(
        train_dataset, batch_size=BATCH_SIZE, shuffle=True, generator=g
    )
    test_loader = DataLoader(
        test_dataset, batch_size=BATCH_SIZE, shuffle=False
    )
    return train_loader, test_loader


# =========================
# 4. MLP 模型
# =========================

class MLP(nn.Module):
    """三层全连接网络：784 -> 128 -> 64 -> 10"""

    def __init__(self):
        super().__init__()
        self.network = nn.Sequential(
            nn.Flatten(),
            nn.Linear(28 * 28, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 10),
        )

    def forward(self, x):
        return self.network(x)


# =========================
# 5. 优化器
# =========================

def create_optimizer(name, model, lr):
    """根据名称创建对应的优化器。"""
    if name == "SGD":
        return optim.SGD(model.parameters(), lr=lr)
    elif name == "Momentum":
        return optim.SGD(model.parameters(), lr=lr, momentum=0.9)
    elif name == "RMSProp":
        return optim.RMSprop(model.parameters(), lr=lr)
    elif name == "Adam":
        return optim.Adam(model.parameters(), lr=lr)
    else:
        raise ValueError(f"未知优化器: {name}")


# =========================
# 6. 评估函数
# =========================

def evaluate(model, test_loader):
    """在测试集上计算准确率（百分比）。"""
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(DEVICE)
            labels = labels.to(DEVICE)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    return 100.0 * correct / total


# =========================
# 7. 训练单个优化器（返回每个 epoch 的历史）
# =========================

def train_optimizer(optimizer_name, lr, seed):
    """
    用指定优化器训练一个全新的模型。

    关键公平性保证：
        - 每次调用都重新 set_seed(seed)，使模型初始化一致；
        - 每次调用都重新 build_loaders(seed)，使 DataLoader 的 shuffle
          顺序也由同一个 seed 决定，从而各优化器在相同 seed 下看到
          完全相同的 batch 顺序。

    返回：
        history: list of dict，每个 epoch 记录 train_loss 与 test_accuracy
        final_accuracy: 最终测试准确率
        final_loss: 最终训练 loss
        training_time: 训练耗时（秒）
    """
    # 关键：每个优化器都从完全相同的模型初始化开始（受 seed 控制）
    set_seed(seed)
    model = MLP().to(DEVICE)

    # 每个优化器独立构建 DataLoader，保证 shuffle 顺序一致
    train_loader, test_loader = build_loaders(seed)

    criterion = nn.CrossEntropyLoss()
    optimizer = create_optimizer(optimizer_name, model, lr)

    history = []
    start_time = time.time()

    for epoch in range(EPOCHS):
        model.train()
        running_loss = 0.0

        for images, labels in train_loader:
            images = images.to(DEVICE)
            labels = labels.to(DEVICE)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        average_loss = running_loss / len(train_loader)
        test_acc = evaluate(model, test_loader)

        history.append({
            "epoch": epoch + 1,
            "train_loss": average_loss,
            "test_accuracy": test_acc,
        })

        print(
            f"    [{optimizer_name}] Epoch {epoch + 1}/{EPOCHS} | "
            f"Loss: {average_loss:.4f} | Test Acc: {test_acc:.2f}%"
        )

    end_time = time.time()
    training_time = end_time - start_time

    final_accuracy = history[-1]["test_accuracy"]
    final_loss = history[-1]["train_loss"]

    return history, final_accuracy, final_loss, training_time


# =========================
# 8. 单次实验（Experiment 1 / 2）
# =========================

def run_single_experiment(lr_map, seed, experiment_name):
    """对每个优化器训练一次，返回结果 DataFrame 与 loss 历史字典。"""
    all_losses = {}
    summary_results = []

    for name in OPTIMIZER_NAMES:
        print(f"\n{'=' * 50}")
        print(f"[{experiment_name}] 优化器: {name} | lr={lr_map[name]} | seed={seed}")
        print("=" * 50)

        history, acc, final_loss, elapsed = train_optimizer(
            name, lr_map[name], seed
        )

        all_losses[name] = [h["train_loss"] for h in history]

        # 训练历史记录（training_history.csv）
        for h in history:
            HISTORY_ROWS.append({
                "experiment": experiment_name,
                "optimizer": name,
                "seed": seed,
                "epoch": h["epoch"],
                "train_loss": h["train_loss"],
                "test_accuracy": h["test_accuracy"],
            })

        summary_results.append({
            "optimizer": name,
            "learning_rate": lr_map[name],
            "accuracy": acc,
            "training_time_seconds": elapsed,
            "final_loss": final_loss,
        })

        print(f"{name} 测试准确率: {acc:.2f}% | 训练时间: {elapsed:.2f}s")

    df = pd.DataFrame(summary_results)
    return df, all_losses


# =========================
# 9. Experiment 1：统一学习率
# =========================

def run_experiment1(seed=42):
    print("\n" + "#" * 60)
    print("# Experiment 1: 统一学习率 (lr = 0.001)")
    print("#" * 60)

    same_lr = {name: SAME_LR for name in OPTIMIZER_NAMES}
    df, all_losses = run_single_experiment(same_lr, seed, "exp1_same_lr")

    os.makedirs(RESULTS_DIR, exist_ok=True)
    os.makedirs(FIGURES_DIR, exist_ok=True)

    df.to_csv(os.path.join(RESULTS_DIR, "experiment1_same_lr.csv"), index=False)
    plot_loss_curves(
        all_losses,
        os.path.join(FIGURES_DIR, "experiment1_same_lr_loss.png"),
        title="Experiment 1: Training Loss (Same Learning Rate = 0.001)",
    )
    print("\n[Experiment 1] 完成。")
    return df, all_losses


# =========================
# 10. Experiment 2：合理学习率
# =========================

def run_experiment2(seed=42):
    print("\n" + "#" * 60)
    print("# Experiment 2: 各优化器经验合理学习率")
    print("#" * 60)

    df, all_losses = run_single_experiment(TUNED_LR, seed, "exp2_tuned_lr")

    os.makedirs(RESULTS_DIR, exist_ok=True)
    os.makedirs(FIGURES_DIR, exist_ok=True)

    df.to_csv(os.path.join(RESULTS_DIR, "experiment2_tuned_lr.csv"), index=False)
    plot_loss_curves(
        all_losses,
        os.path.join(FIGURES_DIR, "experiment2_tuned_lr_loss.png"),
        title="Experiment 2: Training Loss (Tuned LR)",
    )
    plot_accuracy_bar(df, os.path.join(FIGURES_DIR, "accuracy_comparison.png"))
    plot_time_bar(df, os.path.join(FIGURES_DIR, "training_time_comparison.png"))
    print("\n[Experiment 2] 完成。")
    return df, all_losses


# =========================
# 11. Experiment 3：多随机种子
# =========================

def run_experiment3(seed_list):
    print("\n" + "#" * 60)
    print(f"# Experiment 3: 多随机种子稳健性实验 (seeds={seed_list})")
    print("#" * 60)

    raw_rows = []

    for seed in seed_list:
        for name in OPTIMIZER_NAMES:
            print(f"\n{'=' * 50}")
            print(f"[exp3] {name} | lr={TUNED_LR[name]} | seed={seed}")
            print("=" * 50)

            history, acc, final_loss, elapsed = train_optimizer(
                name, TUNED_LR[name], seed
            )

            for h in history:
                HISTORY_ROWS.append({
                    "experiment": "exp3_multiseed",
                    "optimizer": name,
                    "seed": seed,
                    "epoch": h["epoch"],
                    "train_loss": h["train_loss"],
                    "test_accuracy": h["test_accuracy"],
                })

            raw_rows.append({
                "optimizer": name,
                "seed": seed,
                "learning_rate": TUNED_LR[name],
                "final_test_accuracy": acc,
                "final_train_loss": final_loss,
                "training_time_seconds": elapsed,
            })

    raw_df = pd.DataFrame(raw_rows)

    # 统计 mean / std
    summary_rows = []
    for name in OPTIMIZER_NAMES:
        sub = raw_df[raw_df["optimizer"] == name]
        summary_rows.append({
            "optimizer": name,
            "learning_rate": TUNED_LR[name],
            "num_seeds": len(sub),
            "accuracy_mean": sub["final_test_accuracy"].mean(),
            "accuracy_std": sub["final_test_accuracy"].std(),
            "loss_mean": sub["final_train_loss"].mean(),
            "loss_std": sub["final_train_loss"].std(),
            "time_mean": sub["training_time_seconds"].mean(),
            "time_std": sub["training_time_seconds"].std(),
        })
    summary_df = pd.DataFrame(summary_rows)

    os.makedirs(RESULTS_DIR, exist_ok=True)
    os.makedirs(FIGURES_DIR, exist_ok=True)

    raw_df.to_csv(os.path.join(RESULTS_DIR, "experiment3_multiseed_raw.csv"), index=False)
    summary_df.to_csv(os.path.join(RESULTS_DIR, "experiment3_multiseed_summary.csv"), index=False)

    # 图 4：多随机种子 Accuracy（mean ± std）
    plot_multiseed_accuracy(summary_df, raw_df)

    # 图 5：测试准确率随 epoch 变化（多种子取 mean）
    plot_test_accuracy_curve(seed_list)

    print("\n[Experiment 3] 完成。")
    print(summary_df.to_string(index=False))
    return raw_df, summary_df


# =========================
# 12. 可视化
# =========================

def plot_sample_images():
    """生成 Fashion-MNIST 样例图。"""
    label_names = ["T-shirt/top", "Trouser", "Pullover", "Dress", "Coat",
                   "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"]
    transform = transforms.ToTensor()
    dataset = datasets.FashionMNIST(root="./data", train=True, download=True,
                                    transform=transform)
    fig, axes = plt.subplots(2, 5, figsize=(10, 4))
    for i, ax in enumerate(axes.flat):
        img, label = dataset[i]
        ax.imshow(img.squeeze(), cmap="gray")
        ax.set_title(label_names[label], fontsize=9)
        ax.axis("off")
    plt.suptitle("Fashion-MNIST Samples", fontsize=14)
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, "fashion_mnist_samples.png"), dpi=300)
    plt.close()


def plot_loss_curves(all_losses, path, title):
    """绘制各优化器的训练 loss 曲线。"""
    plt.figure(figsize=(8, 5))
    for name, losses in all_losses.items():
        plt.plot(range(1, EPOCHS + 1), losses, marker="o", label=name)
    plt.xlabel("Epoch")
    plt.ylabel("Training Loss")
    plt.title(title)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(path, dpi=300)
    plt.close()


def plot_accuracy_bar(df, path):
    """绘制测试准确率柱状图（0-100%，避免夸张视觉差异）。"""
    plt.figure(figsize=(8, 5))
    names = df["optimizer"].tolist()
    accs = df["accuracy"].tolist()
    bars = plt.bar(names, accs, color="#4C72B0")
    for bar, acc in zip(bars, accs):
        plt.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.5,
            f"{acc:.2f}%",
            ha="center", va="bottom", fontsize=10,
        )
    plt.xlabel("Optimizer")
    plt.ylabel("Test Accuracy (%)")
    plt.title("Test Accuracy by Optimizer (Experiment 2)")
    plt.ylim(0, 100)
    plt.grid(True, axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(path, dpi=300)
    plt.close()


def plot_time_bar(df, path):
    """绘制训练时间柱状图。"""
    plt.figure(figsize=(8, 5))
    names = df["optimizer"].tolist()
    times = df["training_time_seconds"].tolist()
    bars = plt.bar(names, times, color="#55A868")
    for bar, t in zip(bars, times):
        plt.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.3,
            f"{t:.2f}s",
            ha="center", va="bottom", fontsize=10,
        )
    plt.xlabel("Optimizer")
    plt.ylabel("Training Time (seconds)")
    plt.title("Training Time by Optimizer (Experiment 2)")
    plt.grid(True, axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(path, dpi=300)
    plt.close()


def plot_multiseed_accuracy(summary_df, raw_df):
    """图 4：多随机种子 Accuracy（mean ± std 误差棒）。"""
    plt.figure(figsize=(8, 5))
    names = summary_df["optimizer"].tolist()
    means = summary_df["accuracy_mean"].tolist()
    stds = summary_df["accuracy_std"].tolist()

    bars = plt.bar(names, means, yerr=stds, capsize=5,
                   color="#DD8452", alpha=0.9)
    # 叠加散点显示每个 seed 的原始值
    for i, name in enumerate(names):
        vals = raw_df[raw_df["optimizer"] == name]["final_test_accuracy"].tolist()
        x = [i] * len(vals)
        plt.scatter(x, vals, color="black", s=20, zorder=3)

    # 在柱顶标 mean±std
    for bar, m, s in zip(bars, means, stds):
        plt.text(
            bar.get_x() + bar.get_width() / 2,
            m + s + 0.4,
            f"{m:.2f}\n±{s:.2f}",
            ha="center", va="bottom", fontsize=9,
        )

    plt.xlabel("Optimizer")
    plt.ylabel("Test Accuracy (%)")
    plt.title("Multi-seed Test Accuracy (mean ± std)")
    plt.ylim(0, 100)
    plt.grid(True, axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, "multiseed_accuracy_comparison.png"), dpi=300)
    plt.close()


def plot_test_accuracy_curve(seed_list):
    """图 5：测试准确率随 epoch 变化（多种子 mean ± std 阴影）。"""
    plt.figure(figsize=(8, 5))
    colors = {"SGD": "#4C72B0", "Momentum": "#55A868",
              "RMSProp": "#C44E52", "Adam": "#8172B2"}

    for name in OPTIMIZER_NAMES:
        # 收集该优化器所有 seed 的每个 epoch 准确率
        per_epoch = []
        for seed in seed_list:
            rows = [r for r in HISTORY_ROWS
                    if r["experiment"] == "exp3_multiseed"
                    and r["optimizer"] == name
                    and r["seed"] == seed]
            rows.sort(key=lambda r: r["epoch"])
            per_epoch.append([r["test_accuracy"] for r in rows])

        arr = np.array(per_epoch)  # shape (num_seeds, EPOCHS)
        mean = arr.mean(axis=0)
        std = arr.std(axis=0)
        epochs = np.arange(1, EPOCHS + 1)

        plt.plot(epochs, mean, marker="o", label=name, color=colors[name])
        plt.fill_between(epochs, mean - std, mean + std,
                         alpha=0.15, color=colors[name])

    plt.xlabel("Epoch")
    plt.ylabel("Test Accuracy (%)")
    plt.title("Test Accuracy vs Epoch (Multi-seed, mean ± std)")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.xticks(range(1, EPOCHS + 1))
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, "test_accuracy_curve.png"), dpi=300)
    plt.close()


# =========================
# 13. 保存训练历史与环境信息
# =========================

def save_history():
    df = pd.DataFrame(HISTORY_ROWS)
    df.to_csv(os.path.join(RESULTS_DIR, "training_history.csv"), index=False)


def save_environment():
    lines = [
        f"python_version={sys.version.split()[0]}",
        f"torch_version={torch.__version__}",
        f"torchvision_version={torchvision.__version__}",
        f"numpy_version={np.__version__}",
        f"pandas_version={pd.__version__}",
        f"matplotlib_version={matplotlib.__version__}",
        f"device={DEVICE}",
        f"cuda_available={torch.cuda.is_available()}",
        f"platform={platform.platform()}",
        f"machine={platform.machine()}",
        f"processor={platform.processor()}",
    ]
    with open(os.path.join(RESULTS_DIR, "environment.txt"), "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


# =========================
# 14. 入口
# =========================

HISTORY_ROWS = []


def main():
    parser = argparse.ArgumentParser(description="Fashion-MNIST 优化器比较实验")
    parser.add_argument("--experiment", default="all",
                        choices=["all", "exp1", "exp2", "multiseed"],
                        help="选择要运行的实验")
    parser.add_argument("--seed", type=int, default=42,
                        help="Experiment 1/2 使用的随机种子（Experiment 3 固定用多种子）")
    parser.add_argument("--seeds", type=int, nargs="*", default=MULTISEED_LIST,
                        help="Experiment 3 使用的随机种子列表")
    args = parser.parse_args()

    print(f"设备: {DEVICE}")

    os.makedirs(RESULTS_DIR, exist_ok=True)
    os.makedirs(FIGURES_DIR, exist_ok=True)

    if args.experiment in ("all", "exp1"):
        run_experiment1(seed=args.seed)
    if args.experiment in ("all", "exp2"):
        run_experiment2(seed=args.seed)
    if args.experiment in ("all", "multiseed"):
        run_experiment3(args.seeds)

    # 生成 Fashion-MNIST 样例图
    plot_sample_images()

    # 保存历史与环境信息
    save_history()
    save_environment()

    print("\n全部实验完成！结果已保存到 results/ 与 figures/ 目录。")


if __name__ == "__main__":
    main()
