# -*- coding: utf-8 -*-
"""
Fashion-MNIST 优化器比较实验
==============================

本脚本在 Fashion-MNIST 数据集上训练一个简单的 MLP 模型，
比较 4 种优化器（SGD、Momentum、RMSProp、Adam）的表现。

输出文件：
    - results/experiment2_tuned_lr.csv   -> 各优化器的最终结果表
    - figures/experiment2_tuned_lr_loss.png -> 各优化器的训练 loss 曲线
    - figures/accuracy_comparison.png    -> 测试准确率柱状图
    - figures/training_time_comparison.png -> 训练时间柱状图
"""

import os
import time

import matplotlib
matplotlib.use("Agg")  # 无界面环境下也能保存图片

import matplotlib.pyplot as plt
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms


# =========================
# 1. 基础设置
# =========================

# 固定随机种子，保证每次运行结果可复现
SEED = 42
torch.manual_seed(SEED)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("当前设备:", device)

BATCH_SIZE = 64
EPOCHS = 10

# 每个优化器使用各自调优后的学习率
LEARNING_RATES = {
    "SGD": 0.1,
    "Momentum": 0.01,
    "RMSProp": 0.001,
    "Adam": 0.001,
}

# 输出路径
RESULTS_DIR = "results"
FIGURES_DIR = "figures"

RESULTS_CSV = os.path.join(RESULTS_DIR, "experiment2_tuned_lr.csv")
LOSS_FIG = os.path.join(FIGURES_DIR, "experiment2_tuned_lr_loss.png")
ACCURACY_FIG = os.path.join(FIGURES_DIR, "accuracy_comparison.png")
TIME_FIG = os.path.join(FIGURES_DIR, "training_time_comparison.png")


# =========================
# 2. 数据集
# =========================

transform = transforms.ToTensor()

train_dataset = datasets.FashionMNIST(
    root="./data",
    train=True,
    download=True,
    transform=transform,
)

test_dataset = datasets.FashionMNIST(
    root="./data",
    train=False,
    download=True,
    transform=transform,
)

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
)

test_loader = DataLoader(
    test_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
)


# =========================
# 3. 定义 MLP 模型
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
# 4. 创建优化器
# =========================

def create_optimizer(name, model):
    """根据名称创建对应的优化器。"""

    lr = LEARNING_RATES[name]

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
# 5. 测试函数
# =========================

def evaluate(model):
    """在测试集上计算准确率（百分比）。"""

    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            _, predicted = torch.max(outputs, 1)

            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    return 100 * correct / total


# =========================
# 6. 训练单个优化器
# =========================

def train_optimizer(optimizer_name):
    """
    用指定优化器训练一个全新的模型，返回：
        losses: 每个 epoch 的平均训练 loss 列表
        accuracy: 测试集准确率
        training_time: 训练耗时（秒）
    """

    print()
    print("=" * 50)
    print(f"开始训练优化器: {optimizer_name}")
    print(f"学习率: {LEARNING_RATES[optimizer_name]}")
    print("=" * 50)

    # 关键：每个优化器都从完全相同的模型初始化开始，
    # 这样比较结果才公平。
    torch.manual_seed(SEED)
    model = MLP().to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = create_optimizer(optimizer_name, model)

    losses = []
    start_time = time.time()

    for epoch in range(EPOCHS):
        model.train()
        running_loss = 0.0

        for images, labels in train_loader:
            images = images.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        average_loss = running_loss / len(train_loader)
        losses.append(average_loss)

        print(
            f"{optimizer_name} | "
            f"Epoch [{epoch + 1}/{EPOCHS}] | "
            f"Loss: {average_loss:.4f}"
        )

    end_time = time.time()
    training_time = end_time - start_time

    accuracy = evaluate(model)

    print(f"{optimizer_name} 测试准确率: {accuracy:.2f}%")
    print(f"{optimizer_name} 训练时间: {training_time:.2f} 秒")

    return losses, accuracy, training_time


# =========================
# 7. 比较多个优化器
# =========================

def run_experiment():
    """依次训练所有优化器，收集结果并保存。"""

    optimizer_names = [
        "SGD",
        "Momentum",
        "RMSProp",
        "Adam",
    ]

    all_losses = {}
    summary_results = []

    for optimizer_name in optimizer_names:
        losses, accuracy, training_time = train_optimizer(optimizer_name)

        all_losses[optimizer_name] = losses
        summary_results.append(
            {
                "optimizer": optimizer_name,
                "learning_rate": LEARNING_RATES[optimizer_name],
                "accuracy": accuracy,
                "training_time_seconds": training_time,
                "final_loss": losses[-1],
            }
        )

    # 确保输出目录存在
    os.makedirs(RESULTS_DIR, exist_ok=True)
    os.makedirs(FIGURES_DIR, exist_ok=True)

    # 保存结果 CSV
    results_df = pd.DataFrame(summary_results)
    results_df.to_csv(RESULTS_CSV, index=False)

    print()
    print("=" * 50)
    print("最终结果")
    print("=" * 50)
    print(results_df)

    # 绘制 loss 曲线
    plot_loss_curves(all_losses)

    # 绘制准确率与训练时间柱状图
    plot_accuracy_bar(results_df)
    plot_time_bar(results_df)

    print()
    print("实验完成！")
    print(f"结果表已保存到: {RESULTS_CSV}")
    print(f"Loss 曲线已保存到: {LOSS_FIG}")
    print(f"准确率柱状图已保存到: {ACCURACY_FIG}")
    print(f"训练时间柱状图已保存到: {TIME_FIG}")


# =========================
# 8. 可视化
# =========================

def plot_loss_curves(all_losses):
    """绘制各优化器的训练 loss 曲线。"""

    plt.figure(figsize=(10, 6))

    for optimizer_name, losses in all_losses.items():
        plt.plot(
            range(1, EPOCHS + 1),
            losses,
            marker="o",
            label=optimizer_name,
        )

    plt.xlabel("Epoch")
    plt.ylabel("Training Loss")
    plt.title("Fashion-MNIST Optimizer Loss Comparison")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(LOSS_FIG, dpi=300)
    plt.close()


def plot_accuracy_bar(results_df):
    """绘制各优化器的测试准确率柱状图。"""

    plt.figure(figsize=(8, 5))

    names = results_df["optimizer"].tolist()
    accuracies = results_df["accuracy"].tolist()

    bars = plt.bar(names, accuracies, color="#4C72B0")

    # 在柱子上方标注数值
    for bar, acc in zip(bars, accuracies):
        plt.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.1,
            f"{acc:.2f}%",
            ha="center",
            va="bottom",
            fontsize=10,
        )

    plt.xlabel("Optimizer")
    plt.ylabel("Test Accuracy (%)")
    plt.title("Test Accuracy by Optimizer")
    plt.ylim(0, 100)
    plt.grid(True, axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(ACCURACY_FIG, dpi=300)
    plt.close()


def plot_time_bar(results_df):
    """绘制各优化器的训练时间柱状图。"""

    plt.figure(figsize=(8, 5))

    names = results_df["optimizer"].tolist()
    times = results_df["training_time_seconds"].tolist()

    bars = plt.bar(names, times, color="#55A868")

    # 在柱子上方标注数值
    for bar, t in zip(bars, times):
        plt.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.2,
            f"{t:.2f}s",
            ha="center",
            va="bottom",
            fontsize=10,
        )

    plt.xlabel("Optimizer")
    plt.ylabel("Training Time (seconds)")
    plt.title("Training Time by Optimizer")
    plt.grid(True, axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(TIME_FIG, dpi=300)
    plt.close()


# =========================
# 入口
# =========================

if __name__ == "__main__":
    run_experiment()
