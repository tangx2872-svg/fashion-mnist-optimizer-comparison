# -*- coding: utf-8 -*-
"""
结果一致性自动检查脚本
========================

检查项目产出的 CSV、图表、报告引用是否完整且一致。

用法：
    python scripts/check_results.py

退出码：
    0  -> 全部通过（PASS）
    1  -> 存在错误（FAIL）
"""

import os
import re
import sys

import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS = os.path.join(ROOT, "results")
FIGURES = os.path.join(ROOT, "figures")
REPORT = os.path.join(ROOT, "report", "机器学习优化器比较实验报告.md")
README = os.path.join(ROOT, "README.md")

OPTIMIZERS = ["SGD", "Momentum", "RMSProp", "Adam"]

errors = []
warnings = []


def check(cond, msg):
    if cond:
        print(f"  [PASS] {msg}")
    else:
        print(f"  [FAIL] {msg}")
        errors.append(msg)


def warn(msg):
    print(f"  [WARN] {msg}")
    warnings.append(msg)


def main():
    print("=" * 60)
    print("结果一致性检查")
    print("=" * 60)

    # 1. 必需文件是否存在
    print("\n[1] 必需文件是否存在")
    required_files = [
        os.path.join(RESULTS, "experiment1_same_lr.csv"),
        os.path.join(RESULTS, "experiment2_tuned_lr.csv"),
        os.path.join(RESULTS, "experiment3_multiseed_raw.csv"),
        os.path.join(RESULTS, "experiment3_multiseed_summary.csv"),
        os.path.join(RESULTS, "training_history.csv"),
        os.path.join(RESULTS, "environment.txt"),
        os.path.join(FIGURES, "fashion_mnist_samples.png"),
        os.path.join(FIGURES, "experiment1_same_lr_loss.png"),
        os.path.join(FIGURES, "experiment2_tuned_lr_loss.png"),
        os.path.join(FIGURES, "accuracy_comparison.png"),
        os.path.join(FIGURES, "training_time_comparison.png"),
        os.path.join(FIGURES, "multiseed_accuracy_comparison.png"),
        os.path.join(FIGURES, "test_accuracy_curve.png"),
    ]
    for f in required_files:
        check(os.path.isfile(f), f"存在 {os.path.relpath(f, ROOT)}")

    # 2. CSV 可读且无 NaN
    print("\n[2] CSV 数据完整性")
    for f in required_files:
        if not f.endswith(".csv") or not os.path.isfile(f):
            continue
        df = pd.read_csv(f)
        check(not df.isna().any().any(),
              f"{os.path.basename(f)} 无 NaN")

    # 3. 优化器数量与名称
    print("\n[3] 优化器数量与名称")
    exp2 = pd.read_csv(os.path.join(RESULTS, "experiment2_tuned_lr.csv"))
    check(sorted(exp2["optimizer"].tolist()) == sorted(OPTIMIZERS),
          "experiment2 含 4 个正确优化器")

    raw3 = pd.read_csv(os.path.join(RESULTS, "experiment3_multiseed_raw.csv"))
    check(sorted(raw3["optimizer"].unique().tolist()) == sorted(OPTIMIZERS),
          "experiment3 含 4 个正确优化器")

    # 4. 多随机种子数量（应为 optimizer × seed = 12 行）
    print("\n[4] 多随机种子覆盖")
    seeds = sorted(raw3["seed"].unique().tolist())
    expected_rows = len(OPTIMIZERS) * len(seeds)
    check(len(raw3) == expected_rows,
          f"experiment3 行数 = {len(raw3)} (4 优化器 × {len(seeds)} 种子)")
    check(len(seeds) >= 3, f"种子数量 >= 3（当前 {len(seeds)}: {seeds}）")

    # 5. accuracy 范围合法（0-100）
    print("\n[5] accuracy 范围")
    exp1 = pd.read_csv(os.path.join(RESULTS, "experiment1_same_lr.csv"))
    for df, name in [(exp1, "exp1"), (exp2, "exp2")]:
        ok = (df["accuracy"] >= 0).all() and (df["accuracy"] <= 100).all()
        check(ok, f"{name} accuracy 在 [0, 100] 内")
    ok3 = (raw3["final_test_accuracy"] >= 0).all() and \
          (raw3["final_test_accuracy"] <= 100).all()
    check(ok3, "exp3 accuracy 在 [0, 100] 内")

    # 6. 训练历史 epoch 覆盖
    print("\n[6] 训练历史 epoch 覆盖")
    hist = pd.read_csv(os.path.join(RESULTS, "training_history.csv"))
    check(set(hist["experiment"].unique()) >= {"exp1_same_lr", "exp2_tuned_lr", "exp3_multiseed"},
          "training_history 覆盖三组实验")
    check(sorted(hist["epoch"].unique().tolist()) == list(range(1, 11)),
          "epoch 覆盖 1-10")

    # 7. 报告引用图片是否存在
    print("\n[7] 报告引用图片")
    if os.path.isfile(REPORT):
        report_text = open(REPORT, encoding="utf-8").read()
        imgs = re.findall(r"!\[[^\]]*\]\(([^)]+)\)", report_text)
        for img in imgs:
            # 去掉 ../ 前缀
            p = img.replace("../", "").replace("./", "")
            full = os.path.join(ROOT, p)
            check(os.path.isfile(full), f"报告引用图片存在: {img}")
        if not imgs:
            warn("报告未发现图片引用")
    else:
        check(False, "报告文件存在")

    # 8. README 与报告中的训练时间是否与 CSV 一致
    print("\n[8] 训练时间一致性")
    exp2 = pd.read_csv(os.path.join(RESULTS, "experiment2_tuned_lr.csv"))
    csv_times = {r["optimizer"]: r["training_time_seconds"]
                 for _, r in exp2.iterrows()}
    # 检查 README 是否还残留旧的错误时间（19.11/19.53/21.61/26.40）
    bad_times = ["19.11", "19.53", "21.61", "26.40"]
    if os.path.isfile(README):
        readme_text = open(README, encoding="utf-8").read()
        for bt in bad_times:
            if bt in readme_text:
                warn(f"README 中仍含旧训练时间 {bt}")

    # 9. 汇总
    print("\n" + "=" * 60)
    if errors:
        print(f"检查结果：FAIL（{len(errors)} 个错误，{len(warnings)} 个警告）")
        for e in errors:
            print(f"  - {e}")
        sys.exit(1)
    else:
        print(f"检查结果：PASS（{len(warnings)} 个警告）")
        for w in warnings:
            print(f"  ! {w}")
        sys.exit(0)


if __name__ == "__main__":
    main()
