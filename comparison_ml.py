import os

import numpy as np
import pandas as pd
import seaborn as sns
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
from matplotlib import pyplot as plt
from pytorch_tabnet.tab_model import TabNetClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, recall_score, confusion_matrix
from sklearn.model_selection import GridSearchCV

from G_linear_to_G_i import train_test_split_and_ros
from get_data import get_processed_data


if __name__ == "__main__":
    # 数据加载和预处理
    raw_data = os.path.join(os.path.dirname(__file__), 'data', 'nhtsa.csv')
    x, x_norm, y = get_processed_data(raw_data)
    del raw_data
    print("原始标签分布:")
    print(y.value_counts())

    # 标签重映射
    y = y.replace(1, 0)
    y = y.replace(2, 1)
    print("映射后的标签分布:")
    print(y.value_counts())
    print("数据预处理完成！")

    # 数据拆分
    x_train_resampled, x_test, y_train_resampled, y_test = train_test_split_and_ros(x_norm, y)

    # 定义模型和参数搜索范围
    param_grids = {
        "RandomForest": {
            "n_estimators": [50, 100, 200],
            "max_depth": [None, 10, 20],
            "min_samples_split": [2, 5],
            "min_samples_leaf": [1, 2],
        },
        "LightGBM": {
            "n_estimators": [50, 100, 200],
            "learning_rate": [0.01, 0.1, 0.2],
            "max_depth": [-1, 10, 20],
        },
        "CatBoost": {
            "iterations": [50, 100, 200],
            "learning_rate": [0.01, 0.1, 0.2],
            "depth": [4, 6, 8],
        },
        "TabNet": {
            "n_d": [8, 16, 24],  # Feature transformer dimension
            "n_a": [8, 16, 24],  # Attention dimension
            "n_steps": [3, 5],  # Number of decision steps
            "gamma": [1.0],  # Relaxation factor
            "momentum": [0.02],  # Momentum for batch normalization
        }
    }

    models = {
        "RandomForest": RandomForestClassifier(class_weight={0: 1, 1: 5}, random_state=42),
        "LightGBM": LGBMClassifier(class_weight={0: 1, 1: 5}, random_state=42),
        "CatBoost": CatBoostClassifier(class_weights=[1, 5], random_seed=42),
        "TabNet": TabNetClassifier(seed=42)
    }

    # TabNet 需要 NumPy 数组
    x_train_resampled_np = x_train_resampled.values if isinstance(x_train_resampled,
                                                                  pd.DataFrame) else x_train_resampled
    x_test_np = x_test.values if isinstance(x_test, pd.DataFrame) else x_test
    y_train_resampled_np = y_train_resampled.values if isinstance(y_train_resampled, pd.Series) else y_train_resampled
    y_test_np = y_test.values if isinstance(y_test, pd.Series) else y_test

    # 超参数搜索和评估
    best_params = {}
    for name, model in models.items():
        print(f"正在调整 {name} 的超参数...")
        if name != "TabNet":
            grid_search = GridSearchCV(model, param_grids[name], cv=3, scoring='f1', n_jobs=-1, verbose=1)
            grid_search.fit(x_train_resampled, y_train_resampled)
            best_params[name] = grid_search.best_params_
            print(f"{name} 最佳超参数: {grid_search.best_params_}")
        else:
            best_score = -np.inf
            best_params[name] = None
            for n_d in param_grids["TabNet"]["n_d"]:
                for n_a in param_grids["TabNet"]["n_a"]:
                    for n_steps in param_grids["TabNet"]["n_steps"]:
                        for gamma in param_grids["TabNet"]["gamma"]:
                            for momentum in param_grids["TabNet"]["momentum"]:
                                model = TabNetClassifier(
                                    n_d=n_d, n_a=n_a, n_steps=n_steps,
                                    gamma=gamma, momentum=momentum, seed=42
                                )
                                model.fit(
                                    x_train_resampled_np, y_train_resampled_np,
                                    eval_set=[(x_test_np, y_test_np)],
                                    eval_name=["val"],
                                    eval_metric=["auc"]
                                )
                                y_pred = model.predict(x_test_np)
                                f1 = f1_score(y_test, y_pred)
                                if f1 > best_score:
                                    best_score = f1
                                    best_params[name] = {
                                        "n_d": n_d, "n_a": n_a,
                                        "n_steps": n_steps, "gamma": gamma, "momentum": momentum
                                    }
            print(f"{name} 最佳超参数: {best_params[name]}")

    results = []
    for name, model in models.items():
        print(f"正在评估 {name}...")
        if name != "TabNet":
            model.set_params(**best_params[name])
            model.fit(x_train_resampled, y_train_resampled)
        else:
            model = TabNetClassifier(**best_params[name], seed=42)
            model.fit(
                x_train_resampled_np, y_train_resampled_np,
                eval_set=[(x_test_np, y_test_np)],
                eval_name=["val"],
                eval_metric=["auc"],
            )
        y_pred = model.predict(x_test_np)

        # 计算混淆矩阵
        cm = confusion_matrix(y_test_np, y_pred)
        cm_df = pd.DataFrame(cm, index=["Actual 0", "Actual 1"], columns=["Predicted 0", "Predicted 1"])

        # 保存混淆矩阵图
        plt.figure(figsize=(6, 4))
        sns.heatmap(cm_df, annot=True, fmt="d", cmap="Blues")
        plt.title(f"{name} Confusion Matrix")
        plt.ylabel("Actual")
        plt.xlabel("Predicted")
        cm_path = os.path.join(os.path.dirname(__file__), f'save_info/compare_ml/{name}_confusion_matrix.png')
        plt.savefig(cm_path)
        plt.close()
        print(f"{name} 混淆矩阵已保存到文件: {cm_path}")

        # 计算指标
        accuracy = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred, average='weighted')
        recall = recall_score(y_test, y_pred, pos_label=1)
        results.append({"Model": name, "Accuracy": accuracy, "F1 Score": f1, "Recall (Class 1)": recall})
        print(f"{name} - Accuracy: {accuracy:.4f}, F1 Score: {f1:.4f}, Recall (Class 1): {recall:.4f}")

    # 保存结果
    results_df = pd.DataFrame(results)
    output_path = os.path.join(os.path.dirname(__file__), 'save_info/compare_ml/model_comparison_results.csv')
    results_df.to_csv(output_path, index=False)
    print(f"\n模型性能对比结果已保存到文件: {output_path}")