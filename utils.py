import matplotlib.pyplot as plt
import torch
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns


# 绘制曲线的辅助函数
def plot_and_save_curve(values, title, xlabel, ylabel, save_path):
    """
    绘制并保存曲线图
    参数:
        values: list，曲线的纵轴值
        title: str，图的标题
        xlabel: str，横轴标签
        ylabel: str，纵轴标签
        save_path: str，保存路径
    """
    plt.figure(figsize=(8, 6))
    plt.plot(range(1, len(values) + 1), values, marker='o')
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.grid()
    plt.savefig(save_path, dpi=300)
    plt.close()


def plot_and_save_confusion_matrix(cm, class_names, title, save_path):
    """
    绘制并保存混淆矩阵
    参数:
        cm: ndarray，混淆矩阵
        class_names: list，类别名称
        title: str，图的标题
        save_path: str，保存路径
    """
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
    plt.title(title)
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()


class EarlyStopping:
    def __init__(self, patience=30, min_delta=0, path="best_model.pth"):
        """
        参数:
        - patience: int, 没有改进的最大容忍轮数。
        - min_delta: float, 改进的最小幅度。
        - path: str, 保存最佳模型的路径。
        """
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.path = path

    def __call__(self, val_loss, model):
        if self.best_score is None:
            self.best_score = val_loss
            self.save_checkpoint(model)
        elif val_loss > self.best_score - self.min_delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = val_loss
            self.save_checkpoint(model)
            self.counter = 0

    def save_checkpoint(self, model):
        """保存当前最佳模型"""
        torch.save(model.state_dict(), self.path)
        print(f"Model saved at {self.path}")
