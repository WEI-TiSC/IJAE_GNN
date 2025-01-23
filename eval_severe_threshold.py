import os

import torch
from matplotlib import pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix
import numpy as np

from Gi_GAT_learning import GATModel


# params
linear_threshold = 90
# Do not change - model params
batch_size = 512
epochs = 80
input_dim = 1  # 节点初始特征维度
hidden_dim = 128
output_dim = 2  # 二分类
edge_dim = 3  # 边属性的维度
num_layers = 5
dropout = 0.2
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 加载保存的测试数据和最优模型
test_data_path = "save_info/test_data/full_gi_test.pth"
model_path = f"save_info/saved_models/linear_threshold_{100 - linear_threshold}_epochs_{epochs}/focal/best_gat_model.pth"

# 加载数据和模型
Gi_test = torch.load(test_data_path)
model = GATModel(input_dim, hidden_dim, output_dim, edge_dim, num_layers=num_layers, dropout=dropout).to(device)
model.load_state_dict(torch.load(model_path))
model.eval()
print(f"Loaded best model from {model_path} and test data from {test_data_path}.")

# 定义要测试的阈值范围
thresholds = np.arange(0, 0.52, 0.03)

# 存储不同阈值下的分类结果
threshold_results = {}

for threshold in thresholds:
    y_true = []
    y_pred = []

    # 遍历测试集样本，按阈值预测
    with torch.no_grad():
        for batch in Gi_test:
            batch = batch.to(device)
            out = model(batch.x, batch.edge_index, batch.edge_attr, batch.batch)
            probs = torch.softmax(out, dim=1)
            pred = (probs[:, 1] > threshold).long()
            y_true.extend(batch.y.view(-1).cpu().numpy())
            y_pred.extend(pred.cpu().numpy())

    # Calc metrics
    report = classification_report(y_true, y_pred, target_names=["Non-severe", "Severe"], digits=4, output_dict=True)
    precision_global = report["weighted avg"]["precision"]  # 全局 Precision
    f1_global = report["weighted avg"]["f1-score"]  # 全局 F1-score
    recall_severe = report["Severe"]["recall"]  # 类别 1 Recall

    # Save results
    threshold_results[threshold] = {
        "Precision (Global)": precision_global,
        "Recall (Severe)": recall_severe,
        "F1-score (Global)": f1_global
    }
    print(
        f"Threshold: {threshold:.2f} - Precision (Global): {precision_global:.4f}, Recall (Severe): {recall_severe:.4f}, F1-score (Global): {f1_global:.4f}")

thresholds = list(threshold_results.keys())
precisions = [threshold_results[t]["Precision (Global)"] for t in thresholds]
recalls = [threshold_results[t]["Recall (Severe)"] for t in thresholds]
f1_scores = [threshold_results[t]["F1-score (Global)"] for t in thresholds]

# Draw
plt.figure(figsize=(10, 6))
plt.plot(thresholds, precisions, marker="o", label="Precision (Global)")
plt.plot(thresholds, recalls, marker="o", label="Recall (Severe)")
plt.plot(thresholds, f1_scores, marker="o", label="F1-score (Global)")
plt.xlabel("Threshold")
plt.ylabel("Score")
plt.title("Classification Metrics under Different Thresholds")
plt.legend()
plt.grid()

save_dir = f"save_info/eval_threshold/linear_threshold_{100 - linear_threshold}_epochs_{epochs}/"
if not os.path.exists(save_dir):
    os.makedirs(save_dir)
plt.savefig(os.path.join(save_dir, f"threshold_analysis.png"))
plt.show()
