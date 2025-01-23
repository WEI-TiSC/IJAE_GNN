import os

import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import classification_report, confusion_matrix
from torch_geometric.loader import DataLoader
from torch_geometric.nn import GATConv, global_mean_pool, BatchNorm

from G_linear_to_G_i import init_to_gi
from focal_loss import FocalLoss
from utils import plot_and_save_curve, plot_and_save_confusion_matrix, EarlyStopping


class GATModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, edge_dim, heads=4, num_layers=3, dropout=0.2):
        super(GATModel, self).__init__()
        self.num_layers = num_layers
        self.dropout = dropout

        self.gat_layers = nn.ModuleList()
        self.batch_norms = nn.ModuleList()

        self.residual_projections = nn.ModuleList()

        self.gat_layers.append(
            GATConv(input_dim, hidden_dim, heads=heads, concat=True, edge_dim=edge_dim)
        )
        self.batch_norms.append(BatchNorm(hidden_dim * heads))
        self.residual_projections.append(nn.Linear(input_dim, hidden_dim * heads))

        for _ in range(num_layers - 2):
            self.gat_layers.append(
                GATConv(hidden_dim * heads, hidden_dim, heads=heads, concat=True, edge_dim=edge_dim)
            )
            self.batch_norms.append(BatchNorm(hidden_dim * heads))
            self.residual_projections.append(nn.Linear(hidden_dim * heads, hidden_dim * heads))

        self.gat_layers.append(
            GATConv(hidden_dim * heads, hidden_dim, heads=1, concat=False, edge_dim=edge_dim)
        )
        self.batch_norms.append(BatchNorm(hidden_dim))
        self.residual_projections.append(nn.Linear(hidden_dim * heads, hidden_dim))

        self.fc1 = nn.Linear(hidden_dim, hidden_dim // 2)
        self.fc2 = nn.Linear(hidden_dim // 2, output_dim)

    def forward(self, x, edge_index, edge_attr, batch):
        for i in range(self.num_layers):
            residual = x
            residual = self.residual_projections[i](residual)

            x = self.gat_layers[i](x, edge_index, edge_attr=edge_attr)
            x = F.relu(x)
            x = self.batch_norms[i](x)
            x = F.dropout(x, p=self.dropout, training=self.training)
            x += residual

        x = global_mean_pool(x, batch)

        x = F.relu(self.fc1(x))
        x = F.dropout(x, p=self.dropout, training=self.training)
        out = self.fc2(x)
        return out


if __name__ == "__main__":
    raw_data = os.path.join(os.path.dirname(__file__), 'data', 'nhtsa.csv')

    # G_linear edge percentage
    threshold = 70  # leave (100-threshold) edges
    # positive_threshold = 0.4  # 0.5 default!
    loss_criterion = 'focal'  # or 'CE'

    # Params in training
    batch_size = 512
    epochs = 80
    input_dim = 1  # 节点初始特征维度
    hidden_dim = 128
    output_dim = 2  # 二分类
    edge_dim = 3  # 边属性的维度
    num_layers = 5
    dropout = 0.2
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    Gi_train_resampled, Gi_test = init_to_gi(raw_data, threshold)

    # 创建 DataLoader
    train_loader = DataLoader(Gi_train_resampled, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(Gi_test, batch_size=batch_size, shuffle=False)

    model = GATModel(input_dim, hidden_dim, output_dim, edge_dim, num_layers=num_layers, dropout=dropout).to(
        device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)
    # # Init Early Stopping
    # early_stopping = EarlyStopping(
    #     patience=30,
    #     min_delta=0.0001,
    #     path="save_info/saved_models/best_model.pth"
    # )

    if loss_criterion == 'focal':  # Use focal loss!
        class_weights = torch.tensor([1.0, 20.0], dtype=torch.float).to(device)
        criterion = FocalLoss(alpha=1, gamma=4, weight=class_weights)
    else:
        criterion = torch.nn.CrossEntropyLoss()  # Binary classification loss

    # Record for visulization
    losses = []
    f1_scores = []
    severe_recalls = []

    # 初始化记录最优训练 loss 的变量
    best_train_loss = float("inf")  # 初始化为正无穷
    best_epoch = 0  # 记录最佳 epoch

    # Save test data for analysis
    torch.save(Gi_test, "save_info/test_data/full_gi_test.pth")
    print("Gi_test saved successfully.")

    for epoch in range(epochs):
        # Training phase
        model.train()
        total_loss = 0
        batch_count = 0
        for batch in train_loader:
            batch = batch.to(device)
            optimizer.zero_grad()
            out = model(batch.x, batch.edge_index, batch.edge_attr, batch.batch)
            loss = criterion(out, batch.y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            batch_count += 1
        average_loss = total_loss / batch_count
        losses.append(average_loss)  # 记录平均损失
        print(f"Epoch {epoch + 1}, Average Loss: {average_loss:.4f}")

        # 检查是否是最佳训练 Loss
        if average_loss < best_train_loss:
            best_train_loss = average_loss
            best_epoch = epoch + 1  # 记录最佳 epoch
            best_model_parent_path = f"save_info/saved_models/linear_threshold_{100 - threshold}_epochs_{epochs}/{loss_criterion}/"
            if not os.path.exists(best_model_parent_path):
                os.makedirs(best_model_parent_path)
            best_model_path = os.path.join(best_model_parent_path, 'best_gat_model.pth')
            torch.save(model.state_dict(), best_model_path)  # 保存最佳模型
            print(f"New best model saved at epoch {epoch + 1} with Training Loss: {best_train_loss:.4f}")

        # Validation phase
        model.eval()
        y_true = []
        y_pred = []
        with torch.no_grad():
            for batch in test_loader:
                batch = batch.to(device)
                out = model(batch.x, batch.edge_index, batch.edge_attr, batch.batch)
                pred = out.argmax(dim=1)
                y_true.extend(batch.y.cpu().numpy())
                y_pred.extend(pred.cpu().numpy())

        # 计算 F1-score
        report = classification_report(y_true, y_pred, target_names=["Non-severe", "Severe"], digits=4,
                                       output_dict=True)
        f1_score = report['weighted avg']['f1-score']
        f1_scores.append(f1_score)
        print(f"Epoch {epoch + 1}, Weighted F1 Score: {f1_score:.4f}")

        recall_severe = report['Severe']['recall']  # 提取类别 1 的 Recall
        severe_recalls.append(recall_severe)  # 记录 Recall
        print(f"Epoch {epoch + 1}, Recall for Class 1 (Severe): {recall_severe:.4f}")

    # 保存损失和 F1-score 曲线
    save_dir = f"save_info/learning_curves/{loss_criterion}/threshold_{100 - threshold}_epochs_{epochs}/"
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    plot_and_save_curve(
        losses,
        title="Training Loss Curve",
        xlabel="Epoch",
        ylabel="Loss",
        save_path=os.path.join(save_dir, f"training_loss_curve_{100 - threshold}.png")
    )

    plot_and_save_curve(
        f1_scores,
        title="Validation F1-score Curve",
        xlabel="Epoch",
        ylabel="F1-score",
        save_path=os.path.join(save_dir, f"validation_f1_score_curve_{100 - threshold}.png")
    )

    plot_and_save_curve(
        severe_recalls,
        title="Validation Recall for Class 1 Curve",
        xlabel="Epoch",
        ylabel="Severe Recall",
        save_path=os.path.join(save_dir, f"validation_severe_recall_curve_{100 - threshold}.png")
    )

    best_model_path = os.path.join(
        f"save_info/saved_models/linear_threshold_{100 - threshold}_epochs_{epochs}/{loss_criterion}/best_gat_model.pth"
    )
    model.load_state_dict(torch.load(best_model_path))
    print(f"Loaded best model from epoch {best_epoch} for final evaluation.")

    # 用最佳模型重新进行预测
    model.eval()
    y_true = []
    y_pred = []
    with torch.no_grad():
        for batch in test_loader:
            batch = batch.to(device)
            out = model(batch.x, batch.edge_index, batch.edge_attr, batch.batch)
            pred = out.argmax(dim=1)
            y_true.extend(batch.y.cpu().numpy())
            y_pred.extend(pred.cpu().numpy())

    # 计算最终分类报告
    report = classification_report(y_true, y_pred, target_names=["Non-severe", "Severe"], digits=4)
    print(f"Classification Report with Best Model:\n{report}")

    # 保存最终分类报告
    final_report_path = os.path.join(save_dir, f"final_classification_report_{100 - threshold}.txt")
    with open(final_report_path, "w") as f:
        f.write(f"Classification Report with Best Model:\n{report}")
    print(f"Final classification report saved at: {final_report_path}")

    # 计算最终混淆矩阵
    cm = confusion_matrix(y_true, y_pred)
    plot_and_save_confusion_matrix(
        cm,
        class_names=["Non-severe", "Severe"],
        title="Confusion Matrix with Best Model",
        save_path=os.path.join(save_dir, f"confusion_matrix_best_model_{100 - threshold}.png")
    )

    # Save f1 and recall for comparison
    metrics_df = pd.DataFrame({
        "Epoch": list(range(1, len(f1_scores) + 1)),
        "F1_Score": f1_scores,
        "Severe_Recall": severe_recalls
    })

    # 保存为 CSV
    metrics_csv_path = os.path.join(save_dir, f"metrics_{100 - threshold}.csv")
    metrics_df.to_csv(metrics_csv_path, index=False)
    print(f"Metrics saved to {metrics_csv_path}")