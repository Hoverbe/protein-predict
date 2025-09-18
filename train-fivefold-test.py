import sys
import os
import pandas as pd
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, roc_auc_score
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import numpy as np
# 加入了学习率退火和耐心值
# 将项目根目录添加到Python路径
# project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
# print(f'Project root: {project_root}')
# print(f'Current sys.path: {sys.path}')
# if project_root not in sys.path:
#     sys.path.append(project_root)
#     print(f'Added {project_root} to sys.path')
# else:
#     print(f'{project_root} already in sys.path')

project_root = os.path.dirname(__file__)

# 检查esm目录是否存在
esm_path = os.path.join(project_root, 'model')
print(f'ESM path exists: {os.path.exists(esm_path)}')
print(f'ESM path: {esm_path}')

from esm.models.esmc import ESMC
from esm.tokenization.sequence_tokenizer import EsmSequenceTokenizer
from esm.utils.encoding import tokenize_sequence

# 设置随机种子以确保可重复性
torch.manual_seed(42)

# 1. 数据加载与预处理
class ProteinDataset(Dataset):
    def __init__(self, sequences, labels, tokenizer, max_length=512):
        self.sequences = sequences
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        sequence = self.sequences[idx]
        label = self.labels[idx]

        # 对序列进行tokenization
        tokens = tokenize_sequence(sequence, self.tokenizer, add_special_tokens=True)

        # 截断或填充到最大长度
        if len(tokens) > self.max_length:
            tokens = tokens[:self.max_length]
        elif len(tokens) < self.max_length:
            tokens = torch.cat([tokens, torch.full((self.max_length - len(tokens),), self.tokenizer.pad_token_id)])

        return tokens, torch.tensor(label, dtype=torch.float32)

# 读取五折数据集
def load_fold_data(file_path):
    df = pd.read_csv(file_path)
    return df

# 读取测试集数据
def load_test_data(file_path):
    df = pd.read_csv(file_path)
    sequences = df['seq'].tolist()
    labels = df['label'].tolist()
    return sequences, labels

# 2. 模型定义
class ProteinClassifier(nn.Module):
    def __init__(self, esmc_model, hidden_dim=256, dropout=0.5):
        super().__init__()
        self.esmc = esmc_model
        self.classifier = nn.Sequential(
            nn.Linear(esmc_model.embed.embedding_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()
        )

    def forward(self, tokens):
        # 获取ESMC模型的输出
        output = self.esmc(tokens)
        # 使用序列表示的平均值作为分类特征
        embeddings = output.embeddings.mean(dim=1)
        # 分类预测
        logits = self.classifier(embeddings)
        return logits

# 3. 训练函数
def train_epoch(model, dataloader, criterion, optimizer, device):
    model.train()
    total_loss = 0
    all_preds = []
    all_labels = []
    all_probs = []

    for tokens, labels in tqdm(dataloader, desc="Training", leave=False):
        tokens = tokens.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        # 前向传播
        outputs = model(tokens)
        # 确保输出和标签具有相同的形状，只移除批次维度以外的单维度
        outputs = outputs.squeeze(dim=1) if outputs.dim() > 1 else outputs
        loss = criterion(outputs, labels)

        # 反向传播和优化
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

        # 收集预测、概率和标签
        preds = (outputs > 0.5).float()
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())
        all_probs.extend(outputs.cpu().detach().numpy())

    avg_loss = total_loss / len(dataloader)
    accuracy = accuracy_score(all_labels, all_preds)
    precision, recall, f1, _ = precision_recall_fscore_support(all_labels, all_preds, average='binary')
    auc = roc_auc_score(all_labels, all_probs)

    return avg_loss, accuracy, precision, recall, f1, auc

# 4. 评估函数
def evaluate(model, dataloader, criterion, device):
    model.eval()
    total_loss = 0
    all_preds = []
    all_labels = []
    all_probs = []

    with torch.no_grad():
        for tokens, labels in tqdm(dataloader, desc="Evaluating", leave=False):
            tokens = tokens.to(device)
            labels = labels.to(device)

            # 前向传播
            outputs = model(tokens)
            # 确保输出和标签具有相同的形状，只移除批次维度以外的单维度
            outputs = outputs.squeeze(dim=1) if outputs.dim() > 1 else outputs
            loss = criterion(outputs, labels)

            total_loss += loss.item()

            # 收集预测、概率和标签
            preds = (outputs > 0.5).float()
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(outputs.cpu().detach().numpy())

    avg_loss = total_loss / len(dataloader)
    accuracy = accuracy_score(all_labels, all_preds)
    precision, recall, f1, _ = precision_recall_fscore_support(all_labels, all_preds, average='binary')
    auc = roc_auc_score(all_labels, all_probs)

    return avg_loss, accuracy, precision, recall, f1, auc

# 5. 主函数
def main():
    # 解析命令行参数
    parser = argparse.ArgumentParser(description='Train ESMC model with five-fold cross validation')
    parser.add_argument('--filename', required=True, help='Filename parameter to replace CHANGE in paths')
    args = parser.parse_args()
    filename = args.filename

    # 设置设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    #device = torch.device('cuda:1' if torch.cuda.is_available() and torch.cuda.device_count() > 1 else 'cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')
    print(f'Using filename parameter: {filename}')

    # 确保日志目录存在
    log_dir = os.path.join(project_root, 'train_log', filename)
    os.makedirs(log_dir, exist_ok=True)
    print(f'Log directory: {log_dir}')

    # 确保数据目录存在
    data_dir = os.path.join(project_root, 'data', filename)
    os.makedirs(data_dir, exist_ok=True)
    print(f'Data directory: {data_dir}')

    # 加载五折数据集
    data_path = os.path.join(data_dir, 'sequence_dataset_train.csv')
    df = load_fold_data(data_path)
    print(f'Total samples in fold dataset: {len(df)}')

    # 加载测试集数据
    test_data_path = os.path.join(data_dir, 'sequence_dataset_test.csv')
    test_sequences, test_labels = load_test_data(test_data_path)
    print(f'Test samples: {len(test_sequences)}')

    # 加载tokenizer
    tokenizer = EsmSequenceTokenizer()

    # 加载本地预训练模型
    print('Loading local pre-trained ESMC model...')
    # 直接构建模型并加载本地权重
    from esm.tokenization import get_esmc_model_tokenizers
    tokenizer = get_esmc_model_tokenizers()
    esmc_model = ESMC(
        d_model=1152,
        n_heads=18,
        n_layers=36,
        tokenizer=tokenizer,
        use_flash_attn=True
    ).to(device)
    # 加载本地模型权重
    model_path = os.path.join(project_root, 'model', 'esmc-600m', 'esmc_600m_2024_12_v0.pth')
    #model_path = '../../../../esm-main/model/esmc-600m/esmc_600m_2024_12_v0.pth'
    print(f'Loading model from: {model_path}')
    state_dict = torch.load(model_path, map_location=device)
    esmc_model.load_state_dict(state_dict)
    esmc_model.eval()

    # 创建分类器模型
    model = ProteinClassifier(esmc_model).to(device)

    # 冻结预训练模型参数，只训练分类器部分
    for param in esmc_model.parameters():
        param.requires_grad = False
    # 解冻分类器参数
    for param in model.classifier.parameters():
        param.requires_grad = True

    # 定义损失函数、优化器和学习率调度器
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.classifier.parameters(), lr=1e-3)

    # 五折交叉验证
    fold_results = []
    num_epochs = 16
    patience = 3  # 早停耐心值

    # 创建日志文件
    log_file = os.path.join(log_dir, 'training_log.txt')
    with open(log_file, 'w') as f:
        f.write('Training started at: {}\n'.format(pd.Timestamp.now()))
        f.write('Filename: {}\n'.format(filename))
        f.write('Device: {}\n'.format(device))
        f.write('Number of epochs: {}\n'.format(num_epochs))
        f.write('Patience: {}\n'.format(patience))
        f.write('\n')

    for fold in range(1, 6):
        print(f'\n--- Fold {fold}/5 ---')
        # 根据当前fold分割训练集和验证集
        train_mask = df[f'dataset_fold_{fold}'] == 'train'
        val_mask = df[f'dataset_fold_{fold}'] == 'val'

        train_sequences = df.loc[train_mask, 'seq'].tolist()
        train_labels = df.loc[train_mask, 'label'].tolist()
        val_sequences = df.loc[val_mask, 'seq'].tolist()
        val_labels = df.loc[val_mask, 'label'].tolist()

        print(f'Train samples: {len(train_sequences)}, Val samples: {len(val_sequences)}')

        # 创建数据集和数据加载器
        train_dataset = ProteinDataset(train_sequences, train_labels, tokenizer)
        val_dataset = ProteinDataset(val_sequences, val_labels, tokenizer)

        train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False)

        # 重置模型权重和优化器
        model.classifier.apply(lambda m: nn.init.kaiming_normal_(m.weight) if isinstance(m, nn.Linear) else None)
        model.classifier.apply(lambda m: nn.init.zeros_(m.bias) if isinstance(m, nn.Linear) else None)
        optimizer = optim.Adam(model.classifier.parameters(), lr=1e-3)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=3)

        best_val_f1 = 0
        early_stop_count = 0

        print(f'Starting training for fold {fold}...')
        for epoch in range(num_epochs):
            print(f'Epoch {epoch+1}/{num_epochs}:')
            train_loss, train_acc, train_prec, train_rec, train_f1, train_auc = train_epoch(
                model, train_loader, criterion, optimizer, device
            )

            val_loss, val_acc, val_prec, val_rec, val_f1, val_auc = evaluate(
                model, val_loader, criterion, device
            )

            # 打印并记录训练和验证指标
            train_metrics = f'  Train: Loss={train_loss:.4f}, Acc={train_acc:.4f}, Prec={train_prec:.4f}, Rec={train_rec:.4f}, F1={train_f1:.4f}, AUC={train_auc:.4f}'
            val_metrics = f'  Val: Loss={val_loss:.4f}, Acc={val_acc:.4f}, Prec={val_prec:.4f}, Rec={val_rec:.4f}, F1={val_f1:.4f}, AUC={val_auc:.4f}'
            print(train_metrics)
            print(val_metrics)

            # 写入日志文件
            with open(log_file, 'a') as f:
                f.write(f'Fold {fold}, Epoch {epoch+1}/{num_epochs}:\n')
                f.write(train_metrics + '\n')
                f.write(val_metrics + '\n')

            # 更新学习率调度器
            scheduler.step(val_f1)

            # 保存最佳模型（基于验证集F1分数）
            if val_f1 > best_val_f1:
                best_val_f1 = val_f1
                # 确保模型保存目录存在
                save_dir = os.path.join(project_root, 'save_model', filename)
                os.makedirs(save_dir, exist_ok=True)
                fold_model_path = os.path.join(save_dir, f'best_model_fold_{num_epochs}epoch_{fold}.pth')
                torch.save(model.state_dict(), fold_model_path)
                save_msg = f'  Saved best model for fold {fold} with F1: {best_val_f1:.4f} to {fold_model_path}'
                print(save_msg)
                # 写入日志文件
                with open(log_file, 'a') as f:
                    f.write(save_msg + '\n')
                early_stop_count = 0  # 重置早停计数
            else:
                early_stop_count += 1
                no_improve_msg = f'  No improvement in validation F1 for {early_stop_count} epoch(s)'
                print(no_improve_msg)
                # 写入日志文件
                with open(log_file, 'a') as f:
                    f.write(no_improve_msg + '\n')

            # 早停检查
            if early_stop_count >= patience:
                early_stop_msg = f'Early stopping after {epoch+1} epochs due to no improvement in validation F1 for {patience} consecutive epochs'
                print(early_stop_msg)
                # 写入日志文件
                with open(log_file, 'a') as f:
                    f.write(early_stop_msg + '\n')
                break

        # 记录当前fold的最佳验证AUC
        fold_results.append(best_val_f1)
        fold_best_msg = f'Fold {fold} best validation F1: {best_val_f1:.4f}'
        print(fold_best_msg)
        # 写入日志文件
        with open(log_file, 'a') as f:
            f.write(fold_best_msg + '\n\n')

    # 计算平均验证F1
    avg_val_f1 = np.mean(fold_results)
    avg_f1_msg = f'\nAverage validation F1 across all folds: {avg_val_f1:.4f}'
    print(avg_f1_msg)
    # 写入日志文件
    with open(log_file, 'a') as f:
        f.write(avg_f1_msg + '\n\n')

    # 从日志文件中读取每个fold的最佳验证F1分数，找到效果最好的一折
    print('\nReading log file to find the best performing fold...')
    best_fold = 1
    best_fold_f1 = 0
    fold_f1_scores = {}
    
    # 读取日志文件
    with open(log_file, 'r') as f:
        log_content = f.readlines()
        
    # 查找每个fold的最佳F1分数
    for line in log_content:
        if 'fold' in line and 'Saved best model' in line:
            try:
                fold_num = int(line.split('fold')[1].split(' ')[1])
                f1_score = float(line.split('F1:')[1].split()[0].strip())
                fold_f1_scores[fold_num] = f1_score
                if f1_score > best_fold_f1:
                    best_fold_f1 = f1_score
                    best_fold = fold_num
            except:
                continue
    
    print(f'Best performing fold: {best_fold} with validation F1 score: {best_fold_f1:.4f}')
    print(f'All fold F1 scores: {fold_f1_scores}')
    
    # 加载效果最好的一折模型对测试集进行测试
    print(f'\nLoading best model from fold {best_fold} for evaluation on test set...')
    test_dataset = ProteinDataset(test_sequences, test_labels, tokenizer)
    test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)

    # 准备测试
    all_probs = []
    all_labels = []
    
    # 加载最佳模型
    save_dir = os.path.join(project_root, 'save_model', filename)
    os.makedirs(save_dir, exist_ok=True)
    best_model_path = os.path.join(save_dir, f'best_model_fold_{num_epochs}epoch_{best_fold}.pth')
    model.load_state_dict(torch.load(best_model_path, map_location=device))
    model.eval()
    
    with torch.no_grad():
        for tokens, labels in tqdm(test_loader, desc="Testing"):
            tokens = tokens.to(device)
            labels = labels.to(device)
            
            # 使用最佳模型进行预测
            outputs = model(tokens)
            # 确保输出形状正确
            outputs = outputs.squeeze(dim=1) if outputs.dim() > 1 else outputs
            all_probs.extend(outputs.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    # 计算评估指标
    accuracy = accuracy_score(all_labels, (np.array(all_probs) > 0.5).astype(float))
    precision, recall, f1, _ = precision_recall_fscore_support(all_labels, (np.array(all_probs) > 0.5).astype(float), average='binary')
    auc = roc_auc_score(all_labels, all_probs)

    # 打印并记录测试结果
    test_results_msg = f'Final Test Results with Best Fold ({best_fold}): Acc={accuracy:.4f}, Prec={precision:.4f}, Rec={recall:.4f}, F1={f1:.4f}, AUC={auc:.4f}'
    print(test_results_msg)
    # 写入日志文件
    with open(log_file, 'a') as f:
        f.write(f'Best performing fold: {best_fold} with validation F1 score: {best_fold_f1:.4f}\n')
        f.write(f'All fold F1 scores: {fold_f1_scores}\n')
        f.write(test_results_msg + '\n')
        f.write('Training completed at: {}\n'.format(pd.Timestamp.now()))

    print('Training completed!')

if __name__ == '__main__':
    main()
