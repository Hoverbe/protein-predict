import os
import pandas as pd
import argparse
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, roc_auc_score
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import numpy as np

# 设置项目根目录
project_root = os.path.dirname(__file__)

# 导入必要的模型和工具
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

# 3. 评估函数
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
            outputs = model(tokens).squeeze()
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

# 4. 主函数
def main():
    # 解析命令行参数
    parser = argparse.ArgumentParser(description='Test ESMC model with best fold from five-fold cross validation')
    parser.add_argument('--filename', required=True, help='Filename parameter to replace CHANGE in paths')
    args = parser.parse_args()
    filename = args.filename

    # 设置设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')
    print(f'Using filename parameter: {filename}')

    # 检查日志目录是否存在
    log_dir = os.path.join(project_root, 'train_log', filename)
    log_file = os.path.join(log_dir, 'training_log.txt')
    if not os.path.exists(log_file):
        print(f'Error: Log file not found at {log_file}')
        return
    print(f'Log file found: {log_file}')

    # 从日志文件中读取每个fold的最佳验证F1分数，找到效果最好的一折
    print('\nReading log file to find the best performing fold...')
    best_fold = 1
    best_fold_f1 = 0
    fold_f1_scores = {}
    num_epochs = 16  # 默认值，实际值会从日志中读取
    
    # 读取日志文件
    with open(log_file, 'r') as f:
        log_content = f.readlines()
        
    # 查找每个fold的最佳F1分数和epoch数
    for line in log_content:
        if 'Number of epochs:' in line:
            try:
                num_epochs = int(line.split(':')[1].strip())
            except:
                pass
        elif 'fold' in line and 'Saved best model' in line:
            try:
                fold_num = int(line.split('fold')[1].split(' ')[1])
                # f1_score = float(line.split('F1:')[1].split()[0].strip())
                f1_score = float(line.split('AUC:')[1].split()[0].strip())
                fold_f1_scores[fold_num] = f1_score
                if f1_score > best_fold_f1:
                    best_fold_f1 = f1_score
                    best_fold = fold_num
            except:
                continue
    
    if not fold_f1_scores:
        print('Error: Could not find fold information in log file')
        return
        
    print(f'Best performing fold: {best_fold} with validation F1 score: {best_fold_f1:.4f}')
    print(f'All fold F1 scores: {fold_f1_scores}')
    print(f'Detected number of epochs: {num_epochs}')
    
    # 确保数据目录存在并加载测试数据
    data_dir = os.path.join(project_root, 'data', filename)
    test_data_path = os.path.join(data_dir, 'sequence_dataset_test.csv')
    if not os.path.exists(test_data_path):
        print(f'Error: Test data file not found at {test_data_path}')
        return
        
    test_sequences, test_labels = load_test_data(test_data_path)
    print(f'Test samples: {len(test_sequences)}')

    # 加载tokenizer和模型
    print('Loading tokenizer and model...')
    from esm.tokenization import get_esmc_model_tokenizers
    tokenizer = get_esmc_model_tokenizers()
    
    # 构建ESMC模型
    esmc_model = ESMC(
        d_model=1152,
        n_heads=18,
        n_layers=36,
        tokenizer=tokenizer,
        use_flash_attn=True
    ).to(device)
    
    # 加载本地模型权重
    model_path = os.path.join(project_root, 'model', 'esmc-600m', 'esmc_600m_2024_12_v0.pth')
    if not os.path.exists(model_path):
        print(f'Error: Pre-trained model not found at {model_path}')
        return
        
    print(f'Loading pre-trained model from: {model_path}')
    state_dict = torch.load(model_path, map_location=device)
    esmc_model.load_state_dict(state_dict)
    esmc_model.eval()

    # 创建分类器模型
    model = ProteinClassifier(esmc_model).to(device)

    # 冻结预训练模型参数
    for param in esmc_model.parameters():
        param.requires_grad = False

    # 准备测试数据加载器
    test_dataset = ProteinDataset(test_sequences, test_labels, tokenizer)
    test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)
    criterion = nn.BCELoss()

    # 加载最佳模型
    save_dir = os.path.join(project_root, 'save_model', filename)
    best_model_path = os.path.join(save_dir, f'best_model_fold_{num_epochs}epoch_{best_fold}.pth')
    
    if not os.path.exists(best_model_path):
        # 尝试不同的epoch格式（比如ephoc拼写错误的情况）
        alternative_paths = [
            os.path.join(save_dir, f'best_model_fold_{num_epochs}ephoc_{best_fold}.pth'),
            os.path.join(save_dir, f'best_model_fold_{best_fold}.pth')
        ]
        
        found = False
        for path in alternative_paths:
            if os.path.exists(path):
                best_model_path = path
                found = True
                break
        
        if not found:
            print(f'Error: Best model file not found at {best_model_path} or any alternative paths')
            return
    
    print(f'Loading best model from: {best_model_path}')
    model.load_state_dict(torch.load(best_model_path, map_location=device))
    model.eval()
    
    # 在测试集上评估模型
    print('\nEvaluating best model on test set...')
    test_loss, test_acc, test_prec, test_rec, test_f1, test_auc = evaluate(
        model, test_loader, criterion, device
    )

    # 打印测试结果
    test_results_msg = f'Test Results with Best Fold ({best_fold}): Acc={test_acc:.4f}, Prec={test_prec:.4f}, Rec={test_rec:.4f}, F1={test_f1:.4f}, AUC={test_auc:.4f}'
    print(test_results_msg)
    
    # 将测试结果追加到日志文件
    with open(log_file, 'a') as f:
        f.write('\n--- Retest Results ---')
        f.write(f'Retest timestamp: {pd.Timestamp.now()}\n')
        f.write(f'Retest using best fold: {best_fold} with validation F1 score: {best_fold_f1:.4f}\n')
        f.write(test_results_msg + '\n')

    print('Retest completed!')

if __name__ == '__main__':
    main()