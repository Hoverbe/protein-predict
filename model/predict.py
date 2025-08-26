import sys
import os
import torch
from torch import nn

# 将项目根目录添加到Python路径
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
print(f'Project root: {project_root}')
if project_root not in sys.path:
    sys.path.append(project_root)
    print(f'Added {project_root} to sys.path')
else:
    print(f'{project_root} already in sys.path')

# 添加esm模块路径
esm_path = os.path.join(project_root, 'esm')
print(f'ESM path: {esm_path}')

sys.path.append(os.path.join(esm_path, 'models'))
sys.path.append(os.path.join(esm_path, 'tokenization'))
sys.path.append(os.path.join(esm_path, 'utils'))

try:
    from esm.tokenization import get_esmc_model_tokenizers
    from esm.utils.encoding import tokenize_sequence
    from esm.models.esmc import ESMC
    print('Successfully imported ESM modules')
except ImportError as e:
    print(f'Error importing ESM modules: {e}')
    sys.exit(1)

# 将项目根目录添加到Python路径
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.append(project_root)

# 添加esm模块路径
esm_path = os.path.join(project_root, 'esm')
sys.path.append(os.path.join(esm_path, 'models'))
sys.path.append(os.path.join(esm_path, 'tokenization'))
sys.path.append(os.path.join(esm_path, 'utils'))

def load_model():
    # 设置设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')
    
    # 加载tokenizer
    tokenizer = get_esmc_model_tokenizers()
    
    # 初始化ESMC模型
    esmc_model = ESMC(
        d_model=1152,
        n_heads=18,
        n_layers=36,
        tokenizer=tokenizer,
        use_flash_attn=True
    ).to(device)
    
    # 定义分类器模型
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
    
    # 创建分类器
    model = ProteinClassifier(esmc_model).to(device)
    
    # 加载训练好的模型权重
    # model_path = os.path.join(project_root, 'AIModel', 'best_model.pth')
    model_path = os.path.join(project_root, 'model', 'best-model','best_model_11.pth')
    print(f'Loading model from: {model_path}')
    
    try:
        state_dict = torch.load(model_path, map_location=device)
        model.load_state_dict(state_dict)
        model.eval()
        print('Model loaded successfully!')
    except Exception as e:
        print(f'Error loading model: {e}')
        sys.exit(1)
    
    return model, tokenizer, device

def predict_sequence(model, tokenizer, sequence, device, max_length=512):
    # 对序列进行tokenization
    tokens = tokenize_sequence(sequence, tokenizer, add_special_tokens=True)
    
    # 截断或填充到最大长度
    if len(tokens) > max_length:
        tokens = tokens[:max_length]
    elif len(tokens) < max_length:
        tokens = torch.cat([tokens, torch.full((max_length - len(tokens),), tokenizer.pad_token_id)])
    
    # 添加批次维度
    tokens = tokens.unsqueeze(0).to(device)
    
    # 进行预测
    with torch.no_grad():
        output = model(tokens).squeeze()
        probability = output.item()
        prediction = 1 if probability > 0.5 else 0
    
    return prediction, probability

def main():
    # 加载模型
    model, tokenizer, device = load_model()
    
    print('\nProtein Protease Predictor')
    print('-------------------------')
    print('Enter a protein sequence to predict if it is a protease.')
    print('Enter \'q\' to quit.')
    
    while True:
        # 获取用户输入
        sequence = input('\nEnter sequence: ').strip()
        
        if sequence.lower() == 'q':
            print('Exiting program. Goodbye!')
            break
        
        if not sequence:
            print('Please enter a valid sequence.')
            continue
        
        # 检查序列是否只包含有效的氨基酸字符
        valid_chars = set('ACDEFGHIKLMNPQRSTVWY')
        if not all(char in valid_chars for char in sequence.upper()):
            print('Invalid sequence. Please enter only amino acid characters (ACDEFGHIKLMNPQRSTVWY).')
            continue
        
        # 进行预测
        prediction, probability = predict_sequence(model, tokenizer, sequence, device)
        
        # 输出结果
        print(f'Prediction: {"Protease" if prediction == 1 else "Not a protease"}')
        print(f'Probability: {probability:.4f}')

if __name__ == '__main__':
    main()