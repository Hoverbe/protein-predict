# ESMC-Protein-Predict

## 项目介绍
ESMC-Protein-Predict是一个基于进化尺度模型(ESM)的蛋白质预测工具，使用深度学习方法对蛋白质序列进行分析和预测。该项目包含模型训练、预测功能以及一个友好的Web界面，方便用户进行蛋白质特性预测。

## 项目结构
```
├── .idea/              # IDE配置文件
├── Front/              # 前端Web应用
│   ├── app.py          # Flask应用入口
│   ├── home-Chinese.html # 中文主页
│   ├── home.html       # 英文主页
│   └── predict_front.py # 预测前端逻辑
├── data/               # 数据集
│   └── five-folds-data/ # 五折交叉验证数据
├── model/              # 模型代码
│   ├── predict.py      # 预测模块
│   └── train-fivefold.py # 五折交叉验证训练
├── requirements.txt    # 项目依赖
├── save_model/         # 保存的模型
│   ├── best-model/     # 最佳模型
│   └── esmc-600m/      # ESMC-600M模型
└── tools/              # 工具脚本
```

## 安装指南
1. 克隆项目到本地
```bash
git clone https://github.com/yourusername/esmc-protein-predict.git
cd esmc-protein-predict
```

2. 创建并激活虚拟环境
```bash
# 使用conda
conda create -n esmc-protein python=3.10
conda activate esmc-protein

# 或使用venv
python -m venv venv
# Windows
env\Scripts\activate
# Linux/Mac
source venv/bin/activate
```

3. 安装依赖
```bash
pip install -r requirements.txt
```

4. 下载预训练模型
请确保`save_model/`目录下包含必要的预训练模型文件。

## 使用方法

### 1. 模型训练
运行五折交叉验证训练脚本：
```bash
cd model
python train-fivefold.py
```
训练过程中，模型会自动保存在`save_model/best-model/`目录下。

### 2. 启动Web服务
```bash
cd Front
python app.py
```
服务启动后，在浏览器中访问 `http://localhost:5000` 即可使用Web界面进行预测。

### 3. 直接使用预测功能
可以在Python脚本中导入`predict.py`模块使用预测功能：
```python
from model.predict import load_model, predict

# 加载模型
model, tokenizer = load_model()

# 进行预测
sequence = "MTEITAAMVKELRESTGAGMMDCKNALSETNGDFDKAVQLLREKGLGKAAKKADRLAAEG"
result = predict(model, tokenizer, sequence)
print(result)
```

## 依赖项
项目主要依赖以下库：
- pandas==2.2.2
- numpy==1.26.4
- matplotlib==3.8.4
- seaborn==0.13.2
- torch==2.3.1
- scikit-learn==1.4.2
- tqdm==4.66.4
- biopython==1.83
- Flask==2.3.3
- Flask-CORS==4.0.0

完整依赖列表见 `requirements.txt` 文件。

## 许可证
本项目采用 [MIT License](https://opensource.org/licenses/MIT) 开源协议。

## 联系方式
如有问题，请联系项目维护者：[esmc-protein-predict@example.com](mailto:esmc-protein-predict@example.com)

## 许可证
[MIT License](LICENSE)

## 联系方式
如有问题，请联系 [your_email@example.com].