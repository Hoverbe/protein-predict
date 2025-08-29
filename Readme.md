# ESMC-Protein-Predict

## 项目概述
ESMC-Protein-Predict 是一个用于蛋白质功能预测的深度学习模型项目，特别专注于基于序列信息的酶功能预测。本项目使用增强的数据处理方法和深度学习模型来提高预测准确性。

## 项目结构
```
├── Front/             # 前端应用代码
├── data/              # 数据存储目录
│   ├── fasta_processed/  # 处理后的FASTA文件
│   ├── processed/        # 处理后的数据集
│   └── raw/              # 原始数据
├── model/             # 模型代码
├── save_model/        # 保存的模型权重
├── scripts/           # 数据处理脚本
└── tools/             # 工具函数
```

## 环境要求
- Python 3.8+ 
- DIAMOND (序列比对工具)
- 所需Python库见 requirements.txt

## 安装指南
1. 克隆仓库
```bash
git clone https://github.com/yourusername/esmc-protein-predict.git
cd esmc-protein-predict
```

2. 安装依赖
```bash
pip install -r requirements.txt
```

3. 安装DIAMOND
   - 下载适合您系统的DIAMOND可执行文件: https://github.com/bbuchfink/diamond/releases
   - 将DIAMOND可执行文件放入 `scripts` 文件夹

## 数据准备与处理

### 数据预处理
1. 将原始FASTA文件放入 `data/raw` 目录
2. 运行数据处理脚本
```bash
cd scripts
python process_fasta.py
```

### 训练数据生成

#### 步骤1: 移除冗余序列
这一步会移除90%序列相似度下的冗余序列。

```bash
# 1.1 创建DIAMOND数据库
./diamond makedb --in ../data/positive_seqs_v3.fasta -d ../data/pos_seqs_v3

# 1.2 运行BLASTP比对
./diamond blastp -q ../data/positive_seqs_v3.fasta -d ../data/pos_seqs_v3 -o ../data/pos_seqs_v3_self_blast.tsv --ultra-sensitive -k 0

# 1.3 筛选唯一序列
python select_unique_seqs.py -f pos_seqs_v3
```
运行后会生成 `positive_seqs_v3_substrate_pocket_sim_aug_v3_unique.fasta` 文件。

#### 步骤2: 生成训练数据集
将以下文件放入 `data` 文件夹:
- `positive_seqs_v3_substrate_pocket_sim_aug_v3_unique.fasta`
- `negative_seqs_v3_substrate_pocket_sim_aug_v3_unique.fasta`
- `pos_seqs_v3_sub_pok_sim_aug_v3_uniq_self_blast.tsv`

运行数据集生成脚本:
```bash
python generate_datasets_aug.py
```

该脚本会:
1. 根据30%的阈值对增强的正样本进行聚类
2. 随机选择10%的聚类放入测试集
3. 构建5折交叉验证数据集
4. 为每个组选择5倍数量的负样本
5. 确保测试集中正样本比例为1:5

## 模型训练
```bash
cd ../model
python train-fivefold.py
```
训练完成后，模型将保存在 `save_model/best-model` 目录下。

## 预测
```bash
python predict.py --sequence_file path/to/sequences.fasta --output_file path/to/output.csv
```

## 前端应用
```bash
cd ../Front
python app.py
```
然后在浏览器中访问 http://localhost:5000

## 注意事项
1. 确保DIAMOND可执行文件具有执行权限
2. 大数据集处理可能需要较长时间和较大内存
3. 训练模型前请确保已正确生成数据集

## 常见问题
Q: DIAMOND命令执行失败怎么办?
A: 检查DIAMOND是否正确安装，是否具有执行权限，以及路径是否正确。

Q: 如何调整训练参数?
A: 可以在 `train-fivefold.py` 文件中修改相关参数。

## 联系方式
如果您有任何问题或建议，请联系 [your.email@example.com].









