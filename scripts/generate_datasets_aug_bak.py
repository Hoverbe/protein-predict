import os
import random
import pandas as pd
import networkx as nx
from copy import deepcopy

# 导入工具函数
from utils import ReadFastaFile, SaveFastaFile
from process_neg_raw_data import process_test
from process_sequence_data import process_add_seq

# 配置常量
NUM_FOLDS = 5
TEST_RATIO = 0.1
NUM_BINS = 18  # 序列长度分组数量
SIMILARITY_THRESHOLD = 30  # 序列相似度阈值 (%)
NEG_TO_POS_RATIO = 5  # 负样本与正样本的比例


def get_num_neg_seqs_per_bin(seqs):
    """计算每个长度区间需要的负样本数量，基于对应区间的正样本数量"""
    lengths = [len(s) for s in seqs]
    # 初始化每个区间的计数器
    num_neg_seqs_per_bin = [0] * NUM_BINS
    
    # 统计每个长度区间的正样本数量
    for l in lengths:
        bin_idx = min(l // 50 - 3, NUM_BINS - 1)  # 长度映射到区间索引
        num_neg_seqs_per_bin[bin_idx] += 1
    
    # 确保每个区间至少有1个样本，并按比例放大
    num_neg_seqs_per_bin = [max(count, 1) * NEG_TO_POS_RATIO for count in num_neg_seqs_per_bin]
    return num_neg_seqs_per_bin


def flatten(nested_list):
    """将嵌套列表展平为一维列表"""
    return [item for sublist in nested_list for item in sublist]


def divide_k_fold(items, k):
    """
    将列表分成k折，用于交叉验证
    
    Args:
        items: 待划分的项目列表
        k: 折数
    
    Returns:
        包含k个子列表的列表
        例如: input [1, 2, 3, 4, 5, 6, 7, 8], k=3
             output [[1, 4, 7], [2, 5, 8], [3, 6]]
    """
    folds = [[] for _ in range(k)]
    for i, item in enumerate(items):
        folds[i % k].append(item)
    return folds


def select_test_clusters(components, target_size):
    """从聚类结果中选择测试集样本"""
    # 深拷贝组件列表以避免修改原始数据
    shuffled_components = deepcopy(components)
    random.shuffle(shuffled_components)
    
    test_headers = []
    test_cluster_indices = []
    
    # 尝试多次直到找到合适的组合
    max_attempts = 100  # 避免无限循环
    for _ in range(max_attempts):
        test_headers.clear()
        test_cluster_indices.clear()
        
        # 尝试添加聚类直到达到目标大小
        for i, cluster in enumerate(shuffled_components):
            test_headers.extend(cluster)
            test_cluster_indices.append(i)
            
            if len(test_headers) >= target_size:
                # 如果超过目标大小不多，可以截断
                if len(test_headers) - target_size < len(cluster) / 2:
                    excess = len(test_headers) - target_size
                    test_headers = test_headers[:-excess]
                return test_headers, test_cluster_indices
        
        # 如果一次尝试没有成功，重新洗牌
        random.shuffle(shuffled_components)
    
    # 如果达到最大尝试次数仍未找到完全匹配的组合，返回最近似的结果
    return test_headers, test_cluster_indices


def create_sequence_groups(seqs, seq_to_header=None):
    """根据序列长度创建分组"""
    # 创建分组列表
    groups = [[] for _ in range(NUM_BINS)]
    
    # 将序列分配到对应的长度区间
    for seq in seqs:
        bin_idx = min(len(seq) // 50 - 3, NUM_BINS - 1)  # 长度映射到区间索引
        groups[bin_idx].append(seq)
    
    # 如果提供了序列到头部的映射，则转换为头部分组
    if seq_to_header:
        groups = [[seq_to_header[seq] for seq in group] for group in groups]
    
    return groups


def main():
    """主函数，生成数据集"""
    # 获取项目根目录
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    
    # 读取正样本和负样本FASTA文件
    print("读取正样本FASTA文件...")
    pos_headers, pos_seqs = ReadFastaFile(os.path.join(project_root, 'data', 'positive_seqs_v3_substrate_pocket_sim_aug_v3_unique.fasta'))
    pos_header_to_seq = {h: s for h, s in zip(pos_headers, pos_seqs)}
    pos_seq_to_header = {s: h for h, s in zip(pos_headers, pos_seqs)}
    
    print("读取负样本FASTA文件...")
    neg_headers, neg_seqs = ReadFastaFile(os.path.join(project_root, 'data', 'negative_seqs_v3_substrate_pocket_sim_aug_v3_unique.fasta'))
    neg_seq_to_header = {s: h for h, s in zip(neg_headers, neg_seqs)}
    neg_header_to_seq = {h: s for h, s in zip(neg_headers, neg_seqs)}
    
    # 读取BLAST结果并构建序列相似性图
    print("构建序列相似性图...")
    pos_blast = pd.read_csv(
        os.path.join(project_root, 'data', 'pos_seqs_v3_sub_pok_sim_aug_v3_uniq_self_blast.tsv'),
        sep='\t', comment='#',
        names=['query', 'target', 'fident', 'alnlen', 'mismatch', 'gapopen', 'qstart', 'qend', 'tstart', 'tend', 'e_value', 'bits']
    )
    
    # 根据相似度构建图
    seq_graph = nx.Graph()
    # 添加满足相似度阈值的边（排除自环）
    edges = [(t.query, t.target) for _, t in pos_blast.iterrows() if t.fident >= SIMILARITY_THRESHOLD and t.query != t.target]
    seq_graph.add_edges_from(edges)
    
    # 确保所有正样本都在图中
    for header in pos_headers:
        if header not in seq_graph:
            seq_graph.add_node(header)
    
    # 获取连通组件并按大小排序
    components = [list(comp) for comp in nx.connected_components(seq_graph)]
    components.sort(key=len, reverse=True)
    
    # 选择测试集正样本
    num_test_pos_headers = int(round(len(pos_headers) * TEST_RATIO))
    print(f"测试集正样本数：{num_test_pos_headers}")
    
    test_pos_headers, test_pos_cluster_indices = select_test_clusters(components, num_test_pos_headers)
    print(f"选择的测试集簇数量：{len(test_pos_cluster_indices)}")
    print(f"实际测试集正样本数：{len(test_pos_headers)}")
    
    # 选择训练验证集的正负样本
    # 确定非测试集的簇
    no_test_cluster_indices = list(set(range(len(components))) - set(test_pos_cluster_indices))
    
    # 获取非测试集的正样本
    no_test_pos_headers = []
    for idx in no_test_cluster_indices:
        no_test_pos_headers.extend(components[idx])
    
    no_test_pos_seqs = [pos_header_to_seq[h] for h in no_test_pos_headers]
    
    # 根据序列长度分组正样本
    no_test_pos_seq_groups = create_sequence_groups(no_test_pos_seqs)
    no_test_pos_header_groups = create_sequence_groups(no_test_pos_seqs, pos_seq_to_header)
    
    # 计算每个长度区间需要的负样本数量
    num_neg_seqs_per_bin = get_num_neg_seqs_per_bin(no_test_pos_seqs)
    
    # 根据序列长度分组负样本
    neg_seq_groups = create_sequence_groups(neg_seqs)
    no_test_neg_seq_groups = []
    
    for i, (neg_group, target_count) in enumerate(zip(neg_seq_groups, num_neg_seqs_per_bin)):
        if target_count < len(neg_group):
            # 如果负样本数量超过需求，则随机选择
            no_test_neg_seq_groups.append(random.sample(neg_group, target_count))
        else:
            # 否则使用所有负样本
            no_test_neg_seq_groups.append(neg_group)
            print(f'分组 {i + 1}: {len(neg_group)} 个负样本少于需求的 {target_count} 个，全部采用。')
    
    # 转换为负样本头部分组
    no_test_neg_header_groups = create_sequence_groups(flatten(no_test_neg_seq_groups), neg_seq_to_header)
    
    # 测试集的负样本是剩余的负样本
    all_train_val_neg_headers = set(flatten(no_test_neg_header_groups))
    test_neg_headers = list(set(neg_headers) - all_train_val_neg_headers)
    
    # 打印数据集统计信息
    print(f"训练验证集正样本数：{len(set(flatten(no_test_pos_header_groups)))}")
    print(f"训练验证集负样本数：{len(all_train_val_neg_headers)}")
    print(f"测试集负样本数：{len(test_neg_headers)}")
    
    # 打乱分组内的顺序以增加随机性
    for group in no_test_pos_header_groups:
        random.shuffle(group)
    for group in no_test_neg_header_groups:
        random.shuffle(group)
    
    # 将训练验证集的正负样本分成5折
    no_test_neg_header_groups_k_fold = [divide_k_fold(group, NUM_FOLDS) for group in no_test_neg_header_groups]
    no_test_pos_header_groups_k_fold = [divide_k_fold(group, NUM_FOLDS) for group in no_test_pos_header_groups]
    
    # 展平所有折的头部列表
    no_test_neg_headers_flat = flatten([flatten(folds) for folds in no_test_neg_header_groups_k_fold])
    no_test_pos_headers_flat = flatten([flatten(folds) for folds in no_test_pos_header_groups_k_fold])
    
    # 为每个折生成数据集标签（train/val）
    datasets = []
    for fold_idx in range(NUM_FOLDS):
        # 初始化每个长度分组的标签列表
        fold_pos_labels = [[] for _ in range(len(no_test_pos_header_groups_k_fold))]
        fold_neg_labels = [[] for _ in range(len(no_test_neg_header_groups_k_fold))]
        
        # 为每个分组的每个折分配标签
        for group_idx in range(len(no_test_pos_header_groups_k_fold)):
            for f_idx in range(NUM_FOLDS):
                # 当前折为验证集，其他为训练集
                label = 'val' if f_idx == fold_idx else 'train'
                
                # 为该折的所有样本分配相同的标签
                num_samples = len(no_test_pos_header_groups_k_fold[group_idx][f_idx])
                fold_pos_labels[group_idx].append([label] * num_samples)
                
                # 同样处理负样本
                if group_idx < len(no_test_neg_header_groups_k_fold):
                    num_samples_neg = len(no_test_neg_header_groups_k_fold[group_idx][f_idx])
                    fold_neg_labels[group_idx].append([label] * num_samples_neg)
        
        # 展平所有标签列表并合并正负样本标签
        flat_pos_labels = flatten([flatten(labels) for labels in fold_pos_labels])
        flat_neg_labels = flatten([flatten(labels) for labels in fold_neg_labels])
        datasets.append(flat_pos_labels + flat_neg_labels)
    
    # 构建完整的数据集DataFrame
    num_test_seqs = len(test_pos_headers) + len(test_neg_headers)
    data = {
        'header': no_test_pos_headers_flat + no_test_neg_headers_flat + test_pos_headers + test_neg_headers,
        'label': [1] * len(no_test_pos_headers_flat) + [0] * len(no_test_neg_headers_flat) + \
                 [1] * len(test_pos_headers) + [0] * len(test_neg_headers),
        'dataset_fold_1': datasets[0] + ['test'] * num_test_seqs,
        'dataset_fold_2': datasets[1] + ['test'] * num_test_seqs,
        'dataset_fold_3': datasets[2] + ['test'] * num_test_seqs,
        'dataset_fold_4': datasets[3] + ['test'] * num_test_seqs,
        'dataset_fold_5': datasets[4] + ['test'] * num_test_seqs,
    }
    
    df = pd.DataFrame(data)
    
    # 保存数据集
    output_path = os.path.join(project_root, 'data', 'sequence_dataset_train.csv')
    df.to_csv(output_path, index=False)
    print(f"数据集已保存到：{output_path}")
    
    # 提取并保存测试集和非测试集的负样本FASTA文件
    test_neg_df = df.query('label == 0 and dataset_fold_1 == "test"')
    test_neg_headers = test_neg_df['header'].tolist()
    test_neg_seqs = [neg_header_to_seq[h] for h in test_neg_headers]
    
    non_test_neg_df = df.query('label == 0 and dataset_fold_1 != "test"')
    non_test_neg_headers = non_test_neg_df['header'].tolist()
    non_test_neg_seqs = [neg_header_to_seq[h] for h in non_test_neg_headers]
    
    SaveFastaFile(os.path.join(project_root, 'data', 'test_set_neg_all.fasta'), test_neg_headers, test_neg_seqs)
    SaveFastaFile(os.path.join(project_root, 'data', 'non_test_set_neg_all.fasta'), non_test_neg_headers, non_test_neg_seqs)
    
    # 调用其他处理函数
    print("处理测试数据集...")
    process_test()
    
    print("处理序列数据...")
    process_add_seq('test.csv')
    process_add_seq('processed_sequence_dataset_v3_substrate_pocket_aug.csv')
    
    print("数据集生成完成！")


if __name__ == "__main__":
    main()

