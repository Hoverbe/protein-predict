import pandas as pd
import os
import random
def process_test():
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    # 读取CSV文件
    data_path = os.path.join(project_root, 'data', 'sequence_dataset_train.csv')
    df = pd.read_csv(data_path)

    # 筛选出dataset_fold_1为'test'的行
    test_df = df[df['dataset_fold_1'] == 'test']

    # 提取label为1的数据
    label_1_data = test_df[test_df['label'] == 1]

    # 提取label为0的数据
    label_0_data = test_df[test_df['label'] == 0]

    # 计算需要提取的label为0的数量（label为1数量的5倍）
    label_1_count = len(label_1_data)
    label_0_count = min(len(label_0_data), 5 * label_1_count)

    # 随机选择label_0_count数量的label为0的数据
    if label_0_count > 0:
        selected_label_0_data = label_0_data.sample(n=label_0_count, random_state=42)
    else:
        selected_label_0_data = pd.DataFrame()

    # 合并label为1和选中的label为0的数据
    result_df = pd.concat([label_1_data, selected_label_0_data])

    # 保存结果到指定文件
    save_path = os.path.join(project_root, 'data', 'processed', 'neg_raw_data.csv')
    result_df.to_csv(save_path, index=False)

    print(f"处理完成！")
    print(f"- 总共筛选出 test 数据: {len(test_df)} 条")
    print(f"- 提取的 label 为 1 的数据: {len(label_1_data)} 条")
    print(f"- 提取的 label 为 0 的数据: {len(selected_label_0_data)} 条")
    print(f"- 结果已保存到: {save_path}")