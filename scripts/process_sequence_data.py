import csv
import os


def parse_fasta(fasta_path):
    fasta_dict = {}
    current_header = None
    current_sequence = []
    
    with open(fasta_path, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            
            if line.startswith('>'):
                # 如果是新的header，先保存之前的序列
                if current_header:
                    fasta_dict[current_header] = ''.join(current_sequence)
                # 提取新的header（去掉'>'后使用.split()[0]取得）
                current_header = line[1:].split()[0]
                current_sequence = []
            else:
                # 添加到当前序列
                current_sequence.append(line)
        
        # 保存最后一个序列
        if current_header:
            fasta_dict[current_header] = ''.join(current_sequence)
    
    return fasta_dict

# 主函数
def process_add_seq(name):
    # 定义文件路径
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    path_data = os.path.join(project_root, 'data')
    path_five_folds_data = os.path.join(path_data, 'five-folds-data')
    if name == 'sequence_dataset_test.csv':
        csv_file_path = os.path.join(path_data, 'processed', 'neg_raw_data.csv')
    elif name == 'sequence_dataset_v3_substrate_pocket_aug.csv':
        csv_file_path = os.path.join(path_data, 'sequence_dataset_v3_substrate_pocket_aug.csv')
    positive_fasta_path = os.path.join(path_data, 'positive_seqs_v3_substrate_pocket_sim_aug_v3_unique.fasta')
    negative_fasta_path = os.path.join(path_data, 'negative_seqs_v3_substrate_pocket_sim_aug_v3_unique.fasta')
    # output_csv_path = os.path.join(path_data,'processed','processed_sequence_dataset_v3_substrate_pocket_aug.csv')
    output_csv_path = os.path.join(path_data, 'processed', name)
    # 解析FASTA文件，返回header到序列的映射
    # 解析FASTA文件
    print('Parsing positive FASTA file...')
    positive_fasta_dict = parse_fasta(positive_fasta_path)
    print(f'Parsed {len(positive_fasta_dict)} sequences from positive FASTA.')
    
    print('Parsing negative FASTA file...')
    negative_fasta_dict = parse_fasta(negative_fasta_path)
    print(f'Parsed {len(negative_fasta_dict)} sequences from negative FASTA.')
    
    # 读取CSV文件并处理
    print('Processing CSV file...')
    processed_count = 0
    skipped_count = 0
    
    with open(csv_file_path, 'r') as csv_file, open(output_csv_path, 'w', newline='') as output_csv:
        csv_reader = csv.DictReader(csv_file)
        # 创建新的CSV写入器，添加'seq'列
        fieldnames = csv_reader.fieldnames + ['seq']
        csv_writer = csv.DictWriter(output_csv, fieldnames=fieldnames)
        csv_writer.writeheader()
        
        for row in csv_reader:
            # 检查是否需要跳过（dataset_fold_1为'test'）
            if name == 'sequence_dataset_v3_substrate_pocket_aug.csv':
                if row['dataset_fold_1'] == 'test':
                    skipped_count += 1
                    continue
            header = row['header']
            label = int(row['label'])
            
            # 根据label查找序列
            if label == 1:
                seq = positive_fasta_dict.get(header, '')
            else:
                seq = negative_fasta_dict.get(header, '')
            
            # 添加序列到行
            row['seq'] = seq
            csv_writer.writerow(row)
            processed_count += 1
    
    print(f'Processed {processed_count} rows.')
    print(f'Skipped {skipped_count} rows (dataset_fold_1 is "test").')
    print(f'Output file saved to {output_csv_path}')

if __name__ == '__main__':
    # process_add_seq('test.csv')
    process_add_seq('sequence_dataset_v3_substrate_pocket_aug.csv')