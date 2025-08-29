import os
import re

def process():
    # 定义输入和输出目录
    input_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data')
    output_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data', 'fasta_processed')

    # 确保输出目录存在
    os.makedirs(output_dir, exist_ok=True)

    # 获取输入目录中的所有FASTA文件
    fasta_files = [f for f in os.listdir(input_dir) if f.endswith('.fasta')]

    for fasta_file in fasta_files:
        input_path = os.path.join(input_dir, fasta_file)
        output_path = os.path.join(output_dir, f'processed_{fasta_file}')

        print(f'处理文件: {fasta_file}')

        with open(input_path, 'r') as infile, open(output_path, 'w') as outfile:
            current_title = None
            current_sequence = []

            for line in infile:
                line = line.strip()
                if not line:
                    continue

                # 遇到新的标题行
                if line.startswith('>'):
                    # 如果有当前序列未处理，先写入之前的序列
                    if current_title:
                        outfile.write(f'{current_title}\n')
                        outfile.write(f'{''.join(current_sequence)}\n')

                    # 设置新的标题
                    current_title = line
                    current_sequence = []
                else:
                    # 添加到当前序列
                    current_sequence.append(line)

            # 处理最后一个序列
            if current_title:
                outfile.write(f'{current_title}\n')
                outfile.write(f'{''.join(current_sequence)}\n')

        print(f'处理完成，输出文件: {output_path}')

    print('所有文件处理完成！')
if __name__ == '__main__':
    process()