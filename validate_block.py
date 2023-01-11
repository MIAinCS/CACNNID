import os
from tqdm import tqdm
import pandas as pd

def validate(subset_path, filter_mode, test_subset):
    error_list = [each.strip() for each in open(f'./error_file_{filter_mode}.txt')]

    train_quant = {}
    block_list = os.listdir(subset_path)

    for block_index in tqdm(block_list):
        train_file_path = os.path.join(subset_path, block_index, 'train.csv')
        df = pd.read_csv(train_file_path)
        data_path = list(df.data_path)

        count = 0
        
        for error_file in error_list:
            if error_file in data_path:
                count += 1
        
        train_quant[block_index] = count
    
    with open(f'./error_count/{test_subset}_{filter_mode}.txt','w') as fw:
        for key,value in train_quant.items():
            fw.write(f'{key}\t{value}\n')
    return

test_list = [f'subset{i}' for i in range(10)]
mode_list = ['prune_by_class', 'prune_by_noise_rate', 'both', 'confident_learning', 'predicted_neq_given']
for test_subset in test_list:
    for mode in mode_list:
        print(f'start {mode}!')
        validate(f'./data/LUNA16_1v4_block/{test_subset}', mode,test_subset)