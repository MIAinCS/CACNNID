import os
import numpy as np


BLOCK_INDEX = 6
BLOCK_SUM = int(BLOCK_INDEX*BLOCK_INDEX)
BLOCK_STEP = int(48/BLOCK_INDEX)
BLOCK_NUM = int(BLOCK_STEP*BLOCK_STEP)

# 这里将avg设为全局变量
max_avg = np.zeros((BLOCK_SUM))

# 计算1类的最值
def calculate_class1_avg(data):
    block_i = 0
    for i in range(BLOCK_INDEX):
        for j in range(BLOCK_INDEX):
            sum = 0
            for m in range(BLOCK_STEP):
                for n in range(BLOCK_STEP):
                    sum += data[i * BLOCK_STEP + m][j * BLOCK_STEP + n]
            curr_avg = sum/BLOCK_NUM
            if curr_avg > max_avg[block_i]:
                max_avg[block_i] = sum/BLOCK_NUM
            block_i += 1

def get_class1_max(root_dir):
    subdir = os.listdir(root_dir)
    img_set = []
    for i in range(10):
        curr_path = os.path.join(root_dir, subdir[i])
        npy_set = os.listdir(curr_path)
        for each_npy in npy_set:
            npy_path = os.path.join(curr_path, each_npy)
            data = np.load(npy_path)
            calculate_class1_avg(data[24])     # 返回的是一个9*1的数组
        print(str(subdir[i]) + " done!")

# 计算0类的最值
def extract_class0(root_dir, result_list):
    subdir = os.listdir(root_dir)
    for i in range(10):
        curr_path = os.path.join(root_dir, subdir[i])
        npy_set = os.listdir(curr_path)
        for each_npy in npy_set:
            npy_path = os.path.join(curr_path, each_npy)
            data = np.load(npy_path)
            is_saved = calculate_class0_avg(data[24])     # True -- 保存路径并筛除 False -- 不保存路径，不用筛除
            if is_saved:
                result_list.append(npy_path+'\n')
        print(str(subdir[i]) + " done!")

def calculate_class0_avg(data):
    avg = np.zeros((BLOCK_SUM))
    block_i = 0
    for i in range(BLOCK_INDEX):
        for j in range(BLOCK_INDEX):
            sum = 0
            for m in range(BLOCK_STEP):
                for n in range(BLOCK_STEP):
                    sum += data[i * BLOCK_STEP + m][j * BLOCK_STEP + n]
            curr_avg = sum/BLOCK_NUM
            if(curr_avg > max_avg[block_i]):
                return True
            else:
                block_i += 1
    return False

if __name__ == '__main__':
    result_list = []

    class_1 = "./preprocessed_data0/1"
    class_0 = "./preprocessed_data0/0"

    get_class1_max(class_1)

    print(max_avg)

    extract_class0(class_0, result_list)

    with open('extract_list.txt','w') as f:  
        f.writelines('bad data num = {}.\n'.format(len(result_list)))  #设置文件对象
        for item in result_list:
            f.writelines(item)                #将字符串写入文件中