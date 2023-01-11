import torch
from torch.utils.data import Dataset
from PIL import Image
import numpy as np
import os
import pandas as pd

class MyData(Dataset):
    def __init__(self,csv_path,test=False):
        super().__init__()

        # 读取csv
        df = pd.read_csv(csv_path)
        self.im_list = df.data_path
        self.dr = df.label
        

    def __getitem__(self, idx):
        img = np.load(self.im_list[idx])
        img = img.astype(np.float32)
        # img = img[16:32,:,:]

        img = img/255

        # 转换到0,1f
        img = torch.tensor(img)
        
        img = torch.unsqueeze(img,0)
        label = self.dr[idx]
        # return img,label
        return img,label,self.im_list[idx]

    def __len__(self):
        return len(self.im_list)

    def get_cls_num_list(self):
        class0_num = 0
        class1_num = 0
        for idx in range(len(self.im_list)):
            if self.dr[idx] == 0:
                class0_num += 1
            else:
                class1_num += 1
        
        return [class0_num, class1_num]