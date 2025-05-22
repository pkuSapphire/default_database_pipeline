# coding:utf-8

import os
import pickle
import pandas as pd

# 设置数据目录
DATA_DIR = "data"
PICKLE_FILE = os.path.join(DATA_DIR, "base_dataset.pkl")
CSV_FILE = os.path.join(DATA_DIR, "base_dataset.csv")

# 读取 pickle 数据
with open(PICKLE_FILE, "rb") as f:
    df = pickle.load(f)

# 保存为 CSV 文件
df.to_csv(CSV_FILE, index=False)
print(f"CSV file saved to: {CSV_FILE}")
