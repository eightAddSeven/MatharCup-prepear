# ==================有些特征与目标的关系相关性很高，可以只留一个========================
# 例如房价预测中的车库面积和车库可容纳车辆具有较高的相关性，可以只保留一个特征以减少冗余信息。

import pandas as pd;
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_predict
import warnings
warnings.filterwarnings("ignore")

pd.set_option('display.max_rows', None)    # 显示所有行
pd.set_option('display.max_columns', None) # 显示所有列
pd.set_option('display.width', 1000)       # 设置显示宽度
pd.set_option('display.max_colwidth', None) # 不限制列宽

# 设置中文字体支持
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']  # 支持中文的字体
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题

# =========加载数据========
train_location="D:/A数学建模/house-prices-advanced-regression-techniques/train.csv"
train=pd.read_csv(train_location)

# 计算特征间的相关矩阵
corr_matrix = train.corr(numeric_only=True).abs()  # 取绝对值方便比较强度

# 可视化
plt.figure(figsize=(10, 8))
sns.heatmap(corr_matrix, cmap='coolwarm', center=0)
plt.title("特征间相关性热力图")
plt.show()

# 找出高度相关的特征对
high_corr_pairs = []
# 自定义阈值——衡量相关性需要达到的阈值
threshold = 0.8  
for i in range(len(corr_matrix.columns)):
    for j in range(i):
        if corr_matrix.iloc[i, j] > threshold:
            col1 = corr_matrix.columns[i]
            col2 = corr_matrix.columns[j]
            high_corr_pairs.append((col1, col2, corr_matrix.iloc[i, j]))

print("高度相关（可考虑删除其中一个）的特征对：")
for pair in high_corr_pairs:
    print(f"{pair[0]} ↔ {pair[1]}: 相关系数 = {pair[2]:.2f}")
