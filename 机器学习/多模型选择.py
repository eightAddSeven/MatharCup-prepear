import pandas as pd
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
import numpy as np

# 载入数据
data = pd.read_csv(r"filepath")
X = data.drop(columns=['target'])
y = data['target']

# 定义多个模型
models={
    RandomForestClassifier(n_jobs=-1): "Random Forest",
    GradientBoostingClassifier(): "Gradient Boosting",  
    AdaBoostClassifier(): "AdaBoost",
    SVC(): "Support Vector Machine",
    LogisticRegression(max_iter=1000): "Logistic Regression",
    KNeighborsClassifier(): "K-Nearest Neighbors",
    XGBClassifier(use_label_encoder=False, eval_metric='logloss'): "XGBoost",
    LGBMClassifier(): "LightGBM"
}

# 评估每个模型的性能
results = {}
for name, model in models.items():
    try:
        # 使用交叉验证快速评估
        scores = cross_val_score(model, X, y, cv=5, scoring='accuracy', n_jobs=-1)
        results.append({
            'Model': name,
            'Mean_Accuracy': scores.mean(),
            'Std_Accuracy': scores.std(),
            'Best_Score': scores.max()
        })
        print(f"{name}: {scores.mean():.4f} (+/- {scores.std():.4f})")
    except Exception as e:
        print(f"{name} failed: {e}")