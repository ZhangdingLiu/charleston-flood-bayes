import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd

# 定义混淆矩阵的数值
conf_matrix = np.array([[32, 9],   # [TN, FP]
                        [6, 36]])  # [FN, TP]

# 定义标签
labels = ['Actual No Flood', 'Actual Flood']
pred_labels = ['Predicted No Flood', 'Predicted Flood']

# 创建DataFrame用于seaborn可视化
df_cm = pd.DataFrame(conf_matrix, index=labels, columns=pred_labels)

# 绘图
plt.figure(figsize=(6.5, 5.5))
sns.set(font_scale=1.4)  # 控制标签字体大小
ax = sns.heatmap(df_cm, annot=True, fmt='d', cmap='Blues', cbar=True, annot_kws={"size": 24})

# 设置轴标签字体大小
ax.set_xlabel("Predicted Label", fontsize=18, weight='bold')
ax.set_ylabel("Actual Label", fontsize=18, weight='bold')

# 设置坐标轴刻度字体
ax.set_xticklabels(ax.get_xticklabels(), fontsize=15, weight='bold')
ax.set_yticklabels(ax.get_yticklabels(), fontsize=15, weight='bold', rotation=0)

# 设置标题
plt.title("Test Set Confusion Matrix\n(Best Configuration)", fontsize=20, weight='bold')

plt.tight_layout()
plt.show()
