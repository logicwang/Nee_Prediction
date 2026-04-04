import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# 设置全局绘图风格，适合学术论文
plt.style.use('seaborn-v0_8-muted')
plt.rcParams['font.sans-serif'] = ['SimHei']  # 解决中文显示
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['font.size'] = 12

# ==========================================
# 1. 自动搜索并识别实验文件夹
# ==========================================
result_root = 'result'
# 定义你想对比的模型关键词
target_models = ['Informer', 'Ablation_TCN', 'Baseline_PatchTST', 'LightGBM', 'LSTM', 'TCN_Informer']

summary_data = []
plot_data_list = {}

print("正在扫描实验结果...")

for folder in os.listdir(result_root):
    folder_path = os.path.join(result_root, folder)
    if not os.path.isdir(folder_path): continue

    # 匹配模型类型
    matched_model = None
    for m in target_models:
        if m.lower() in folder.lower():
            matched_model = m
            break

    if matched_model:
        # 提取指标
        metrics_files = [f for f in os.listdir(folder_path) if 'metrics.csv' in f]
        data_files = [f for f in os.listdir(folder_path) if 'data.csv' in f]

        if metrics_files and data_files:
            # 读取指标
            df_m = pd.read_csv(os.path.join(folder_path, metrics_files[0]))
            df_m['Model'] = matched_model
            summary_data.append(df_m)

            # 读取预测数据用于绘图 (取前 200 个点展示细节)
            df_d = pd.read_csv(os.path.join(folder_path, data_files[0]))
            plot_data_list[matched_model] = df_d

# 合并所有指标
df_summary = pd.concat(summary_data).reset_index(drop=True)
# 按照 R2 从高到低排序，让你的模型排在最前面或最后面
df_summary = df_summary.sort_values(by='R2', ascending=False)
print("\n--- 实验指标对比表 ---")
print(df_summary[['Model', 'R2', 'MSE', 'MAE']])

# ==========================================
# 2. 绘制指标对比柱状图 (R2 和 MSE)
# ==========================================
fig, ax1 = plt.subplots(figsize=(10, 6))

x = np.arange(len(df_summary))
width = 0.35

# 画 R2 柱状图 (左轴)
bars1 = ax1.bar(x - width / 2, df_summary['R2'], width, label='R2 Score', color='#3498db')
ax1.set_ylabel('R2 Score (越高越好)')
ax1.set_xticks(x)
ax1.set_xticklabels(df_summary['Model'], rotation=15)
ax1.set_ylim(0, 1.1)

# 画 MSE 柱状图 (右轴)
ax2 = ax1.twinx()
bars2 = ax2.bar(x + width / 2, df_summary['MSE'], width, label='MSE Loss', color='#e74c3c', alpha=0.7)
ax2.set_ylabel('MSE Loss (越低越好)')

plt.title('不同模型在农田碳通量预测任务的性能对比')
ax1.legend(loc='upper left')
ax2.legend(loc='upper right')
plt.tight_layout()
plt.savefig('total_metrics_comparison.png', dpi=300)

# ==========================================
# 3. 绘制预测曲线对比图 (选取一段典型时段)
# ==========================================
plt.figure(figsize=(15, 6))
start_idx, end_idx = 100, 300  # 选取 200 个点展示昼夜节律

# 先画真实值 (所有模型真实值一致，取任意一个即可)
first_model = list(plot_data_list.keys())[0]
plt.plot(plot_data_list[first_model]['真实值'].iloc[start_idx:end_idx].values,
         label='真实观测值 (Real)', color='black', linewidth=2.5, linestyle='--')

# 循环画出各个模型的预测值
colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b']
for i, model_name in enumerate(df_summary['Model']):
    if model_name in plot_data_list:
        data = plot_data_list[model_name]['预测值'].iloc[start_idx:end_idx].values
        plt.plot(data, label=f'{model_name} 预测', alpha=0.8, linewidth=1.5)

plt.title('各模型预测碳通量(NEE)昼夜变化拟合效果对比')
plt.xlabel('时间步 (30min)')
plt.ylabel('NEE ($\mu mol \cdot m^{-2} \cdot s^{-1}$)')
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('prediction_curve_comparison.png', dpi=300)

print("\n[SUCCESS] 对比图表已生成: total_metrics_comparison.png, prediction_curve_comparison.png")
plt.show()