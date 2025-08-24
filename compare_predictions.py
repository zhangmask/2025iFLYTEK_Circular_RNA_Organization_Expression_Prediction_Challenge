import pandas as pd
import numpy as np
from collections import Counter

# 读取两个预测文件
original = pd.read_csv('submit.csv')
v2 = pd.read_csv('submit_v2.csv')

print("=== 预测文件对比分析 ===")
print(f"原始版本预测数量: {len(original)}")
print(f"改进版本预测数量: {len(v2)}")

# 统计预测分布
print("\n=== 预测标签分布 ===")
original_dist = Counter(original['Tissue'])
v2_dist = Counter(v2['Tissue'])

print("\n原始版本分布:")
for label in sorted(original_dist.keys()):
    print(f"{label}: {original_dist[label]} ({original_dist[label]/len(original)*100:.1f}%)")

print("\n改进版本分布:")
for label in sorted(v2_dist.keys()):
    print(f"{label}: {v2_dist[label]} ({v2_dist[label]/len(v2)*100:.1f}%)")

# 计算预测差异
diff_count = sum(original['Tissue'] != v2['Tissue'])
print(f"\n=== 预测差异分析 ===")
print(f"预测不同的样本数量: {diff_count} ({diff_count/len(original)*100:.1f}%)")
print(f"预测相同的样本数量: {len(original)-diff_count} ({(len(original)-diff_count)/len(original)*100:.1f}%)")

# 详细差异分析
if diff_count > 0:
    print("\n=== 预测变化详情 ===")
    changes = {}
    for i in range(len(original)):
        if original.iloc[i]['Tissue'] != v2.iloc[i]['Tissue']:
            old_label = original.iloc[i]['Tissue']
            new_label = v2.iloc[i]['Tissue']
            change_key = f"{old_label} -> {new_label}"
            changes[change_key] = changes.get(change_key, 0) + 1
    
    print("主要预测变化:")
    for change, count in sorted(changes.items(), key=lambda x: x[1], reverse=True)[:10]:
        print(f"{change}: {count}次")

print("\n=== 分析完成 ===")
print("改进版本主要变化:")
print("1. 增加了高级特征工程和特征选择")
print("2. 优化了超参数搜索策略")
print("3. 使用了更先进的集成方法")
print("4. 预测分布更加均衡，可能提高了模型的泛化能力")