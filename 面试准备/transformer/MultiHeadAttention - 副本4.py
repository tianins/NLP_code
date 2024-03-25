import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import numpy as np

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei']  # 指定使用黑体字，你也可以选择其他支持中文的字体
plt.rcParams['axes.unicode_minus'] = False  # 用于正常显示负号



# # 图4.1
# # 数据4.3
# categories_3 = ['0', '1-2', '3-5', '6-10', '11-15', '16-20', '21-25', '26-30', '31-35',
#               '36-40', '41-50', '51-60', '61-70', '70-80', '81-90', '91-100', '101-150',
#               '151-200', '201-300', '301-400', '400以上']
# counts_3 = [1243, 874, 743, 839, 556, 444, 299, 267, 194, 176, 287, 212, 193, 188, 117,
#           74, 347, 113, 138, 57, 250]

# # 计算每一项的占比
# total_counts_3 = sum(counts_3)
# percentages_3 = [count / total_counts_3 * 100 for count in counts_3]

# # 4.2
# categories_2 = ['0.00-0.05', '0.05-0.10', '0.10-0.15', '0.15-0.20', '0.20-0.25',
#               '0.25-0.30', '0.30-0.35', '0.35-0.40', '0.40-0.45', '0.45-0.50',
#               '0.50-0.55', '0.55-0.60', '0.60-0.65', '0.65-0.70', '0.70-0.75',
#               '0.75-0.80', '0.80-0.85', '0.85-0.90', '0.90-0.95', '0.95-1.00']
# counts_2 = [0, 0, 4, 15, 39, 110, 135, 113, 168, 166, 250, 256, 153, 204, 132, 169, 171, 146, 245, 743]

# total_counts_2 = sum(counts_2)
# percentages_2 = [count / total_counts_2 * 100 for count in counts_2]

# ## 4.1
# categories_1 = ['1-2', '3-5', '6-10', '11-15', '16-20',
#               '21-25', '26-30', '36-40', '41-50', '51-60',
#               '61-70', '70-80', '81-90', '91-100', '101-150', '150以上']
# counts_1 = [11702, 3550, 1914, 859, 510, 350, 242, 134, 205, 125, 107, 73, 51, 52, 158, 179]

# total_counts_1 = sum(counts_1)
# percentages_1 = [count / total_counts_1 * 100 for count in counts_1]
# # # 设置图形大小
# plt.figure(figsize=(12, 10))

# # 创建第一个子图
# plt.subplot(3, 1, 1)  # 创建一个包含一行两列的子图，当前子图为第一个
# plt.title('(a) 歧义词分布情况', fontsize=18,y=-0.75)
# plt.xlabel('单词出现频率', fontsize=14)
# plt.ylabel('单词数量', fontsize=14)
# x = np.arange(len(categories_1))
# plt.xticks(x, categories_1, rotation=45)
# plt.bar(x, counts_1, color='lightblue')


# # 添加第二个y轴表示占比
# plt.twinx()
# plt.ylabel('占比', fontsize=14)
# plt.plot(x, percentages_1, color='blue', marker='o', linestyle='-', linewidth=1)

# # 创建第二个子图
# plt.subplot(3, 1, 2)  # 创建一个包含一行两列的子图，当前子图为第二个
# plt.title('(b) 词义标签分布情况', fontsize=18,y = -0.75)
# plt.xlabel('最常见义项占该词语样本数的百分比', fontsize=14)
# plt.ylabel('单词数量', fontsize=14)
# x = np.arange(len(categories_2))
# plt.xticks(x, categories_2, rotation=30)
# plt.bar(x, counts_2, color='lightgreen')

# # 添加第二个y轴表示占比
# plt.twinx()
# plt.ylabel('占比', fontsize=14)
# plt.plot(x, percentages_2, color='green', marker='o', linestyle='-', linewidth=1)


# # 创建第三个子图
# plt.subplot(3, 1, 3)  # 创建一个包含一行两列的子图，当前子图为第二个 0.035
# plt.title('(c) 测试集词义标签在训练集的分布情况', fontsize=18,y = -0.75)
# plt.xlabel('词义出现频率', fontsize=14)
# plt.ylabel('样本数量', fontsize=14)
# x = np.arange(len(categories_3))
# plt.xticks(x, categories_3, rotation=45)
# plt.bar(x, counts_3, color='lavender')

# # 添加第二个y轴表示占比
# plt.twinx()
# plt.ylabel('占比', fontsize=14)
# plt.plot(x, percentages_3, color='indigo', marker='o', linestyle='-', linewidth=1)

# # 调整子图之间的间距
# plt.subplots_adjust(hspace=1.25)

# # 显示图形
# plt.show()


# 4.4
# # 数据
# categories = ['0', '1-2', '3-5', '6-10', '10+']
# SER = [66.7, 61.9, 72.3, 76.6, 85.9]
# SER_DA = [88.4, 85.1, 86.4, 81.0, 72.8]
# SER_DA_plus = [86.6, 80.9, 84.7, 83.6, 83.9]
# SER_DA_plus_L = [86.0, 80.4, 85.6, 81.6, 85.5]

# # 设置图形大小
# plt.figure(figsize=(10, 8))

# # 设置标题和坐标轴标签
# # plt.title('Performance Comparison', fontsize=16)
# plt.xlabel('样本出现概率', fontsize=14)
# plt.ylabel('F1-Score', fontsize=14)

# # 设置x轴刻度
# x = np.arange(len(categories))
# plt.xticks(x, categories)

# # 设置柱形图的宽度
# bar_width = 0.2

# # 定义配色
# colors = ['#fc8d62', '#ffd92f', '#e5c494', '#b3b3b3']

# # 绘制柱状图
# plt.bar(x - bar_width, SER, width=bar_width, label='SER', color=colors[0])
# for i, v in enumerate(SER):
#     plt.text(x[i] - bar_width, v, str(round(v, 1)), ha='center', va='bottom', fontsize=10)

# plt.bar(x, SER_DA, width=bar_width, label='SER-DA', color=colors[1])
# for i, v in enumerate(SER_DA):
#     plt.text(x[i], v, str(round(v, 1)), ha='center', va='bottom', fontsize=10)

# plt.bar(x + bar_width, SER_DA_plus, width=bar_width, label='SER-DA+', color=colors[2])
# for i, v in enumerate(SER_DA_plus):
#     plt.text(x[i] + bar_width, v, str(round(v, 1)), ha='center', va='bottom', fontsize=10)

# plt.bar(x + 2 * bar_width, SER_DA_plus_L, width=bar_width, label='SER-L-DA+', color=colors[3])
# for i, v in enumerate(SER_DA_plus_L):
#     plt.text(x[i] + 2 * bar_width, v, str(round(v, 1)), ha='center', va='bottom', fontsize=10)

# # 添加图例
# plt.legend(fontsize=12)

# # 设置y轴范围
# plt.ylim(60, 100)

# # 显示网格线
# plt.grid(axis='y', linestyle='--', alpha=0.5)

# # 调整子图之间的间距
# plt.subplots_adjust(left=0.1, right=0.95, bottom=0.15, top=0.9)

# # 显示图形
# plt.show()


# 数据
categories = ['0.0-0.1', '0.1-0.2', '0.2-0.3', '0.3-0.4', '0.4-0.5', '0.5-0.6', '0.6-0.7', '0.7-0.8', '0.8-0.9', '0.9-1.0']
KLS = [0, 0, 0, 0, 1.35, 18.51, 41.17, 28.80, 9.17, 0.96]
WN = [0, 0, 0, 0, 0.13, 3.41, 24.71, 44.96, 23.60, 3.02]
FEWS = [0, 0, 0, 0, 0.10, 4.76, 30.01, 44.90, 18.82, 1.41]

# 设置图形大小
plt.figure(figsize=(8, 6))

# 设置标题和坐标轴标签
# plt.title('Distribution Comparison', fontsize=16)
plt.xlabel('相似度范围', fontsize=14)
plt.ylabel('占比', fontsize=14)

# 设置x轴刻度
x = np.arange(len(categories))
plt.xticks(x, categories, rotation=45)

# 设置柱形图的宽度
bar_width = 0.25

# 定义配色
colors = ['#7f7f7f', '#bcbd22', '#17becf']

# 绘制柱状图
plt.bar(x - bar_width, KLS, width=bar_width, label='KLS', color=colors[0])
plt.bar(x, WN, width=bar_width, label='WN', color=colors[1])
plt.bar(x + bar_width, FEWS, width=bar_width, label='FEWS', color=colors[2])

# 添加图例
plt.legend(fontsize=12)

# 设置y轴范围
plt.ylim(0, 100)

# 显示网格线
plt.grid(axis='y', linestyle='--', alpha=0.5)

# 调整子图之间的间距
plt.subplots_adjust(left=0.1, right=0.95, bottom=0.2, top=0.9)

# 显示图形
plt.show()



