import pandas as pd
import json
import random
# 设置随机数种子
random_seed = 22  # 你可以选择任何整数作为种子
random.seed(random_seed)
# 整理csv文件，保存为训练集和验证集的json文件
# orient='records'表示转换成字典的列表,其中每个字典表示DataFrame的一行,json格式[{},{}]
def get_true_label_sum(data_list):
    sum_true = 0
    for i in data_list:
        if i['label'] == 1:
            sum_true += 1
    print(sum_true, len(data_list), sum_true/len(data_list))
    return sum_true
def reduce_negative_samples(data_list):
    sum_true = get_true_label_sum(data_list)
    sum_false = 0
    new_data_list = []
    for i in data_list:
        if i['label'] == 0:
            if sum_false <= sum_true:
                new_data_list.append(i)
            sum_false += 1
        else:
            new_data_list.append(i)
    return new_data_list
all_data = pd.read_csv("文本分类练习.csv").to_dict(orient='records')

test_data = random.sample(all_data, 2400)
print("test_data:")
get_true_label_sum(test_data)
with open('test_data.json', 'w', encoding='utf-8') as f:
    json.dump(test_data, f, ensure_ascii=False, indent=4)  # 这里的ensure_ascii=False表示在Dump时不要ASCII编码中文字符,保留中文的原始Unicode编码
train_data = [x for x in all_data if x not in test_data]
with open('train_data.json', 'w', encoding='utf-8') as f:
    json.dump(train_data, f, ensure_ascii=False, indent=4)
print("train_data:")
get_true_label_sum(train_data)
# 减少负样本数量
new_train_data = reduce_negative_samples(train_data)
with open('new_train_data.json', 'w', encoding='utf-8') as f:
    json.dump(new_train_data, f, ensure_ascii=False, indent=4)
print("train_data:")
get_true_label_sum(new_train_data)
# 需要注意的是这里的正负样本是不均匀的，只有30%的正样本，而且部分样本的标签是错误的
# 样本分布不均可以采用的方法
"""
1.移除训练集中的部分负样本，使得负样本和正样本的比例达到1：1
2.给正样本在训练时添加更大的权重
3.数据增强，再添加一些正样本的标注数据
"""
print(1)