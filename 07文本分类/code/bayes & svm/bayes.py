import math
import jieba
import re
import os
import json
from collections import defaultdict
from sklearn.metrics import precision_recall_fscore_support
jieba.initialize()

"""
贝叶斯分类实践

P(A|B) = (P(A) * P(B|A)) / P(B)
事件A：文本属于类别x1。文本属于类别x的概率，记做P(x1)
事件B：文本为s (s=w1w2w3..wn)
P(x1|s) = 文本为s，属于x1类的概率.   #求解目标#
P(x1|s) = P(x1|w1, w2, w3...wn) = P(w1, w2..wn|x1) * P(x1) / P(w1, w2, w3...wn)

P(x1) 任意样本属于x1的概率。x1样本数/总样本数
P(w1, w2..wn|x1) = P(w1|x1) * P(w2|x1)...P(wn|x1)  词的独立性假设
P(w1|x1) x1类样本中，w1出现的频率

公共分母的计算，使用全概率公式：
P(w1, w2, w3...wn) = P(w1,w2..Wn|x1)*P(x1) + P(w1,w2..Wn|x2)*P(x2) ... P(w1,w2..Wn|xn)*P(xn)

# 这种做法的前提是样本的类别分布要相对均匀，符合实际，不然计算出的P(x1)，P(x2)的差距过大，影响最终的结果
# 尝试修改为评论情感分类
"""


class BayesApproach:
    def __init__(self, data_path):
        self.p_class = defaultdict(int)
        self.word_class_prob = defaultdict(dict)
        self.load(data_path)
        self.test = False
        self.test_path = '../data/Sentiment_classification_data_processing/test_data.json'
        self.max_len = 80
        self.clean = True

    def load(self, path):
        self.class_name_to_word_freq = defaultdict(dict)
        self.all_words = set()  # 汇总一个词表
        all_data = json.load(open(path, 'rb'))
        for line in all_data:
            class_name = line["label"]
            title = line["review"]
            words = jieba.lcut(title)
            self.all_words.union(set(words))
            self.p_class[class_name] += 1  # 记录每个类别样本数量
            word_freq = self.class_name_to_word_freq[class_name]
            # 记录每个类别下的词频
            for word in words:
                if word not in word_freq:
                    word_freq[word] = 1
                else:
                    word_freq[word] += 1
        self.freq_to_prob()
        return

    # 将记录的词频和样本频率都转化为概率
    def freq_to_prob(self):
        # 样本概率计算
        total_sample_count = sum(self.p_class.values())
        self.p_class = dict([c, self.p_class[c] / total_sample_count] for c in self.p_class)
        # 词概率计算
        self.word_class_prob = defaultdict(dict)
        for class_name, word_freq in self.class_name_to_word_freq.items():
            total_word_count = sum(count for count in word_freq.values())  # 每个类别总词数
            for word in word_freq:
                # 加1平滑，避免出现概率为0，计算P(wn|x1)
                prob = (word_freq[word] + 1) / (total_word_count + len(self.all_words))
                self.word_class_prob[class_name][word] = prob
            self.word_class_prob[class_name]["<unk>"] = 1 / (total_word_count + len(self.all_words))
        return

    # P(w1|x1) * P(w2|x1)...P(wn|x1)
    def get_words_class_prob(self, words, class_name):
        result = 1
        for word in words:
            unk_prob = self.word_class_prob[class_name]["<unk>"]
            result *= self.word_class_prob[class_name].get(word, unk_prob)
        return result

    # 计算P(w1, w2..wn|x1) * P(x1)
    def get_class_prob(self, words, class_name):
        # P(x1)
        p_x = self.p_class[class_name]
        # P(w1, w2..wn|x1) = P(w1|x1) * P(w2|x1)...P(wn|x1)
        p_w_x = self.get_words_class_prob(words, class_name)
        return p_x * p_w_x

    # 做文本分类
    def clean_text(self, text):
        # 移除标点
        cleaned_text = re.sub(r'[^\w\s\u4e00-\u9fa5]', ' ', text)
        return cleaned_text
    def classify(self, sentence):
        if self.clean:
            sentence = self.clean_text(sentence)  # 将标点替换为空
        words = jieba.lcut(sentence)  # 切词
        words = [word for word in words if word != ' ']  # 移除空字符
        # max_len
        words = words[:self.max_len]
        results = []  # 这里为什么会出现都是零的情况？ 文本太长，单词在这里类别中出现的概率的累乘会无限接近0，需要提前截断
        for class_name in self.p_class:
            prob = self.get_class_prob(words, class_name)  # 计算class_name类概率
            results.append([class_name, prob])
        results = sorted(results, key=lambda x: x[1], reverse=True)  # 排序

        # 计算公共分母：P(w1, w2, w3...wn) = P(w1,w2..Wn|x1)*P(x1) + P(w1,w2..Wn|x2)*P(x2) ... P(w1,w2..Wn|xn)*P(xn)
        # 不做这一步也可以，对顺序没影响，只不过得到的不是0-1之间的概率值
        pw = sum([x[1] for x in results])  # P(w1, w2, w3...wn)
        try:
            # 尝试计算并更新结果列表
            results = [[c, prob / pw] for c, prob in results]
        except (AttributeError, ZeroDivisionError) as e:
            # 处理异常，例如打印错误信息
            print(f"An error occurred: {e}")
            # 可以选择在这里设置默认值，或者采取其他恢复措施
            results = []  # 或者设置为其他默认值

        # 打印结果
        if self.test:
            for class_name, prob in results:
                print("属于类别[%s]的概率为%f" % (class_name, prob))
        return results

    def evaluate(self):
        test_data = json.load(open(self.test_path, 'rb'))
        label_data = []
        pred_data = []
        for i in test_data:
            label_data.append(i['label'])
            prob1, prob2 = self.classify(i['review'])
            pred_data.append(prob1[0])
        confusion_matrix = {}
        matches = 0
        if len(pred_data) != 0:
            for i in range(len(label_data)):
                if label_data[i] == pred_data[i]:
                    matches += 1
                string = str(int(label_data[i])) + " --> " + str(int(pred_data[i]))
                if string in confusion_matrix:
                    confusion_matrix[string] += 1
                else:
                    confusion_matrix[string] = 1
            acc = float(matches) / float(len(label_data))
            print("accuracy:", acc)
            print("confusion_matrix[target --> pred]: %s", confusion_matrix)
        avg = precision_recall_fscore_support(
            label_data, pred_data, average='macro')
        avg_P, avg_R, avg_F = avg[0], avg[1], avg[2]
        print("avg_P: {}, avg_R: {}, avg_F：{}".format(round(avg_P, 4), round(avg_R, 4), round(avg_F, 4)))
        return

if __name__ == "__main__":
    path = "../data/Sentiment_classification_data_processing/train_data.json"
    ba = BayesApproach(path)
    ba.evaluate()
    # query = "专门打电话过去说不放葱，接电话的是个小姑娘，我专门查了下我点的订单，跟她核实了，她也答应按我的要求做。可是送来的时候皮蛋豆腐整整一层葱，从天亮挑葱挑到了天黑。接电话的姑娘，你是瞬间记忆的脑子吗？无力吐槽。。韭菜盒子太焦了，已经趋向于糊了。皮蛋豆腐送来的时候，三分之一是渣状，不是层状的，对此，我也表示理解。。总之不会再跟你家打交道了，连电话都接不好的前台也只能是前台，顶多收个钱。"
    # print(ba.clean_text(query))
    # ba.classify(query)
