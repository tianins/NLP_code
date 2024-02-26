#!/usr/bin/env python3  
# coding: utf-8

# 使用基于词向量的分类器
# 对比几种模型的效果

import json
import jieba
import numpy as np
from gensim.models import Word2Vec
from sklearn.metrics import classification_report
from sklearn.svm import SVC
from collections import defaultdict

LABELS = {'健康': 0, '军事': 1, '房产': 2, '社会': 3, '国际': 4, '旅游': 5, '彩票': 6, '时尚': 7, '文化': 8, '汽车': 9,
          '体育': 10, '家居': 11, '教育': 12, '娱乐': 13, '科技': 14, '股票': 15, '游戏': 16, '财经': 17}


# 输入模型文件路径
# 加载训练好的模型
def load_word2vec_model(path):
    model = Word2Vec.load(path)
    return model


# 加载数据集
def load_sentence(path, model):
    sentences = []
    labels = []
    all_data = json.load(open(path, 'rb'))
    for line in all_data:
        content = line["review"]
        sentences.append(" ".join(jieba.lcut(content)))
        labels.append(line["label"])
    train_x = sentences_to_vectors(sentences, model)
    train_y = labels
    return train_x, train_y


# tag标签转化为类别标号
def label_to_label_index(labels):
    return [LABELS[y] for y in labels]


# 文本向量化，使用了基于这些文本训练的词向量
def sentences_to_vectors(sentences, model):
    vectors = []
    for sentence in sentences:
        words = sentence.split()
        vector = np.zeros(model.vector_size)
        for word in words:
            try:
                vector += model.wv[word]
                # vector = np.max([vector, model.wv[word]], axis=0)
            except KeyError:
                vector += np.zeros(model.vector_size)
        vectors.append(vector / len(words))
    return np.array(vectors)


def main():
    model = load_word2vec_model("model.w2v")
    train_x, train_y = load_sentence("../data/Sentiment_classification_data_processing/new_train_data.json", model)
    test_x, test_y = load_sentence("../data/Sentiment_classification_data_processing/test_data.json", model)
    classifier = SVC()
    classifier.fit(train_x, train_y)
    y_pred = classifier.predict(test_x)
    print(classification_report(test_y, y_pred))


if __name__ == "__main__":
    main()
