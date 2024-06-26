# -*- coding: utf-8 -*-

import json
import re
import os
import torch
import random
import jieba
import numpy as np
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer
"""
数据加载
"""


class DataGenerator:
    def __init__(self, data_path, config):
        self.config = config
        self.path = data_path
        self.vocab = self.load_vocab(config["vocab_path"])
        self.config["vocab_size"] = len(self.vocab)
        self.max_length = self.config["max_length"]
        self.sentences = []
        self.schema = self.load_schema(config["schema_path"])
        self.config["class_num"] = len(self.schema)
        self.load()

    def load(self):
        self.data = []
        with open(self.path, encoding="utf8") as f:
            segments = f.read().split("\n\n")
            for segment in segments:
                sentenece = []
                labels = []
                for line in segment.split("\n"):
                    if line.strip() == "":
                        continue
                    char, label = line.split()
                    sentenece.append(char)
                    labels.append(self.schema[label])
                self.sentences.append("".join(sentenece))
                input_ids = self.encode_sentence(sentenece)
                # 将文本对应标签进行padding,如果要采用预训练模型还需要在cls位置添加一个-1，
                # 取消添加[CLS]和[SEP]标记 model.config.add_special_tokens = False
                
                # 为什么add_special_tokens=False，不起作用，理论上应该不添加cls和sep的。需要在encoder里面加
                
                # 取消cls和sep后就不需要改变label的位置
                if self.config["use_bert"]:
                    labels = [8] + labels  # 在cls位置添加一个-1,sep是不是也要加一个-1,不需要，sep在后面会自动补上-1
                labels = self.padding(labels, -1)
                self.data.append([torch.LongTensor(input_ids), torch.LongTensor(labels)])
        return

    def encode_sentence(self, text, padding=True):
        input_id = []
        if self.config["vocab_path"] == "words.txt":
            for word in jieba.cut(text):
                input_id.append(self.vocab.get(word, self.vocab["[UNK]"]))
        elif self.config["use_bert"]:
                input_id = self.vocab.encode(text,
                                            truncation='longest_first',
                                            max_length=self.max_length,
                                            padding='max_length',
                                            # add_special_tokens=False
                                            )
        else:
            for char in text:
                input_id.append(self.vocab.get(char, self.vocab["[UNK]"]))
        if padding:
            input_id = self.padding(input_id)
        return input_id

    #补齐或截断输入的序列，使其可以在一个batch内运算
    def padding(self, input_id, pad_token=0):
        input_id = input_id[:self.config["max_length"]]
        input_id += [pad_token] * (self.config["max_length"] - len(input_id))
        return input_id

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return self.data[index]

    def load_schema(self, path):
        with open(path, encoding="utf8") as f:
            return json.load(f)

    #加载字表或词表
    def load_vocab(self,vocab_path):
        token_dict = {}
        if self.config["use_bert"]:
                tokenizer = BertTokenizer(vocab_path)
                return tokenizer
        else:
            with open(vocab_path, encoding="utf8") as f:
                for index, line in enumerate(f):
                    token = line.strip()
                    token_dict[token] = index + 1  #0留给padding位置，所以从1开始
        return token_dict

#用torch自带的DataLoader类封装数据
def load_data(data_path, config, shuffle=True):
    dg = DataGenerator(data_path, config)
    dl = DataLoader(dg, batch_size=config["batch_size"], shuffle=shuffle)
    return dl



if __name__ == "__main__":
    from config import Config
    dg = DataGenerator("ner_data/train", Config)

