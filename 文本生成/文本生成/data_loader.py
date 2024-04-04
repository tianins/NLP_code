import json
import re
import os
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from config import Config
from datasets import Dataset, concatenate_datasets

class DataGenerator:
    def __init__(self, config):
        self.data = None
        self.config = config
        # self.path = config['data_path']
        self.model_id = config['model_id']
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_id)
        self.load()


    def load(self):
        with open(self.config['org_train_data'], 'r') as f:
            train_data = json.load(f)
        with open(self.config['org_test_data'], 'r') as f:
            test_data = json.load(f)
        with open(self.config['org_dev_data'], 'r') as f:
            dev_data = json.load(f)
        # 使用 Dataset.from_dict() 创建三个 Dataset 对象
        self.train_dataset = Dataset.from_list(train_data)
        self.test_dataset = Dataset.from_list(test_data)
        self.dev_dataset = Dataset.from_list(dev_data)
        # 合并数据集
        self.dataset = concatenate_datasets([self.train_dataset, self.test_dataset, self.dev_dataset])

        self.train_tokenized_dataset = self.train_dataset.map(self.preprocess_function, batched=True).save_to_disk("data/train")
        self.dev_tokenized_dataset = self.dev_dataset.map(self.preprocess_function, batched=True).save_to_disk("data/dev")
        self.test_tokenized_dataset = self.test_dataset.map(self.preprocess_function, batched=True).save_to_disk("data/test")

        return

    def preprocess_function(self, sample, padding="max_length"):
        # add prefix to the input for t5
        # inputs = ["summarize: " + item for item in sample["dialogue"]]

        # tokenize inputs
        model_inputs = self.tokenizer(sample["question"], max_length=self.config["max_source_length"], padding=padding, truncation=True)

        # Tokenize targets with the `text_target` keyword argument
        labels = self.tokenizer(text_target=sample["answer"], max_length=self.config["max_target_length"], padding=padding,
                           truncation=True)

        # If we are padding here, replace all tokenizer.pad_token_id in the labels by -100 when we want to ignore
        # padding in the loss.
        if padding == "max_length":
            labels["input_ids"] = [
                [(l if l != self.tokenizer.pad_token_id else -100) for l in label] for label in labels["input_ids"]
            ]

        model_inputs["labels"] = labels["input_ids"]
        return model_inputs


    def statistical_max_len(self):
        tokenized_inputs = []
        for i in self.dataset:
            tokenized_inputs.append([self.tokenizer(i["question"], truncation=True),self.tokenizer(i["answer"], truncation=True)])
        input_lenghts = [len(x[0]['input_ids']) for x in tokenized_inputs]
        target_lenghts = [len(x[1]['input_ids']) for x in tokenized_inputs]
        # take 95 percentile of max length for better utilization
        max_source_length = int(np.percentile(input_lenghts, 95))
        print(f"Max source length: {max_source_length}")
        max_target_length = int(np.percentile(target_lenghts, 95))
        print(f"Max target length: {max_target_length}")


dg = DataGenerator(Config)
t1 = {
        "id": "identity_0",
        "ex_id": "weak!@!adj!@!11!@!2",
        "conversations": [
            {
                "from": "human",
                "value": "\"His short stories tend to be weak on plot .\" What does the word \"weak\" mean in this context?"
            },
            {
                "from": "gpt",
                "value": "Your weak points are the qualities or talents you do not possess , or the things you are not very good at ."
            }
        ]
    }

# print(dg.encode_sentence(t1))

dg.statistical_max_len()