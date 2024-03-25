# -*- coding: utf-8 -*-

"""
配置参数信息
"""

Config = {
    "use_bert": True,
    "pretrain_model_path": r"E:\data\hub\bert_base_chinese",
    "vocab_path": r"E:\data\hub\bert_base_chinese\vocab.txt",
    "model_path": "model_output",
    "schema_path": "data/schema.json",
    "train_data_path": "data/train_corpus.txt",
    "valid_data_path": "data/valid_corpus.txt",
    "max_length": 50,
    "hidden_size": 768,
    "epoch": 5,
    "batch_size": 32,
    "optimizer": "adam",
    "learning_rate": 1e-5,
    "use_crf": False,
    "class_num": None
}