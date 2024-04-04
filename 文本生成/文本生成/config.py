# -*- coding: utf-8 -*-

"""
配置参数信息
"""
import os
import torch

Config = {
    "lora": True,
    "org_test_data": "org_data/test_data_temp.json",
    "org_dev_data": "org_data/dev_data_temp.json",
    "org_train_data": "org_data/train_data_temp.json",
    "test_data": "data/test/",
    "dev_data": "data/dev/",
    "train_data": "data/train/",
    "peft_model_id": None,  # lora-flan-t5-base
    "model_id": r"E:\data\hub\flan_t5\base",
    "train_data_path": "flant5/train_data_temp.json",
    "max_source_length": 39,
    "max_target_length": 43,
    "test_data_path": "flant5/test_data_temp.json",
    "dev_data_path": "flant5/dev_data_temp.json",
    "input_max_length": 120,
    "output_max_length": 30,
    "epoch": 200,
    "batch_size": 32,
    "optimizer": "adam",
    "learning_rate":1e-3,
    "seed":42,
    "vocab_size":6219,
    "vocab_path":"vocab.txt",
    "valid_data_path": r"sample_data.json",
    "beam_size":5
    }

