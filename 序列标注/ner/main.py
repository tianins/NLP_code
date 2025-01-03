# -*- coding: utf-8 -*-

import torch
import os
import random
import torch.nn as nn
import os
import numpy as np
import logging
from config import Config
from model import TorchModel, choose_optimizer
from evaluate import Evaluator
from loader import load_data
# from tensorboardX import SummaryWriter
logging.basicConfig(level = logging.INFO,format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)
from transformers import AutoTokenizer, AutoModelForSequenceClassification, AutoModel
from peft import get_peft_model, LoraConfig, TaskType, \
    PromptTuningConfig, PrefixTuningConfig, PromptEncoderConfig 

"""
模型训练主程序

增加peft微调方法
"""

def main(config):
    #创建保存模型的目录
    if not os.path.isdir(config["model_path"]):
        os.mkdir(config["model_path"])
    if not os.path.isdir(config["path_log_loss"]):
        os.mkdir(config["path_log_loss"])
    #加载训练数据
    train_data = load_data(config["train_data_path"], config)
    #加载模型
    
    # if config['use_peft']:
    #     Torch_Model = AutoModel.from_pretrained(config["pretrain_model_path"])
    #     model = Torch_Model
    # else:
    #     model = TorchModel(config)
    model = TorchModel(config)

    # 标识是否使用gpu
    cuda_flag = torch.cuda.is_available()
    if cuda_flag:
        logger.info("gpu可以使用，迁移模型至gpu")
        model = model.cuda()
    #加载优化器
    optimizer = choose_optimizer(config, model)
    #加载效果测试类
    evaluator = Evaluator(config, model, logger)
    #训练
    # writer = SummaryWriter(log_dir=config["path_log_loss"])
    step = 0
    for epoch in range(config["epoch"]):
        epoch += 1
        model.train()
        logger.info("epoch %d begin" % epoch)
        train_loss = []
        for index, batch_data in enumerate(train_data):
            optimizer.zero_grad()
            if cuda_flag:
                batch_data = [d.cuda() for d in batch_data]
            input_id, labels = batch_data   #输入变化时这里需要修改，比如多输入，多输出的情况
            loss = model(input_id, labels)
            # loss_tensor = torch.tensor(loss)
            step += 1
            # writer.add_scalar('Loss/train', loss_tensor.item(), step)
            loss.backward()
            optimizer.step()
            train_loss.append(loss.item())
            if index % int(len(train_data) / 2) == 0:
                logger.info("batch loss %f" % loss)
        logger.info("epoch average loss: %f" % np.mean(train_loss))
        evaluator.eval(epoch)
    model_path = os.path.join(config["model_path"], "epoch_%d.pth" % epoch)
    # torch.save(model.state_dict(), model_path)
    # writer.close()
    # writer.flush()
    if config['use_peft']:
        save_tunable_parameters(model, model_path)
    return model, train_data

def save_tunable_parameters(model, path):
    saved_params = {
        k: v.to("cpu")
        for k, v in model.named_parameters()
        if v.requires_grad
    }
    torch.save(saved_params, path)

if __name__ == "__main__":
    model, train_data = main(Config)
