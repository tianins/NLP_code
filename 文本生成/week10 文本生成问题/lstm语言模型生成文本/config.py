Config = {
    "bert_requires_grad": False,
    "path_log_loss":"model_output/loss_file/bert",
    "use_bert": True,
    "pretrain_model_path": r"/data1/hqp_w/pre_train_model/models--bert-base-chinese/",
    "vocab_path": r"/data1/hqp_w/pre_train_model/models--bert-base-chinese/vocab.txt",
    "model_path": "model_output",
    "schema_path": "ner_data/schema.json",
    "train_data_path": "ner_data/train",
    "valid_data_path": "ner_data/test",
    "max_length": 50,
    "hidden_size": 768,
    "epoch": 10,
    "batch_size": 32,
    "optimizer": "adam",
    "learning_rate": 1e-3,
    "use_crf": False,
    "class_num": None
}