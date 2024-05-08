## 基于lora的序列标注任务

### 未使用lora

```
2024-05-06 10:49:17,983 - __main__ - INFO - epoch 10 begin
2024-05-06 10:49:18,151 - __main__ - INFO - batch loss 0.013058
2024-05-06 10:49:21,649 - __main__ - INFO - batch loss 0.013560
2024-05-06 10:49:25,040 - __main__ - INFO - batch loss 0.003226
2024-05-06 10:49:25,040 - __main__ - INFO - epoch average loss: 0.015124
2024-05-06 10:49:25,040 - __main__ - INFO - 开始测试第10轮模型效果：
2024-05-06 10:49:25,630 - __main__ - INFO - PERSON类实体，准确率：0.976608, 召回率: 0.965318, F1: 0.970925
2024-05-06 10:49:25,631 - __main__ - INFO - LOCATION类实体，准确率：0.848780, 召回率: 0.915789, F1: 0.881008
2024-05-06 10:49:25,631 - __main__ - INFO - TIME类实体，准确率：0.898089, 召回率: 0.865031, F1: 0.881245
2024-05-06 10:49:25,631 - __main__ - INFO - ORGANIZATION类实体，准确率：0.804348, 召回率: 0.870588, F1: 0.836153
2024-05-06 10:49:25,632 - __main__ - INFO - Macro-F1: 0.892333
2024-05-06 10:49:25,632 - __main__ - INFO - Micro-F1 0.899671
2024-05-06 10:49:25,632 - __main__ - INFO - --------------------



2024-05-06 11:10:38,025 - __main__ - INFO - --------------------
2024-05-06 11:10:38,026 - __main__ - INFO - epoch 10 begin
2024-05-06 11:10:38,193 - __main__ - INFO - batch loss 0.009942
2024-05-06 11:10:41,762 - __main__ - INFO - batch loss 0.016558
2024-05-06 11:10:45,159 - __main__ - INFO - batch loss 0.055546
2024-05-06 11:10:45,159 - __main__ - INFO - epoch average loss: 0.023834
2024-05-06 11:10:45,159 - __main__ - INFO - 开始测试第10轮模型效果：
2024-05-06 11:10:45,746 - __main__ - INFO - PERSON类实体，准确率：0.853261, 召回率: 0.907514, F1: 0.879547
2024-05-06 11:10:45,746 - __main__ - INFO - LOCATION类实体，准确率：0.811321, 召回率: 0.895833, F1: 0.851480
2024-05-06 11:10:45,746 - __main__ - INFO - TIME类实体，准确率：0.833333, 召回率: 0.889570, F1: 0.860529
2024-05-06 11:10:45,747 - __main__ - INFO - ORGANIZATION类实体，准确率：0.734694, 召回率: 0.847059, F1: 0.786880
2024-05-06 11:10:45,747 - __main__ - INFO - Macro-F1: 0.844609
2024-05-06 11:10:45,747 - __main__ - INFO - Micro-F1 0.852454
2024-05-06 11:10:45,747 - __main__ - INFO - --------------------
为什么取消特殊标记后效果下降？难道是因为没有对应上位置？
仔细检查后发现，位置是对应的，可能是因为预训练时有cls和sep取消后导致预训练的参数不再适用？

2024-05-06 11:21:03,900 - __main__ - INFO - --------------------
2024-05-06 11:21:03,901 - __main__ - INFO - epoch 10 begin
2024-05-06 11:21:04,065 - __main__ - INFO - batch loss 0.011752
2024-05-06 11:21:07,558 - __main__ - INFO - batch loss 0.005611
2024-05-06 11:21:10,955 - __main__ - INFO - batch loss 0.010024
2024-05-06 11:21:10,956 - __main__ - INFO - epoch average loss: 0.013555
2024-05-06 11:21:10,957 - __main__ - INFO - 开始测试第10轮模型效果：
2024-05-06 11:21:11,554 - __main__ - INFO - PERSON类实体，准确率：0.948276, 召回率: 0.953757, F1: 0.951004
2024-05-06 11:21:11,555 - __main__ - INFO - LOCATION类实体，准确率：0.890625, 召回率: 0.900000, F1: 0.895283
2024-05-06 11:21:11,555 - __main__ - INFO - TIME类实体，准确率：0.903846, 召回率: 0.865031, F1: 0.884007
2024-05-06 11:21:11,555 - __main__ - INFO - ORGANIZATION类实体，准确率：0.797980, 召回率: 0.929412, F1: 0.858691       
2024-05-06 11:21:11,556 - __main__ - INFO - Macro-F1: 0.897246
2024-05-06 11:21:11,556 - __main__ - INFO - Micro-F1 0.902592

差距还是很明显的0.897246 -> 0.844609
```



使用lora

1. 使用peft.get_peft_model(model, peft_config)函数的model必须是AutoModelForTokenClassification导出的模型，不能是AutoModel方法

   ```
   model = AutoModelForTokenClassification.from_pretrained(config["pretrain_model_path"], num_labels=class_num)
   正常
   
   
   model = BertModel.from_pretrained(pretrain_model_path) 
   
   model = AutoModel.from_pretrained(pretrain_model_path) 
   
   但是BertModel和AutoModel方法导出的模型，会报错
   AttributeError: 'NoneType' object has no attribute 'named_parameters'
   ```

   

2. 不能使用拼接提示模板的方法

   ```
   predict = self.bert_encoder(x)[0]
   输出应该是 bs,max_len,class_nums
   但是使用提示模板方法后会在max_len上增加num_virtual_tokens=10
   导致输出的形状为bs,max_len+num_virtual_tokens,class_nums
   
   一方面导致后面分类器的max_len维度对不上，另一方面对于序列标注任务
   增加的序列长度没有对应的标签，还需要额外的处理，如增加label的长度，但是这后面很难处理
   
   所以只能使用lora_tuning微调
   
   from peft import TaskType 参数含义
   SEQ_CLS: 代表序列分类任务（Sequence Classification）。这种任务通常涉及将输入序列分类到不同的类别中。
   SEQ_2_SEQ_LM: 代表序列到序列的语言建模任务（Sequence-to-Sequence Language Modeling）。这种任务涉及将输入序列映射到输出序列，通常用于机器翻译、文本摘要等任务。
   CAUSAL_LM: 代表因果语言建模任务（Causal Language Modeling）。这种任务涉及使用语言模型来生成一个序列，其中每个标记都是通过前面的标记预测得到的。
   TOKEN_CLS: 代表标记分类任务（Token Classification）。这种任务涉及对输入序列中的每个标记进行分类，通常用于命名实体识别、词性标注等任务。
   应该使用TOKEN_CLS进行序列标注
   
   
   很奇怪，为什么使用lora_tuning微调结果一直为零，loss在下降
   难道又是解码出了问题？
   
   经过排查，发现是lr的问题，lr设置太小了对应lora微调方法来说
   全量微调的bert-base，lr设置为1e-5
   但是局部微调1e-5的lr，预测的准确率一直为零
   调整为1e-3后，结果正常
   2024-05-08 18:17:59,335 - __main__ - INFO - epoch 10 begin
   2024-05-08 18:17:59,442 - __main__ - INFO - batch loss 0.007840
   2024-05-08 18:18:01,806 - __main__ - INFO - batch loss 0.029627
   2024-05-08 18:18:04,113 - __main__ - INFO - batch loss 0.001422
   2024-05-08 18:18:04,113 - __main__ - INFO - epoch average loss: 0.014895
   2024-05-08 18:18:04,114 - __main__ - INFO - 开始测试第10轮模型效果：
   2024-05-08 18:18:04,761 - __main__ - INFO - PERSON类实体，准确率：0.970414, 召回率: 0.947977, F1: 0.959059
   2024-05-08 18:18:04,761 - __main__ - INFO - LOCATION类实体，准确率：0.857868, 召回率: 0.889474, F1: 0.873380
   2024-05-08 18:18:04,761 - __main__ - INFO - TIME类实体，准确率：0.899371, 召回率: 0.877301, F1: 0.888194
   2024-05-08 18:18:04,761 - __main__ - INFO - ORGANIZATION类实体，准确率：0.843373, 召回率: 0.823529, F1: 0.833328
   2024-05-08 18:18:04,761 - __main__ - INFO - Macro-F1: 0.888490
   2024-05-08 18:18:04,762 - __main__ - INFO - Micro-F1 0.895811
   2024-05-08 18:18:04,762 - __main__ - INFO - --------------------
   ```

   



