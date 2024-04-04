## 基于flan-t5的指令微调

### 环境设置

```shell
conda create -n flan_t5 python=3.8 

# install Hugging Face Libraries
pip install "peft==0.2.0"
pip install "transformers==4.27.2" "datasets==2.9.0" "accelerate==0.17.1" "evaluate==0.4.0" "bitsandbytes==0.37.1" loralib --upgrade --quiet
# install additional dependencies needed for training
pip install rouge-score tensorboard py7zr


# 注意peft==0.2.0要求torch>=1.13.0
# bitsandbytes 在win上要安装0.43.0版本，不然检测不到cuda环境
```

### 词义消歧任务描述

根据单词的上下文识别目标词的正确词义

```
目标词：mention
上下文：At the community centre , mention of funds produces pained looks .
词义：A mention is a reference to something or someone .
```

传统做法一般是将词义消歧任务转为文本分类或者文本匹配问题，计算目标词在上下文中的词向量与词典中的候选词义向量之间的相似度，取最高作为预测结果，计算预测结果的F1得分。

本项目将词义消歧视为文本生成任务，结合提示模板，直接生成目标词在上下文中的词义。

### 数据集


使用（[**Improved Word Sense Disambiguation via Prompt-based Contextual Word Representation**](10.1109/IJCNN54540.2023.10191929)）论文中构建的柯林斯词典数据集，整理成问答对的形式
指令模板为
```
"<context></context>" What does the word "<word></word>" mean in this context?
其中<word></word>标签为目标词，<context></context>为包含目标词的上下文

{
    "id": "identity_21",
    "data_name": "dev_data",
    "ex_id": "cost!@!noun!@!1!@!1",
    "question": "\"The company admits its costs are still too high .\" What does the word \"cost\" mean in this context?",
    "answer": "Your costs are the total amount of money that you must spend on running your home or business ."
},

其中训练集47754条，发展集、测试集各5000条
```

### 实验

#### 模型介绍

```
Flan-T5是Google的一篇工作，通过在超大规模的任务上进行微调，让语言模型具备了极强的泛化性能，做到单个模型就可以在1800多个NLP任务上都能有很好的表现。这意味着模型一旦训练完毕，可以直接在几乎全部的NLP任务上直接使用，实现One model for ALL tasks。相比T5模型有很大提升，本项目在flan-t5-base模型上进行微调实验。
```

#### 实验结果

```
基础模型：google/flan-t5-base 参数量248M params

直接使用未微调过的基础模型测试结果：
    Rogue1: 7.614877%
    rouge2: 0.100000%
    rougeL: 6.694853%
    rougeLsum: 6.716570%

使用loar微调结果
trainable params: 1769472 || all params: 249347328 || trainable%: 0.7096414524241463
    epoch 5 
    lr 1e-5 
    bs 256 
    lora 
    显存占用 8G 
    训练时间 15min

    Rogue1: 21.447311%
    rouge2: 3.178228%
    rougeL: 16.996222%
    rougeLsum: 16.900268%
    
    epoch 5 
    lr 1e-5 
    bs 64 
    显存占用 8G 
    训练时间 12min
    {'loss': 2.708, 'learning_rate': 8.69281045751634e-06, 'epoch': 0.65}
    {'loss': 2.3104, 'learning_rate': 7.385620915032681e-06, 'epoch': 1.31}
    {'loss': 2.1956, 'learning_rate': 6.07843137254902e-06, 'epoch': 1.96}
    {'loss': 2.1365, 'learning_rate': 4.77124183006536e-06, 'epoch': 2.61}
    {'loss': 2.0997, 'learning_rate': 3.4640522875816997e-06, 'epoch': 3.27}
    {loss': 2.0803, 'learning_rate': 2.1568627450980393e-06, 'epoch': 3.92}
    {'loss': 2.0697, 'learning_rate': 8.496732026143792e-07, 'epoch': 4.58}
    {'train_runtime': 676.8072, 'train_samples_per_second': 361.565, 'train_steps_per_second': 5.652, 'train_loss': 2.215144697201797, 'epoch': 5.0}
    Rogue1: 27.351753%
    rouge2: 5.118346%
    rougeL: 21.637615%
    rougeLsum: 21.718479%
    
全量微调结果
    epoch 5 
    lr 1e-5 
    bs 64 （256会爆显存） 
    all 
    显存占用 12G 
    训练时间 12min
    测试时间 6min
	{'loss': 2.0697, 'learning_rate': 8.69281045751634e-06, 'epoch': 0.65}
	{'loss': 1.851, 'learning_rate': 7.385620915032681e-06, 'epoch': 1.31}
	{'loss': 1.7898, 'learning_rate': 6.07843137254902e-06, 'epoch': 1.96}
	{'loss': 1.7553, 'learning_rate': 4.77124183006536e-06, 'epoch': 2.61}
	{'loss': 1.7257, 'learning_rate': 3.4640522875816997e-06, 'epoch': 3.27}             
	{'loss': 1.7111, 'learning_rate': 2.1568627450980393e-06, 'epoch': 3.92}
	{'loss': 1.7014, 'learning_rate': 8.496732026143792e-07, 'epoch': 4.58}       
	{'train_runtime': 748.7587, 'train_samples_per_second': 326.821, 		'train_steps_per_second': 5.108, 'train_loss': 1.791899733200572, 'epoch': 5.0}
	
	Rogue1: 33.810928%
	rouge2: 9.934081%
	rougeL: 27.916232%
	rougeLsum: 27.883607%
	tensorboard --logdir F:\code\NLP_code\文本生成\文本生成\log
```



### 结论

使用指令微调能够提升模型在下游让任务的效果

直接使用Flant5_base进行词义消歧测试的效果很差rougeL值6.69，lora微调后提升到21.63，全量微调提升到27.9。

loar方法在提升性能的同时还能降低显存的占用，如果使用更大规模Flant5模型会进一步提高下游任务的效果。







### 附录

```
datasets使用教程
https://zhuanlan.zhihu.com/p/564816807
```

#### 样例展示

```
org model

Input sentence: 
"Exposure of unprotected skin to the sun carries the risk of developing skin cancer ." What does the word "unprotected" mean in this context?
Ref answer: 
If something is unprotected , it is not covered or treated with anything , and so it may easily be damaged .
Predict answer:
skin that is exposed to the sun
------------------------------------------------------------
Input sentence: 
"He would only aggravate the injury by rubbing it ." What does the word "aggravate" mean in this context?
Ref answer: 
If someone or something aggravates a situation , they make it worse .
Predict answer:
he rubbing a
------------------------------------------------------------
Input sentence: 
"The company consolidated some operations last summer ." What does the word "consolidate" mean in this context?
Ref answer: 
To consolidate a number of small groups or firms means to make them into one large organization .
Predict answer:
oneself
------------------------------------------------------------
Input sentence: 
"The road was reduced to 18ft in width by adding parking bays ." What does the word "width" mean in this context?
Ref answer: 
The width of something is the distance it measures from one side or edge to the other .
Predict answer:
width
------------------------------------------------------------
Input sentence: 
"The whole atmosphere has really been poisoned ." What does the word "poison" mean in this context?
Ref answer: 
Something that poisons a good situation or relationship spoils it or destroys it .
Predict answer:
is an atmosphere
------------------------------------------------------------
```



```
full parameter ft

Input sentence: 
"Exposure of unprotected skin to the sun carries the risk of developing skin cancer ." What does the word "unprotected" mean in this context?
Ref answer: 
If something is unprotected , it is not covered or treated with anything , and so it may easily be damaged .
Predict answer:
Something that is unprotected is not very well protected.
------------------------------------------------------------
Input sentence: 
"He would only aggravate the injury by rubbing it ." What does the word "aggravate" mean in this context?
Ref answer: 
If someone or something aggravates a situation , they make it worse .
Predict answer:
If you aggravate a bad thing, you cause it to worsen.
------------------------------------------------------------
Input sentence: 
"The company consolidated some operations last summer ." What does the word "consolidate" mean in this context?
Ref answer: 
To consolidate a number of small groups or firms means to make them into one large organization .
Predict answer:
If someone or something consolidates or becomes more popular, they grow in numbers, popularity, or importance.
------------------------------------------------------------
Input sentence: 
"The road was reduced to 18ft in width by adding parking bays ." What does the word "width" mean in this context?
Ref answer: 
The width of something is the distance it measures from one side or edge to the other .
Predict answer:
You can use width in quotes to describe something that you are referring to.
------------------------------------------------------------
Input sentence: 
"The whole atmosphere has really been poisoned ." What does the word "poison" mean in this context?
Ref answer: 
Something that poisons a good situation or relationship spoils it or destroys it .
Predict answer:
Poison is a poison that causes or causes a lot of harm to people or to animals.
```



```
lora 64

Input sentence: 
"Exposure of unprotected skin to the sun carries the risk of developing skin cancer ." What does the word "unprotected" mean in this context?
Ref answer: 
If something is unprotected , it is not covered or treated with anything , and so it may easily be damaged .
Predict answer:
If one person's skin is unprotected, there is no chance of developing a disease.
------------------------------------------------------------
Input sentence: 
"He would only aggravate the injury by rubbing it ." What does the word "aggravate" mean in this context?
Ref answer: 
If someone or something aggravates a situation , they make it worse .
Predict answer:
If something happens to increase in intensity, it aggravates its condition or damage.
------------------------------------------------------------
Input sentence: 
"The company consolidated some operations last summer ." What does the word "consolidate" mean in this context?
Ref answer: 
To consolidate a number of small groups or firms means to make them into one large organization .
Predict answer:
A consolidated operation is a business activity that enables people to do something, or to do something more formally.
------------------------------------------------------------
Input sentence: 
"The road was reduced to 18ft in width by adding parking bays ." What does the word "width" mean in this context?
Ref answer: 
The width of something is the distance it measures from one side or edge to the other .
Predict answer:
If a road or building is wide, it is in a place where it is tall, and it is narrow.
------------------------------------------------------------
Input sentence: 
"The whole atmosphere has really been poisoned ." What does the word "poison" mean in this context?
Ref answer: 
Something that poisons a good situation or relationship spoils it or destroys it .
Predict answer:
If you poison something, you destroy it.
------------------------------------------------------------
```



```
lora 256

Input sentence: 
"Exposure of unprotected skin to the sun carries the risk of developing skin cancer ." What does the word "unprotected" mean in this context?
Ref answer: 
If something is unprotected , it is not covered or treated with anything , and so it may easily be damaged .
Predict answer:
Unprotected skin may expose the skin to a disproportionate amount of sun exposure, or to a small amount of ultraviolet rays, thereby making it vulnerable to the effects of the
------------------------------------------------------------
Input sentence: 
"He would only aggravate the injury by rubbing it ." What does the word "aggravate" mean in this context?
Ref answer: 
If someone or something aggravates a situation , they make it worse .
Predict answer:
aggravate, cause, or affect the damage, or effect the condition of an object or person.
------------------------------------------------------------
Input sentence: 
"The company consolidated some operations last summer ." What does the word "consolidate" mean in this context?
Ref answer: 
To consolidate a number of small groups or firms means to make them into one large organization .
Predict answer:
Because this word means putting together several things in order to consolidate one thing.
------------------------------------------------------------
Input sentence: 
"The road was reduced to 18ft in width by adding parking bays ." What does the word "width" mean in this context?
Ref answer: 
The width of something is the distance it measures from one side or edge to the other .
Predict answer:
A width means the height, or width, of a road.
------------------------------------------------------------
Input sentence: 
"
The whole atmosphere has really been poisoned ." What does the word "poison" mean in this context?
Ref answer: 
Something that poisons a good situation or relationship spoils it or destroys it .
Predict answer:
The poisoned people can be killed.
------------------------------------------------------------
```

