
# 取消cls和sep特殊标记
from transformers import BertTokenizer
tokenizer = BertTokenizer(r"E:\data\hub\bert_base_chinese\vocab.txt")
print(tokenizer.encode("今天天气不错",add_special_tokens=False))