import torch
from peft import PeftModel, PeftConfig
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from datasets import load_from_disk
from random import randrange
from config import Config

if Config['lora']:
    # Load peft config for pre-trained checkpoint etc.
    peft_config = PeftConfig.from_pretrained(Config['peft_model_id'])
    # load base LLM model and tokenizer
    model = AutoModelForSeq2SeqLM.from_pretrained(peft_config.base_model_name_or_path, load_in_8bit=True,
                                                       device_map={"": 0})
    tokenizer = AutoTokenizer.from_pretrained(peft_config.base_model_name_or_path)

    # Load the Lora model
    model = PeftModel.from_pretrained(model, Config['peft_model_id'], device_map={"": 0})
    print("Peft model loaded")
    model.eval()
else:
    model = AutoModelForSeq2SeqLM.from_pretrained(Config['model_id'], load_in_8bit=True, device_map={"": 0})
    tokenizer = AutoTokenizer.from_pretrained(Config['model_id'])
    model.eval()
    print("model loaded")



# Load dataset from the hub and get a sample
dataset = load_from_disk("data/test").select(range(5))
for i in dataset:
    input_ids = tokenizer(i["question"], return_tensors="pt", truncation=True).input_ids.cuda()
    # with torch.inference_mode():
    outputs = model.generate(input_ids=input_ids, max_new_tokens=20, do_sample=True, top_p=0.9)
    print(f"Input sentence: \n{i['question']}")
    print(f"Ref answer: \n{i['answer']}")
    print(f"Predict answer:\n{tokenizer.batch_decode(outputs.detach().cpu().numpy(), skip_special_tokens=True)[0]}\n{'---'* 20}")

