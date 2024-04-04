import evaluate
import numpy as np
from datasets import load_from_disk
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from tqdm import tqdm
from peft import PeftModel, PeftConfig
from config import Config

class ModelEvaluator:
    def __init__(self, config, num_samples=None):
        self.model_id = config["model_id"]
        self.peft_model_id = config["peft_model_id"]
        self.max_target_length = config["max_target_length"]
        self.data_path = config["test_data"]
        self.num_samples = num_samples

        if config['lora']:
            # Load peft config for pre-trained checkpoint etc.
            peft_config = PeftConfig.from_pretrained(self.peft_model_id)
            # load base LLM model and tokenizer
            self.model = AutoModelForSeq2SeqLM.from_pretrained(peft_config.base_model_name_or_path, load_in_8bit=True,
                                                               device_map={"": 0})
            self.tokenizer = AutoTokenizer.from_pretrained(peft_config.base_model_name_or_path)

            # Load the Lora model
            self.model = PeftModel.from_pretrained(self.model, self.peft_model_id, device_map={"": 0})
            self.model.eval()
        else:
            self.model = AutoModelForSeq2SeqLM.from_pretrained(self.model_id, load_in_8bit=True, device_map={"": 0})
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_id)
            self.model.eval()

        # Metric
        self.metric = evaluate.load(r"F:\code\evaluate-main\evaluate-main\metrics\rouge\rouge.py")

    def evaluate_peft_model(self, sample):
        # generate summary
        outputs = self.model.generate(input_ids=sample["input_ids"].unsqueeze(0).cuda(), do_sample=True, top_p=0.9,
                                      max_new_tokens=self.max_target_length)
        prediction = self.tokenizer.decode(outputs[0].detach().cpu().numpy(), skip_special_tokens=True)
        # 解码label
        # Replace -100 in the labels as we can't decode them.
        # labels = np.where(sample['labels'] != -100, sample['labels'], tokenizer.pad_token_id)
        # labels = tokenizer.decode(labels, skip_special_tokens=True)
        # 这里不需要解码label，直接从sample里面取
        labels = sample['answer']
        # Some simple post-processing
        return prediction, labels

    def run_evaluation(self):
        # load test dataset from disk
        test_dataset = load_from_disk(self.data_path)
        if self.num_samples is None:
            self.num_samples = len(test_dataset)
        test_dataset = test_dataset.shuffle(seed=42).select(range(self.num_samples)).with_format("torch")

        # run predictions
        # this can take ~45 minutes
        predictions, references = [], []
        for sample in tqdm(test_dataset):
            p, l = self.evaluate_peft_model(sample)
            predictions.append(p)
            references.append(l)

        # compute metric
        rogue = self.metric.compute(predictions=predictions, references=references, use_stemmer=True)

        # print results
        print(f"Rogue1: {rogue['rouge1'] * 100:2f}%")
        print(f"rouge2: {rogue['rouge2'] * 100:2f}%")
        print(f"rougeL: {rogue['rougeL'] * 100:2f}%")
        print(f"rougeLsum: {rogue['rougeLsum'] * 100:2f}%")


# Example usage,很奇怪，每次测的结果都不一样
evaluator = ModelEvaluator(Config, num_samples=100)
evaluator.run_evaluation()
