from datasets import load_dataset
import os
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, DataCollatorForSeq2Seq, Seq2SeqTrainer, \
    Seq2SeqTrainingArguments
from peft import LoraConfig, get_peft_model, prepare_model_for_int8_training, TaskType
from datasets import Dataset
from config import Config
# E:\data\hub\flan_t5\base
# Load dataset from the hub
# os.environ["HTTP_PROXY"] = "http://192.168.29.1:3034"
# os.environ["HTTPS_PROXY"] = "http://192.168.29.1:3034"
# dataset = load_dataset(r"E:\data\hub\hf_cache\downloads\samsum")
# 梯子代理端口，设置成你自己的


class ModelTrainer:
    def __init__(self, config):
        self.config = config
        self.peft_model_id = config.get('peft_model_id', 'lora-flan-t5-base')
        self.model_id = config.get('model_id')
    def train_model(self):
        tokenizer = AutoTokenizer.from_pretrained(self.model_id, device_map='0')
        # s = "\"His short stories tend to be weak on plot .\" What does the word \"weak\" mean in this context?"
        # print(tokenizer(s))

        # Load model from the hub
        model = AutoModelForSeq2SeqLM.from_pretrained(self.model_id, load_in_8bit=True, device_map="auto")
        if self.config['lora']:
            # Define LoRA Config
            lora_config = LoraConfig(
                r=16,
                lora_alpha=32,
                target_modules=["q", "v"],
                lora_dropout=0.05,
                bias="none",
                task_type=TaskType.SEQ_2_SEQ_LM
            )

            # prepare int-8 model for training
            model = prepare_model_for_int8_training(model)

            # add LoRA adaptor
            model = get_peft_model(model, lora_config)
            model.print_trainable_parameters()

        # We want to ignore tokenizer pad token in the loss
        label_pad_token_id = -100

        # Data collator
        data_collator = DataCollatorForSeq2Seq(
            tokenizer,
            model=model,
            label_pad_token_id=label_pad_token_id,
            pad_to_multiple_of=8
        )

        output_dir = self.config.get('output_dir', "lora-flan-t5-base")

        # Define training args
        training_args = Seq2SeqTrainingArguments(
            output_dir=output_dir,
            auto_find_batch_size=True,
            learning_rate=1e-3,  # Higher learning rate
            num_train_epochs=5,
            logging_dir=f"{output_dir}/logs",
            logging_strategy="steps",
            logging_steps=500,
            save_strategy="epoch",
            report_to="tensorboard"
        )

        tokenized_dataset = Dataset.load_from_disk(self.config["train_data"])

        # Create Trainer instance
        trainer = Seq2SeqTrainer(
            model=model,
            args=training_args,
            data_collator=data_collator,
            train_dataset=tokenized_dataset
        )

        model.config.use_cache = False  # Silence the warnings. Please re-enable for inference!

        # Train model
        trainer.train()

        # Save our LoRA model & tokenizer results
        peft_model_id = self.peft_model_id
        trainer.model.save_pretrained(peft_model_id)
        tokenizer.save_pretrained(peft_model_id)
        # If you want to save the base model to call
        # trainer.model.base_model.save_pretrained(peft_model_id)

# Example usage:

trainer = ModelTrainer(Config)
trainer.train_model()