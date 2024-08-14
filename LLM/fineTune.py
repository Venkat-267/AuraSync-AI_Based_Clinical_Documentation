from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig, DataCollatorForLanguageModeling, Trainer, TrainingArguments
import torch
from torch.utils.data import DataLoader, Dataset
from peft import prepare_model_for_kbit_training, LoraConfig, get_peft_model
import pandas as pd

model_id = "tiiuae/falcon-7b-instruct"
tokenizer_name = "tiiuae/falcon-7b-instruct"  # Update with the actual tokenizer name

tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
tokenizer.pad_token = tokenizer.eos_token

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16
)
model = AutoModelForCausalLM.from_pretrained(model_id, quantization_config=bnb_config, device_map={"":0}, trust_remote_code=True)

model.gradient_checkpointing_enable()
model = prepare_model_for_kbit_training(model)

config = LoraConfig(
    r=16,
    lora_alpha=32,
    target_modules=["query_key_value"],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM"
)
model = get_peft_model(model, config)

def print_trainable_parameters(model):
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    all_params = sum(p.numel() for p in model.parameters())
    print(f"trainable params: {trainable_params} || all params: {all_params} || trainable%: {100 * trainable_params / all_params}")
print_trainable_parameters(model)

dataset_path = 'dataset.csv'
df = pd.read_csv(dataset_path)

input_encodings = tokenizer(list(df['Input Data'].values), truncation=True, padding=True, max_length=None, return_tensors='pt')
output_encodings = tokenizer(list(df['Output Data'].values), truncation=True, padding=True, max_length=None, return_tensors='pt')

class CustomDataset(Dataset):
    def __init__(self, input_encodings, output_encodings):
        self.input_encodings = input_encodings
        self.output_encodings = output_encodings

    def __getitem__(self, idx):
        input_item = {key: torch.tensor(val[idx]) for key, val in self.input_encodings.items()}
        output_item = {key: torch.tensor(val[idx]) for key, val in self.output_encodings.items()}
        return input_item, output_item

    def __len__(self):
        return len(self.input_encodings["input_ids"])

train_dataset = CustomDataset(input_encodings, output_encodings)

training_args = TrainingArguments(
    output_dir="outputs",
    num_train_epochs=50,
    per_device_train_batch_size=8,
    save_steps=100,
    save_total_limit=2,
    logging_dir="logs",
    logging_steps=10,
    do_train=True,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    data_collator=DataCollatorForLanguageModeling(tokenizer, mlm=False),
)


trainer.train()

eval_results = trainer.evaluate()

print("Evaluation results:")
for key, value in eval_results.items():
    print(f"{key}: {value}")

tokenizer.save_pretrained("tokenizer_path")
