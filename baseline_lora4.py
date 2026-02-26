import pandas as pd
import torch
import numpy as np
from scipy.stats import spearmanr
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
)
from datasets import Dataset
from peft import LoraConfig, get_peft_model, TaskType

# --- CONFIGURATION ---
DATA_PATH = "datasets/RNC_ECOLI_Weeks_2023.csv"
# MODEL_CHECKPOINT = "facebook/esm2_t36_3B_UR50D"
MODEL_CHECKPOINT = "esm2_t33_650M_UR50D"

# LoRA hyperparameters
LORA_R = 16
LORA_ALPHA = 32
LORA_DROPOUT = 0.1

# Training hyperparameters
BATCH_SIZE = 4
LR = 1e-4              # LoRA can use higher LR than full FT
EPOCHS = 3

print(f"--- STARTING LoRA FINE-TUNING: {MODEL_CHECKPOINT} ---")

# 1. Load & Prepare Data
df = pd.read_csv(DATA_PATH)
df = df[['mutated_sequence', 'DMS_score']]
df = df.rename(columns={'mutated_sequence': 'sequence', 'DMS_score': 'label'})
df = df.dropna(subset=['label'])

dataset = Dataset.from_pandas(df)
dataset = dataset.train_test_split(test_size=0.2, seed=42)

print(f"Training on {len(dataset['train'])} sequences.")
print(f"Testing on {len(dataset['test'])} sequences.")

# 2. Tokenization
tokenizer = AutoTokenizer.from_pretrained(MODEL_CHECKPOINT)

def tokenize_function(examples):
    return tokenizer(
        examples["sequence"],
        padding="max_length",
        truncation=True,
        max_length=512
    )

tokenized_datasets = dataset.map(tokenize_function, batched=True)
tokenized_datasets = tokenized_datasets.remove_columns(["sequence"])
tokenized_datasets.set_format("torch")

# 3. Load Base Model
model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_CHECKPOINT,
    num_labels=1,
    problem_type="regression"
)

# 4. LoRA Configuration
lora_config = LoraConfig(
    r=LORA_R,
    lora_alpha=LORA_ALPHA,
    target_modules=["query", "key", "value", "dense"],
    lora_dropout=LORA_DROPOUT,
    bias="none",
    task_type=TaskType.SEQ_CLS
)

model = get_peft_model(model, lora_config)

# Print trainable parameter count
model.print_trainable_parameters()

# 5. Metrics
def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    predictions = predictions.squeeze()

    rho, _ = spearmanr(predictions, labels)
    if np.isnan(rho):
        rho = 0.0

    mse = ((predictions - labels) ** 2).mean()
    return {"spearman_rho": rho, "mse": mse}

# 6. Training Arguments
training_args = TrainingArguments(
    output_dir="./results/baseline_results_lora",
    eval_strategy="epoch",
    save_strategy="epoch",
    learning_rate=LR,
    per_device_train_batch_size=BATCH_SIZE,
    per_device_eval_batch_size=BATCH_SIZE,
    gradient_accumulation_steps=2,
    num_train_epochs=EPOCHS,
    weight_decay=0.01,
    fp16=True,
    logging_dir="./logs_lora",
    load_best_model_at_end=True,
    metric_for_best_model="spearman_rho",
    greater_is_better=True,
    dataloader_num_workers=4,
    save_total_limit=1,
)

# 7. Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["test"],
    compute_metrics=compute_metrics,
)

# 8. Train
print("Starting LoRA Training...")

max_gpu_mem_gb = None
if torch.cuda.is_available():
    torch.cuda.reset_peak_memory_stats()

trainer.train()

if torch.cuda.is_available():
    max_gpu_mem_gb = torch.cuda.max_memory_allocated() / (1024 ** 3)

# 9. Final Evaluation
print("Evaluating on Test Set...")
metrics = trainer.evaluate()

print("\n--- FINAL LoRA RESULTS ---")
print(f"Model: {MODEL_CHECKPOINT}")
print(f"Spearman Rho: {metrics['eval_spearman_rho']:.4f}")
print(f"MSE Loss: {metrics['eval_mse']:.4f}")
if max_gpu_mem_gb:
    print(f"Max GPU memory allocated during training: {max_gpu_mem_gb:.2f} GB")
else:
    print("Max GPU memory allocated during training: N/A (no CUDA)")

print("---------------------------")

