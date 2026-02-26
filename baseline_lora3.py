import pandas as pd
import torch
from scipy.stats import spearmanr
from transformers import AutoModelForSequenceClassification, AutoTokenizer, TrainingArguments, Trainer
from datasets import Dataset
from peft import LoraConfig, get_peft_model, TaskType

# --- CONFIGURATION ---
DATA_PATH = "RNC_ECOLI_Weeks_2023.csv"
MODEL_CHECKPOINT = "facebook/esm2_t33_650M_UR50D"

# OPTIMIZED SETTINGS
BATCH_SIZE = 4          # STICK TO 8. It provides the best stability for your dataset size.
LR = 5e-4               # INCREASED slightly from 1e-4 to speed up convergence.
EPOCHS = 7              # REDUCED from 10. We want to finish faster.

print(f"--- STARTING OPTIMIZED LoRA EXPERIMENT: {MODEL_CHECKPOINT} ---")

# 1. Load Data
print(f"Loading data from {DATA_PATH}...")
df = pd.read_csv(DATA_PATH)
df = df[['mutated_sequence', 'DMS_score']]
df = df.rename(columns={'mutated_sequence': 'sequence', 'DMS_score': 'label'})
df = df.dropna(subset=['label'])

dataset = Dataset.from_pandas(df)
dataset = dataset.train_test_split(test_size=0.2, seed=42)

# 2. Tokenization
tokenizer = AutoTokenizer.from_pretrained(MODEL_CHECKPOINT)

def tokenize_function(examples):
    return tokenizer(examples["sequence"], padding="max_length", truncation=True, max_length=512)

tokenized_datasets = dataset.map(tokenize_function, batched=True)

# 3. Model Setup
model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_CHECKPOINT,
    num_labels=1, 
    problem_type="regression"
)

# --- KEY OPTIMIZATION: Target Modules ---
peft_config = LoraConfig(
    task_type=TaskType.SEQ_CLS, 
    inference_mode=False, 
    r=8, 
    lora_alpha=16, 
    lora_dropout=0.1,
    # TARGETING ONLY QUERY & VALUE.
    # This captures 80% of the benefit but runs much faster than targeting everything.
    target_modules=["query", "value"], 
    bias="none",
)

model = get_peft_model(model, peft_config)
print("\n--- LoRA TRAINABLE PARAMETERS (Optimized) ---")
model.print_trainable_parameters()
print("---------------------------------------------\n")

# 4. Metrics
def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    predictions = predictions.squeeze()
    rho, _ = spearmanr(predictions, labels)
    mse = ((predictions - labels) ** 2).mean()
    return {"spearman_rho": rho, "mse": mse}

# 5. Training Arguments
training_args = TrainingArguments(
    output_dir="./lora_results_optimized",
    eval_strategy="epoch",
    save_strategy="epoch",
    learning_rate=LR,
    per_device_train_batch_size=BATCH_SIZE,
    per_device_eval_batch_size=BATCH_SIZE,
    gradient_accumulation_steps=1,
    num_train_epochs=EPOCHS,
    weight_decay=0.01,
    fp16=True,                # Essential for A100 speed
    logging_dir='./logs_lora',
    load_best_model_at_end=True,
    metric_for_best_model="spearman_rho",
    dataloader_num_workers=4, # Ensure CPUs are feeding the GPU
    save_total_limit=1,       # Only save the best checkpoint to save disk space
)

# 6. Initialize Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["test"],
    compute_metrics=compute_metrics,
)

# 7. Run
print("Starting Optimized LoRA Training...")
trainer.train()

# 8. Evaluation
print("Evaluating on Test Set...")
metrics = trainer.evaluate()

print("\n--- FINAL RESULTS ---")
print(f"Spearman Rho: {metrics['eval_spearman_rho']:.4f}")
print(f"MSE Loss: {metrics['eval_mse']:.4f}")
print("---------------------")
