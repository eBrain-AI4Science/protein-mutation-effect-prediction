import pandas as pd
import torch
from scipy.stats import spearmanr
from transformers import AutoModelForSequenceClassification, AutoTokenizer, TrainingArguments, Trainer
from datasets import Dataset
from peft import LoraConfig, get_peft_model, TaskType # <--- NEW IMPORT

# --- CONFIGURATION ---
DATA_PATH = "datasets/RNC_ECOLI_Weeks_2023.csv"
MODEL_CHECKPOINT = "facebook/esm2_t33_650M_UR50D"
BATCH_SIZE = 8  # changed from 4. This will stabilize training and be much faster
LR = 1e-4        # <--- CHANGED: LoRA usually needs a higher LR (1e-4 or 1e-3), changed from 1e-4. This helps it learn faster
EPOCHS = 10      # changed from 3 or 5 

print(f"--- STARTING LoRA EXPERIMENT: {MODEL_CHECKPOINT} ---")

# 1. Load Data
print(f"Loading data from {DATA_PATH}...")
df = pd.read_csv(DATA_PATH)
df = df[['mutated_sequence', 'DMS_score']]
df = df.rename(columns={'mutated_sequence': 'sequence', 'DMS_score': 'label'})
df = df.dropna(subset=['label'])

dataset = Dataset.from_pandas(df)
dataset = dataset.train_test_split(test_size=0.2, seed=42)

print(f"Data loaded Successfully.")
print(f"Training on {len(dataset['train'])} sequences.")

# 2. Tokenization
tokenizer = AutoTokenizer.from_pretrained(MODEL_CHECKPOINT)

def tokenize_function(examples):
    return tokenizer(examples["sequence"], padding="max_length", truncation=True, max_length=512)

tokenized_datasets = dataset.map(tokenize_function, batched=True)

# 3. Model Setup (Base Model)
model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_CHECKPOINT,
    num_labels=1, 
    problem_type="regression"
)

# --- NEW: LoRA CONFIGURATION ---
peft_config = LoraConfig(
    task_type=TaskType.SEQ_CLS, # Sequence Classification/Regression
    inference_mode=False, 
    r=8,                        # Rank: Higher = more parameters (8 is standard)
    lora_alpha=16,              # Alpha: Scaling factor (usually 2x rank)
    lora_dropout=0.1,           # Dropout to prevent overfitting
    target_modules=["query", "key", "value", "dense"], # Target attention heads in ESM-2
    bias="none",
)

# Wrap the base model with LoRA
model = get_peft_model(model, peft_config)

print("\n--- LoRA TRAINABLE PARAMETERS ---")
model.print_trainable_parameters()
print("---------------------------------\n")

# 4. Metrics Function
def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    predictions = predictions.squeeze()
    rho, _ = spearmanr(predictions, labels)
    mse = ((predictions - labels) ** 2).mean()
    return {"spearman_rho": rho, "mse": mse}

# 5. Training Arguments
training_args = TrainingArguments(
    output_dir="./results/lora_results_rnc",  # <--- CHANGED FOLDER NAME
    eval_strategy="epoch",
    save_strategy="epoch",
    learning_rate=LR,
    per_device_train_batch_size=BATCH_SIZE,
    per_device_eval_batch_size=BATCH_SIZE,
    gradient_accumulation_steps=2,
    num_train_epochs=EPOCHS,
    weight_decay=0.01,
    fp16=True,
    logging_dir='./logs_lora',        # <--- CHANGED LOG FOLDER
    load_best_model_at_end=True,
    metric_for_best_model="spearman_rho",
)

# 6. Initialize Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["test"],
    compute_metrics=compute_metrics,
)

# 7. Run Training
print("Starting LoRA Training...")
trainer.train()

# 8. Final Evaluation
print("Evaluating on Test Set...")
metrics = trainer.evaluate()

print("\n--- FINAL LoRA RESULTS ---")
print(f"Model: {MODEL_CHECKPOINT}")
print(f"Spearman Rho: {metrics['eval_spearman_rho']:.4f}")
print(f"MSE Loss: {metrics['eval_mse']:.4f}")
print("------------------------------")
