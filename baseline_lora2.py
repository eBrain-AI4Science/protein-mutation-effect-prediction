import pandas as pd
import torch
from scipy.stats import spearmanr
from transformers import AutoModelForSequenceClassification, AutoTokenizer, TrainingArguments, Trainer
from datasets import Dataset
from peft import LoraConfig, get_peft_model, TaskType # <--- NEW IMPORT

# --- CONFIGURATION ---
DATA_PATH = "RNC_ECOLI_Weeks_2023.csv"
MODEL_CHECKPOINT = "facebook/esm2_t33_650M_UR50D"

# OPTIMIZATION 1: Push Batch Size. LoRA saves memory, so let's use it.
# If 16 crashes (OOM), go back to 8.
BATCH_SIZE = 4

# OPTIMIZATION 2: Higher LR. LoRA needs to learn aggressively.
LR = 1e-3 

# OPTIMIZATION 3: More Epochs. LoRA takes longer to converge.
EPOCHS = 5      

print(f"--- STARTING LoRA EXPERIMENT: {MODEL_CHECKPOINT} on RNC_ECOLI ---")

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

# --- OPTIMIZED LoRA CONFIGURATION ---
peft_config = LoraConfig(
    task_type=TaskType.SEQ_CLS, 
    inference_mode=False, 
    r=8, 
    lora_alpha=16, 
    lora_dropout=0.1,
    # OPTIMIZATION 4: Target fewer modules. 
    # Query/Value is the "80/20 rule" of attention—most gain, least cost.
    target_modules=["query", "value"], 
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
    output_dir="./lora_results_rnc_optimized",
    eval_strategy="epoch",
    save_strategy="epoch",
    learning_rate=LR,
    per_device_train_batch_size=BATCH_SIZE,
    per_device_eval_batch_size=BATCH_SIZE,
    gradient_accumulation_steps=1, # We increased batch size to 16, so we can reduce this
    num_train_epochs=EPOCHS,
    weight_decay=0.01,
    fp16=True,
    logging_dir='./logs_lora',
    load_best_model_at_end=True,
    metric_for_best_model="spearman_rho",
    
    # OPTIMIZATION 5: Utilize the CPUs you requested in Slurm!
    dataloader_num_workers=4, 
    group_by_length=True, # Speed up training by grouping similar length sequences
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

