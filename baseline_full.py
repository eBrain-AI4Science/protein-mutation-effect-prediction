import pandas as pd
import torch
from scipy.stats import spearmanr
from transformers import AutoModelForSequenceClassification, AutoTokenizer, TrainingArguments, Trainer
from datasets import Dataset

# --- CONFIGURATION ---
DATA_PATH = "datasets/Doud_NCAP_I34A1_2015.csv"
# MODEL_CHECKPOINT = "facebook/esm2_t36_3B_UR50D" 
MODEL_CHECKPOINT = "esm2_t33_650M_UR50D"
BATCH_SIZE = 4   # 650M is large; keep batch size small to prevent crashing
LR = 1e-5        # Standard fine-tuning learning rate
EPOCHS = 5

print(f"--- STARTING BASELINE: {MODEL_CHECKPOINT}  ---")

# 1. Load Data Directly from file
print(f"Loading data from {DATA_PATH}...")
df = pd.read_csv(DATA_PATH)

# The dataset columns are 'primary' (sequence) and 'log_fluorescence' (label)
# We limit to 5000 samples for the baseline to save time (The full set is ~50k)
# Remove the .head(5000) if you want to run the full experiment (takes longer)
# df = df[['primary', 'log_fluorescence']].rename(columns={'primary': 'sequence', 'log_fluorescence': 'label'}).head(5000)

# 1. Select only the columns we need
# We map 'mutated_sequence' -> 'sequence' (Input)
# We map 'DMS_score' -> 'label' (Target)
df = df[['mutated_sequence', 'DMS_score']]

# Rename columns to match Hugging Face standard
# (Check the file with 'head -n 5 filename.csv' if these names are wrong)
df = df.rename(columns={'mutated_sequence': 'sequence', 'DMS_score': 'label'})

# 2. Clean Data
# Remove any rows where the experiment failed and didn't give a score (NaN)
df = df.dropna(subset=['label'])

# 3. Create Dataset
dataset = Dataset.from_pandas(df)

# 4. Split: 80% Train, 20% Test (Validation)
dataset = dataset.train_test_split(test_size=0.2, seed=42)
print(f"Data loaded Successfully.")
print(f"Training on {len(dataset['train'])} sequences.")
print(f"Testing on {len(dataset['test'])} seuqences.")


# 2. Tokenization
tokenizer = AutoTokenizer.from_pretrained(MODEL_CHECKPOINT)

def tokenize_function(examples):
    # Truncate to 512 to ensure we don't hit memory limits
    return tokenizer(examples["sequence"], padding="max_length", truncation=True, max_length=512)

tokenized_datasets = dataset.map(tokenize_function, batched=True)

# 3. Model Setup (REGRESSION)
# num_labels=1 because we are predicting a single continuous number (brightness)
model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_CHECKPOINT,
    num_labels=1, 
    problem_type="regression"
)

# 4. Metrics Function (Spearman's Rho)
def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    predictions = predictions.squeeze() 
    
    # Calculate Spearman's Rank Correlation
    rho, _ = spearmanr(predictions, labels)
    mse = ((predictions - labels) ** 2).mean()
    
    return {"spearman_rho": rho, "mse": mse}

# 5. Training Arguments
training_args = TrainingArguments(
    output_dir="./results/baseline_results_full",
    eval_strategy="epoch",
    save_strategy="epoch",
    learning_rate=LR,
    per_device_train_batch_size=BATCH_SIZE,
    per_device_eval_batch_size=BATCH_SIZE,
    gradient_accumulation_steps=2, # Helps simulate larger batches
    num_train_epochs=EPOCHS,
    weight_decay=0.01,
    fp16=True,  # Mixed precision (Crucial for A100 speed)
    logging_dir='./logs',
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
print("Starting Training...")

max_gpu_mem_gb = None
if torch.cuda.is_available():
    torch.cuda.reset_peak_memory_stats()

trainer.train()

if torch.cuda.is_available():
    max_gpu_mem_gb = torch.cuda.max_memory_allocated() / (1024 ** 3)

# 8. Final Evaluation
print("Evaluating on Test Set...")
metrics = trainer.evaluate()

print("\n--- FINAL BASELINE RESULTS ---")
print(f"Model: {MODEL_CHECKPOINT}")
print(f"Spearman Rho: {metrics['eval_spearman_rho']:.4f}")
print(f"MSE Loss: {metrics['eval_mse']:.4f}")
if max_gpu_mem_gb is not None:
    print(f"Max GPU memory allocated during training: {max_gpu_mem_gb:.2f} GB")
else:
    print("Max GPU memory allocated during training: N/A (no CUDA)")
print("------------------------------")
