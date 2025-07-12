import os
import pickle
import pandas as pd
import matplotlib.pyplot as plt
import csv

from sklearn.model_selection import train_test_split
from datasets import Dataset, ClassLabel
import evaluate
from sklearn.metrics import confusion_matrix, classification_report

from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    DataCollatorWithPadding,
)

# 1. Load DataFrame
with open('/home/rock/vulExp/vulCodeBERT/vul_BERT_dataframe_cleaned.pkl', 'rb') as f:
    df = pickle.load(f)

# 2. First split: 80% train, 20% temp (for valid + test)
train_df, temp_df = train_test_split(
    df, test_size=0.2, stratify=df["label"], random_state=42
)

# 3. Second split: 50% valid, 50% test from 20%
valid_df, test_df = train_test_split(
    temp_df, test_size=0.5, stratify=temp_df["label"], random_state=42
)

# Reset index
train_df = train_df.reset_index(drop=True)
valid_df = valid_df.reset_index(drop=True)
test_df = test_df.reset_index(drop=True)

# 4. Create Hugging Face Datasets
label_feature = ClassLabel(names=["normal", "buggy"])
train_ds = Dataset.from_pandas(train_df).cast_column('label', label_feature)
valid_ds = Dataset.from_pandas(valid_df).cast_column('label', label_feature)
test_ds = Dataset.from_pandas(test_df).cast_column('label', label_feature)

# 5. Load tokenizer and model
model_name = 'microsoft/codebert-base'
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)

# 6. Preprocessing
def preprocess(example):
    tokens = tokenizer(example['content'], truncation=True, max_length=512)
    tokens['labels'] = example['label']
    return tokens

train_ds = train_ds.map(preprocess, batched=True)
valid_ds = valid_ds.map(preprocess, batched=True)
test_ds = test_ds.map(preprocess, batched=True)

# 7. Data collator
data_collator = DataCollatorWithPadding(tokenizer)

# 8. Metrics
metric = evaluate.load('accuracy')

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = logits.argmax(-1)
    return metric.compute(predictions=preds, references=labels)

# 9. Training arguments
output_dir = './codebert_finetuned_V13'
training_args = TrainingArguments(
    output_dir=output_dir,
    eval_strategy='epoch',
    save_strategy='epoch',
    logging_strategy='epoch', # 改成只在每個 epoch 結束時 log
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    num_train_epochs=10,
    weight_decay=0.01,
    save_total_limit=2,
    load_best_model_at_end=True,
    metric_for_best_model='accuracy',
)

# 10. Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_ds,
    eval_dataset=valid_ds,
    tokenizer=tokenizer,
    data_collator=data_collator,
    compute_metrics=compute_metrics,
)

# 11. Train and save
trainer.train()
trainer.save_model(output_dir)
print(f"Model and tokenizer saved to {output_dir}")

# 12. Evaluate on validation set and output confusion matrix
print("\nValidation Set Evaluation:")
val_predictions = trainer.predict(valid_ds)
val_preds = val_predictions.predictions.argmax(-1)
val_labels = val_predictions.label_ids

val_cm = confusion_matrix(val_labels, val_preds)
val_report = classification_report(val_labels, val_preds, target_names=["normal", "buggy"])

print("Confusion Matrix (Valid):\n", val_cm)
print("\nClassification Report (Valid):\n", val_report)

# 13. Evaluate on test set with confusion matrix
print("\nTest Set Evaluation:")
test_predictions = trainer.predict(test_ds)
test_preds = test_predictions.predictions.argmax(-1)
test_labels = test_predictions.label_ids

test_cm = confusion_matrix(test_labels, test_preds)
test_report = classification_report(test_labels, test_preds, target_names=["normal", "buggy"])

print("Confusion Matrix (Test):\n", test_cm)
print("\nClassification Report (Test):\n", test_report)

# Also include HuggingFace's metric output
test_metrics = trainer.evaluate(test_ds)
print("Test set metrics:", test_metrics)

# 14. Parse training and evaluation loss history, and save to CSV
log_history = trainer.state.log_history
train_loss_log = []
eval_loss_log = []

for log in log_history:
    if 'loss' in log and 'epoch' in log and 'eval_loss' not in log:
        train_loss_log.append({
            'epoch': int(log['epoch']),
            'train_loss': log['loss'],
        })
    elif 'eval_loss' in log and 'epoch' in log:
        eval_loss_log.append({
            'epoch': int(log['epoch']),
            'eval_loss': log['eval_loss'],
        })

# 整合成同一個列表
combined_loss_log = []
for train_entry in train_loss_log:
    epoch = train_entry['epoch']
    matching_eval = next((e for e in eval_loss_log if e['epoch'] == epoch), None)
    combined_loss_log.append({
        'epoch': epoch,
        'train_loss': train_entry['train_loss'],
        'eval_loss': matching_eval['eval_loss'] if matching_eval else None
    })

# Save to CSV
csv_path = os.path.join(output_dir, "training_validation_loss_log.csv")
with open(csv_path, mode='w', newline='') as csv_file:
    writer = csv.DictWriter(csv_file, fieldnames=['epoch', 'train_loss', 'eval_loss'])
    writer.writeheader()
    writer.writerows(combined_loss_log)

print(f"Training & validation loss log saved to {csv_path}")

# 15. Plot both training and validation loss curves
epochs = [entry['epoch'] for entry in combined_loss_log]
train_losses = [entry['train_loss'] for entry in combined_loss_log]
eval_losses = [entry['eval_loss'] for entry in combined_loss_log]

plt.figure(figsize=(12, 7))
plt.plot(epochs, train_losses, marker='o', label='Training Loss')
plt.plot(epochs, eval_losses, marker='s', label='Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training and Validation Loss Curve')
plt.xticks(epochs)
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig(os.path.join(output_dir, "training_and_validation_loss_curve.png"))
plt.close()

print(f"Loss curve saved to {os.path.join(output_dir, 'training_and_validation_loss_curve.png')}")
