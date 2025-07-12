import os
import pickle
import pandas as pd
import matplotlib.pyplot as plt
import csv

from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_recall_fscore_support, accuracy_score
from datasets import Dataset, ClassLabel
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    DataCollatorWithPadding,
    TrainerCallback,
)

# 1. Load data
with open('/home/rock/vulExp/vulCodeBERT/vul_BERT_dataframe_cleaned.pkl', 'rb') as f:
    df = pickle.load(f)

# 2. Split into train/valid/test (80/10/10)
train_df, temp_df = train_test_split(df, test_size=0.2, stratify=df['label'], random_state=42)
valid_df, test_df = train_test_split(temp_df, test_size=0.5, stratify=temp_df['label'], random_state=42)
for split_df in (train_df, valid_df, test_df):
    split_df.reset_index(drop=True, inplace=True)

# 3. Create HF Datasets
label_feature = ClassLabel(names=["normal", "buggy"])
train_ds = Dataset.from_pandas(train_df).cast_column('label', label_feature)
valid_ds = Dataset.from_pandas(valid_df).cast_column('label', label_feature)
test_ds  = Dataset.from_pandas(test_df).cast_column('label', label_feature)

# 4. Load tokenizer and model
model_name = 'microsoft/codebert-base'
tokenizer  = AutoTokenizer.from_pretrained(model_name)
model      = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)

# 5. Preprocessing

def preprocess(examples):
    tokens = tokenizer(examples['content'], truncation=True, max_length=512)
    tokens['labels'] = examples['label']
    return tokens

train_ds = train_ds.map(preprocess, batched=True)
valid_ds = valid_ds.map(preprocess, batched=True)
test_ds  = test_ds.map(preprocess, batched=True)

data_collator = DataCollatorWithPadding(tokenizer)

# 6. Callback to log metrics per epoch
class MetricsLoggerCallback(TrainerCallback):
    def __init__(self, train_ds, valid_ds, test_ds, output_dir):
        self.train_ds = train_ds
        self.valid_ds = valid_ds
        self.test_ds  = test_ds
        self.output_dir = output_dir
        self.records = []

    def on_epoch_end(self, args, state, control, **kwargs):
        epoch = int(state.epoch)
        for name, ds in [('train', self.train_ds), ('valid', self.valid_ds), ('test', self.test_ds)]:
            # get predictions
            preds_output = kwargs['model']
            # fallback: use trainer reference
            pass

# 7. Training arguments
output_dir = './codebert_finetuned_V13'
training_args = TrainingArguments(
    output_dir=output_dir,
    eval_strategy='epoch',
    save_strategy='epoch',
    logging_strategy='epoch',
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    num_train_epochs=10,
    weight_decay=0.01,
    save_total_limit=10,
    load_best_model_at_end=True,
    metric_for_best_model='accuracy',
)

# 8. Define compute_metrics for validation

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = logits.argmax(-1)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average='macro', zero_division=0)
    acc = accuracy_score(labels, preds)
    return {
        'accuracy': acc,
        'precision': precision,
        'recall': recall,
        'f1': f1,
    }

# 9. Initialize Trainer
metrics_callback = MetricsLoggerCallback(train_ds, valid_ds, test_ds, output_dir)
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_ds,
    eval_dataset=valid_ds,
    tokenizer=tokenizer,
    data_collator=data_collator,
    compute_metrics=compute_metrics,
    callbacks=[metrics_callback],
)

# 10. Train
trainer.train()
# best model auto loaded at end
trainer.save_model(output_dir)
print(f"Best model saved to {output_dir}")

# 11. Save loss history
log_history = trainer.state.log_history
train_loss_log = [ {'epoch': int(l['epoch']), 'train_loss': l['loss']} 
                   for l in log_history if 'loss' in l and 'eval_loss' not in l ]
eval_loss_log  = [ {'epoch': int(l['epoch']), 'eval_loss': l['eval_loss']} 
                   for l in log_history if 'eval_loss' in l ]
combined = []
for t in train_loss_log:
    e = t['epoch']
    ev = next((v for v in eval_loss_log if v['epoch']==e), {})
    combined.append({
        'epoch': e,
        'train_loss': t['train_loss'],
        'eval_loss': ev.get('eval_loss')
    })
loss_csv = os.path.join(output_dir, 'training_validation_loss_log.csv')
with open(loss_csv, 'w', newline='') as f:
    writer = csv.DictWriter(f, fieldnames=['epoch','train_loss','eval_loss'])
    writer.writeheader(); writer.writerows(combined)
print(f"Loss log saved to {loss_csv}")

# 12. Save metrics log from callback
metrics_csv = os.path.join(output_dir, 'metrics_log.csv')
with open(metrics_csv, 'w', newline='') as f:
    writer = csv.DictWriter(f, fieldnames=['epoch','split','precision','recall','f1','accuracy','support'])
    writer.writeheader()
    for rec in metrics_callback.records:
        writer.writerow(rec)
print(f"Metrics log saved to {metrics_csv}")

# 13. Plot loss curve
epochs = [e['epoch'] for e in combined]
train_ls = [e['train_loss'] for e in combined]
eval_ls  = [e['eval_loss']   for e in combined]
plt.figure()
plt.plot(epochs, train_ls, marker='o', label='train_loss')
plt.plot(epochs, eval_ls, marker='s', label='eval_loss')
plt.xlabel('Epoch'); plt.ylabel('Loss'); plt.title('Loss Curve')
plt.legend(); plt.grid(True)
plt.savefig(os.path.join(output_dir, 'training_and_validation_loss_curve.png'))
plt.close()
print("Loss curve saved.")
