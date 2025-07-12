import os
import pickle
import pandas as pd
import matplotlib.pyplot as plt
import csv
import shutil
import numpy as np

from sklearn.model_selection import train_test_split
from datasets import Dataset, ClassLabel
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score, precision_recall_f1_score

from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    DataCollatorWithPadding,
    TrainerCallback,
    EvalPrediction
)

# ==============================================================================
# 1. 資料載入與分割 (與原始碼相同)
# ==============================================================================
print("1. 正在載入與分割資料集...")
with open('/home/rock/vulExp/vulCodeBERT/vul_BERT_dataframe_cleaned.pkl', 'rb') as f:
    df = pickle.load(f)

train_df, temp_df = train_test_split(
    df, test_size=0.2, stratify=df["label"], random_state=42
)
valid_df, test_df = train_test_split(
    temp_df, test_size=0.5, stratify=temp_df["label"], random_state=42
)

train_df = train_df.reset_index(drop=True)
valid_df = valid_df.reset_index(drop=True)
test_df = test_df.reset_index(drop=True)

print(f"資料集大小 - 訓練集: {len(train_df)}, 驗證集: {len(valid_df)}, 測試集: {len(test_df)}")

# ==============================================================================
# 2. 建立 Hugging Face Datasets (與原始碼相同)
# ==============================================================================
print("2. 正在建立 Hugging Face Datasets...")
label_feature = ClassLabel(names=["normal", "buggy"])
train_ds = Dataset.from_pandas(train_df).cast_column('label', label_feature)
valid_ds = Dataset.from_pandas(valid_df).cast_column('label', label_feature)
test_ds = Dataset.from_pandas(test_df).cast_column('label', label_feature)

# ==============================================================================
# 3. 載入 Tokenizer 與模型 (與原始碼相同)
# ==============================================================================
print("3. 正在載入 Tokenizer 與 CodeBERT 模型...")
model_name = 'microsoft/codebert-base'
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)

# ==============================================================================
# 4. 資料預處理 (與原始碼相同)
# ==============================================================================
print("4. 正在進行資料預處理...")
def preprocess(example):
    tokens = tokenizer(example['content'], truncation=True, max_length=512, padding="max_length")
    return tokens

# 注意：此處不再需要手動加入 'labels'，Trainer 會自動處理
train_ds = train_ds.map(preprocess, batched=True)
valid_ds = valid_ds.map(preprocess, batched=True)
test_ds = test_ds.map(preprocess, batched=True)

data_collator = DataCollatorWithPadding(tokenizer)

# ==============================================================================
# 5. ✨ 全新修改：定義指標計算函數 ✨
#    - 這個函數會被 Trainer 在每次`評估驗證集`時自動呼叫。
# ==============================================================================
def compute_metrics(p: EvalPrediction):
    preds = np.argmax(p.predictions, axis=1)
    labels = p.label_ids
    precision, recall, f1, _ = precision_recall_f1_score(labels, preds, average='weighted', zero_division=0)
    acc = accuracy_score(labels, preds)
    return {
        'accuracy': acc,
        'f1': f1,
        'precision': precision,
        'recall': recall,
    }

# ==============================================================================
# 6. ✨ 全新修改：自定義 Callback 來記錄所有指標 ✨
#    - 這是實現您需求的核心部分。
# ==============================================================================
class ComprehensiveMetricsLogger(TrainerCallback):
    def __init__(self, trainer, train_dataset, test_dataset, metrics_log_path, loss_log_path):
        self.trainer = trainer
        self.train_dataset = train_dataset
        self.test_dataset = test_dataset
        self.metrics_log_path = metrics_log_path
        self.loss_log_path = loss_log_path
        self._is_initialized = False

    def _initialize_logs(self):
        # 建立指標記錄檔的表頭
        with open(self.metrics_log_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["epoch", "split", "accuracy", "f1", "precision", "recall", "support"])
        # 建立 Loss 記錄檔的表頭
        with open(self.loss_log_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["epoch", "train_loss", "eval_loss"])
        self._is_initialized = True

    def on_epoch_end(self, args, state, control, logs=None, **kwargs):
        if not self._is_initialized:
            self._initialize_logs()

        current_epoch = int(state.epoch)
        print(f"\n--- Epoch {current_epoch} 結束，開始全面評估 ---")

        # --- 記錄 Loss ---
        train_loss = state.log_history[-2].get("loss", None) if len(state.log_history) > 1 else None
        eval_loss = logs.get("eval_loss", None)
        with open(self.loss_log_path, "a", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow([current_epoch, train_loss, eval_loss])

        # --- 記錄驗證集指標 (由 Trainer 自動計算) ---
        with open(self.metrics_log_path, "a", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow([
                current_epoch,
                "validation",
                logs.get("eval_accuracy"),
                logs.get("eval_f1"),
                logs.get("eval_precision"),
                logs.get("eval_recall"),
                len(self.trainer.eval_dataset)
            ])
        print(f"Epoch {current_epoch} [Validation] Metrics logged.")

        # --- 手動評估並記錄測試集指標 ---
        test_preds = self.trainer.predict(self.test_dataset)
        test_metrics = compute_metrics(test_preds)
        with open(self.metrics_log_path, "a", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow([
                current_epoch,
                "test",
                test_metrics.get("accuracy"),
                test_metrics.get("f1"),
                test_metrics.get("precision"),
                test_metrics.get("recall"),
                len(self.test_dataset)
            ])
        print(f"Epoch {current_epoch} [Test] Metrics logged.")
        
        # --- 手動評估並記錄訓練集指標 ---
        # 警告：在大型資料集上，這會顯著增加每個 epoch 的時間！
        print("警告：正在評估訓練集指標，這可能會花費一些時間...")
        train_preds = self.trainer.predict(self.train_dataset)
        train_metrics = compute_metrics(train_preds)
        with open(self.metrics_log_path, "a", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow([
                current_epoch,
                "train",
                train_metrics.get("accuracy"),
                train_metrics.get("f1"),
                train_metrics.get("precision"),
                train_metrics.get("recall"),
                len(self.train_dataset)
            ])
        print(f"Epoch {current_epoch} [Train] Metrics logged.")

# ==============================================================================
# 7. ✨ 全新修改：設定訓練參數 ✨
# ==============================================================================
print("7. 正在設定訓練參數...")
output_dir = './codebert_finetuned_final'
if not os.path.exists(output_dir): os.makedirs(output_dir)

training_args = TrainingArguments(
    output_dir=output_dir,
    num_train_epochs=30,  # 訓練 30 個 epochs
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    weight_decay=0.01,
    
    # --- 評估與儲存策略 ---
    evaluation_strategy='epoch',
    logging_strategy='epoch',
    save_strategy='epoch',
    
    # --- 核心設定：只保留最好的模型，並節省空間 ---
    load_best_model_at_end=True,      # 訓練結束時自動載入最佳模型
    metric_for_best_model='f1',       # 使用 'f1' 分數作為判斷最佳模型的標準
    save_total_limit=1,               # 硬碟上最多只保留 1 個 checkpoint
    greater_is_better=True,           # 對於 f1 分數，越高越好
)

# ==============================================================================
# 8. ✨ 全新修改：初始化 Trainer 與 Callback ✨
# ==============================================================================
print("8. 正在初始化 Trainer...")
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_ds,
    eval_dataset=valid_ds,
    tokenizer=tokenizer,
    data_collator=data_collator,
    compute_metrics=compute_metrics,
)

# 實例化我們的 Callback
metrics_log_path = os.path.join(output_dir, "all_epochs_metrics.csv")
loss_log_path = os.path.join(output_dir, "all_epochs_loss.csv")
metrics_callback = ComprehensiveMetricsLogger(trainer, train_ds, test_ds, metrics_log_path, loss_log_path)
trainer.add_callback(metrics_callback)

# ==============================================================================
# 9. 訓練模型
# ==============================================================================
print("\n9. === 開始模型訓練 ===")
trainer.train()
print("=== 模型訓練完成 ===\n")

# ==============================================================================
# 10. ✨ 新增：顯示最佳模型資訊並儲存 ✨
# ==============================================================================
print("10. 正在顯示最佳模型資訊並儲存...")

# 從 trainer state 獲取最佳模型資訊
best_checkpoint = trainer.state.best_model_checkpoint
best_metric_score = trainer.state.best_metric

# 根據 checkpoint 路徑找到對應的 epoch
best_epoch = -1
if best_checkpoint:
    df_metrics_temp = pd.read_csv(metrics_log_path)
    # 從路徑中提取 step 數
    step_str = best_checkpoint.split('-')[-1]
    if step_str.isdigit():
        best_step = int(step_str)
        # 找到對應的 validation log
        best_log_entry = df_metrics_temp[
            (df_metrics_temp['split'] == 'validation') & 
            (np.isclose(df_metrics_temp['f1'], best_metric_score))
        ].sort_values(by='epoch', ascending=False).iloc[0]
        best_epoch = int(best_log_entry['epoch'])

print("\n--- 最佳模型結果 ---")
print(f"最佳模型是在 Epoch: {best_epoch} 找到的。")
print(f"最佳模型的驗證集 F1-Score: {best_metric_score:.4f}")
print(f"最佳模型的 Checkpoint 路徑: {best_checkpoint}")
print("---------------------\n")


# 因為 load_best_model_at_end=True，現在 trainer 內部持有的就是最佳模型
trainer.save_model(output_dir)
print(f"最佳模型 (Epoch {best_epoch}) 已儲存至 {output_dir}")
print(f"所有指標已記錄至 {metrics_log_path}")
print(f"所有 Loss 已記錄至 {loss_log_path}")

# 清理掉訓練過程中殘留的 checkpoint 資料夾
for item in os.listdir(output_dir):
    if item.startswith("checkpoint-"):
        shutil.rmtree(os.path.join(output_dir, item))
print("已清理臨時的 checkpoint 檔案。")


# ==============================================================================
# 11. ✨ 全新修改：從 CSV 讀取數據並進行最終分析與繪圖 ✨
# ==============================================================================
print("\n11. 正在從記錄檔進行最終分析與繪圖...")

# --- 讀取數據 ---
df_loss = pd.read_csv(loss_log_path)
df_metrics = pd.read_csv(metrics_log_path)

print("\n--- Loss 記錄 ---")
print(df_loss.to_string())
print("\n--- 完整指標記錄 ---")
print(df_metrics.to_string())

# --- 繪製 Loss 曲線圖 ---
plt.figure(figsize=(12, 7))
plt.plot(df_loss['epoch'], df_loss['train_loss'], marker='o', label='Training Loss')
plt.plot(df_loss['epoch'], df_loss['eval_loss'], marker='s', label='Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training and Validation Loss Curve')
plt.xticks(np.arange(1, 31, step=1))
plt.legend()
plt.grid(True)
plt.tight_layout()
loss_curve_path = os.path.join(output_dir, "loss_curve.png")
plt.savefig(loss_curve_path)
plt.close()
print(f"Loss 曲線圖已儲存至: {loss_curve_path}")

# --- 繪製主要指標曲線圖 (F1-score, Accuracy) ---
fig, axes = plt.subplots(2, 1, figsize=(14, 12), sharex=True)
for split in ['train', 'validation', 'test']:
    subset = df_metrics[df_metrics['split'] == split]
    axes[0].plot(subset['epoch'], subset['f1'], marker='o', linestyle='--', label=f'{split} F1-Score')
    axes[1].plot(subset['epoch'], subset['accuracy'], marker='s', linestyle=':', label=f'{split} Accuracy')

axes[0].set_ylabel('F1-Score')
axes[0].set_title('F1-Score per Epoch for All Datasets')
axes[0].legend()
axes[0].grid(True)

axes[1].set_ylabel('Accuracy')
axes[1].set_title('Accuracy per Epoch for All Datasets')
axes[1].set_xlabel('Epoch')
axes[1].legend()
axes[1].grid(True)

plt.xticks(np.arange(1, 31, step=1))
plt.tight_layout()
metrics_curve_path = os.path.join(output_dir, "metrics_curves.png")
plt.savefig(metrics_curve_path)
plt.close()
print(f"主要指標曲線圖已儲存至: {metrics_curve_path}")


# --- 在測試集上輸出最終混淆矩陣與報告 ---
print("\n--- 對載入的最佳模型進行最終測試集評估 ---")
final_predictions = trainer.predict(test_ds)
final_preds = np.argmax(final_predictions.predictions, axis=1)
final_labels = final_predictions.label_ids

final_cm = confusion_matrix(final_labels, final_preds)
final_report = classification_report(final_labels, final_preds, target_names=["normal", "buggy"])

print("最終混淆矩陣 (Test Set):\n", final_cm)
print("\n最終分類報告 (Test Set):\n", final_report)

print("\n=== 所有流程執行完畢 ===")
