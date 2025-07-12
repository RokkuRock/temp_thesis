import os
import pickle
import pandas as pd
import matplotlib.pyplot as plt
import csv
import shutil
import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from sklearn.model_selection import train_test_split
from datasets import Dataset
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    DataCollatorWithPadding,
    EvalPrediction,
    TrainerCallback
)

# ==============================================================================
# ✨ 通用函數與設定 ✨
# ==============================================================================

def get_project_path(sub_path: str) -> str:
    """取得相對於專案根目錄的絕對路徑"""
    return os.path.join(os.path.dirname(__file__), sub_path)

def compute_all_metrics(p: EvalPrediction):
    """計算所有我們需要的指標 (P, R, F1, Acc, Support)"""
    preds = np.argmax(p.predictions, axis=1)
    labels = p.label_ids
    precision = precision_score(labels, preds, average='weighted', zero_division=0)
    recall = recall_score(labels, preds, average='weighted', zero_division=0)
    f1 = f1_score(labels, preds, average='weighted', zero_division=0)
    acc = accuracy_score(labels, preds)
    total_support = len(labels)
    
    return {
        'accuracy': acc,
        'f1': f1,
        'precision': precision,
        'recall': recall,
        'support': total_support
    }

def evaluate_in_chunks(model, tokenizer, data_collator, device, dataset: Dataset, eval_batch_size: int = 64):
    """安全地、分塊評估大型資料集，避免記憶體溢位"""
    print(f"開始分塊評估，資料集大小: {len(dataset)}, 批次大小: {eval_batch_size}")
    
    model_input_names = tokenizer.model_input_names + ["labels"]
    dataset_cleaned = dataset.remove_columns([col for col in dataset.column_names if col not in model_input_names])
    
    dataloader = DataLoader(dataset_cleaned, batch_size=eval_batch_size, collate_fn=data_collator, shuffle=False)
    
    model.eval()
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="分塊評估中"):
            inputs = {k: v.to(device) for k, v in batch.items() if k != 'labels'}
            labels = batch['labels']
            outputs = model(**inputs)
            preds = torch.argmax(outputs.logits, dim=-1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.numpy())
            
    return compute_all_metrics(EvalPrediction(predictions=np.array(all_preds), label_ids=np.array(all_labels)))


# ==============================================================================
# ✨ 自定義 Callback 來記錄所有指標 ✨
# ==============================================================================
class ComprehensiveMetricsLogger(TrainerCallback):
    # ✨ 關鍵修改：初始化時接收所有必要的元件，包括 tokenizer
    def __init__(self, train_dataset, eval_dataset, test_dataset, tokenizer, data_collator, metrics_log_path, loss_log_path):
        self.train_dataset = train_dataset
        self.eval_dataset = eval_dataset
        self.test_dataset = test_dataset
        self.tokenizer = tokenizer
        self.data_collator = data_collator
        self.metrics_log_path = metrics_log_path
        self.loss_log_path = loss_log_path
        self._is_initialized = False

    def _initialize_logs(self):
        with open(self.metrics_log_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["epoch", "split", "accuracy", "f1", "precision", "recall", "support"])
        with open(self.loss_log_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["epoch", "train_loss", "eval_loss"])
        self._is_initialized = True

    def on_epoch_end(self, args, state, control, logs=None, **kwargs):
        if not self._is_initialized:
            self._initialize_logs()

        current_epoch = int(state.epoch)
        # ✨ 關鍵修改：從 kwargs 安全地取得 model，並使用 self.tokenizer
        model = kwargs['model']
        device = args.device
        
        print(f"\n--- Epoch {current_epoch} 訓練結束，開始全面評估 ---")

        # 記錄 Loss
        train_loss_logs = [log for log in state.log_history if 'loss' in log and 'eval_loss' not in log]
        train_loss = train_loss_logs[-1].get("loss", None) if train_loss_logs else None
        eval_loss = logs.get("eval_loss", None)
        with open(self.loss_log_path, "a", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow([current_epoch, train_loss, eval_loss])

        # 記錄驗證集指標 (由 Trainer 自動計算)
        with open(self.metrics_log_path, "a", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow([current_epoch, "validation", logs.get("eval_accuracy"), logs.get("eval_f1"), logs.get("eval_precision"), logs.get("eval_recall"), len(self.eval_dataset)])
        print(f"Epoch {current_epoch} [Validation] Metrics logged.")

        # ✨ 關鍵修改：使用 self.tokenizer 進行評估
        # 記錄測試集指標
        test_metrics = evaluate_in_chunks(model, self.tokenizer, self.data_collator, device, self.test_dataset)
        with open(self.metrics_log_path, "a", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow([current_epoch, "test", test_metrics.get("accuracy"), test_metrics.get("f1"), test_metrics.get("precision"), test_metrics.get("recall"), len(self.test_dataset)])
        print(f"Epoch {current_epoch} [Test] Metrics logged.")
        
        # 記錄訓練集指標
        train_metrics = evaluate_in_chunks(model, self.tokenizer, self.data_collator, device, self.train_dataset)
        with open(self.metrics_log_path, "a", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow([current_epoch, "train", train_metrics.get("accuracy"), train_metrics.get("f1"), train_metrics.get("precision"), train_metrics.get("recall"), len(self.train_dataset)])
        print(f"Epoch {current_epoch} [Train] Metrics logged.")


# ==============================================================================
# ✨ 主執行流程 ✨
# ==============================================================================
def main():
    # --- 1. 資料準備 ---
    print("--- 1. 正在準備資料 ---")
    DATASET_PATH = get_project_path('vul_BERT_dataframe_cleaned.pkl')
    with open(DATASET_PATH, 'rb') as f:
        df = pickle.load(f)

    train_df, temp_df = train_test_split(df, test_size=0.2, stratify=df["label"], random_state=42)
    valid_df, test_df = train_test_split(temp_df, test_size=0.5, stratify=temp_df["label"], random_state=42)
    
    train_ds = Dataset.from_pandas(train_df)
    valid_ds = Dataset.from_pandas(valid_df)
    test_ds = Dataset.from_pandas(test_df)

    model_name = 'microsoft/codebert-base'
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    def preprocess(example):
        return tokenizer(example['content'], truncation=True, max_length=512)

    train_ds = train_ds.map(preprocess, batched=True, remove_columns=['content'])
    valid_ds = valid_ds.map(preprocess, batched=True, remove_columns=['content'])
    test_ds = test_ds.map(preprocess, batched=True, remove_columns=['content'])
    
    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)
    data_collator = DataCollatorWithPadding(tokenizer)

    # --- 2. 訓練參數與 Callback 設定 ---
    print("\n--- 2. 正在設定訓練參數 ---")
    output_dir = get_project_path('training_output')
    if os.path.exists(output_dir):
        print(f"警告：輸出目錄 {output_dir} 已存在，將會覆寫其內容。")
        shutil.rmtree(output_dir)
    os.makedirs(output_dir)

    NUM_EPOCHS = 10

    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=NUM_EPOCHS,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        weight_decay=0.01,
        eval_strategy='epoch',
        logging_strategy='epoch',
        save_strategy='epoch',
        load_best_model_at_end=True,
        metric_for_best_model='f1',
        save_total_limit=1,
        greater_is_better=True,
        dataloader_pin_memory=False,
    )
    
    # ✨ 關鍵修改：初始化 Callback 時傳入所有需要的元件，包括 tokenizer
    metrics_log_path = os.path.join(output_dir, "all_metrics_log.csv")
    loss_log_path = os.path.join(output_dir, "loss_log.csv")
    metrics_callback = ComprehensiveMetricsLogger(
        train_dataset=train_ds, 
        eval_dataset=valid_ds, 
        test_dataset=test_ds, 
        tokenizer=tokenizer, 
        data_collator=data_collator, 
        metrics_log_path=metrics_log_path, 
        loss_log_path=loss_log_path
    )
    
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=valid_ds,
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_all_metrics,
        callbacks=[metrics_callback],
    )

    # --- 3. 開始訓練 ---
    print("\n--- 3. === 開始模型訓練 === ---")
    trainer.train()
    print("\n=== 模型訓練完成 ===\n")
    
    # --- 4. 儲存最佳模型 ---
    print("--- 4. 正在儲存最終的最佳模型 ---")
    trainer.save_model(output_dir)
    print(f"最佳模型已儲存至 {output_dir}")

    # --- 5. 產生報告與圖表 ---
    print("\n--- 5. 正在產生最終報告與圖表 ---")
    df_loss = pd.read_csv(loss_log_path)
    df_metrics = pd.read_csv(metrics_log_path)

    # 圖 1: Training Set 四大指標
    plt.figure(figsize=(12, 7))
    df_train = df_metrics[df_metrics['split'] == 'train']
    plt.plot(df_train['epoch'], df_train['f1'], marker='o', label='F1-Score')
    plt.plot(df_train['epoch'], df_train['accuracy'], marker='s', label='Accuracy')
    plt.plot(df_train['epoch'], df_train['precision'], marker='^', label='Precision')
    plt.plot(df_train['epoch'], df_train['recall'], marker='x', label='Recall')
    plt.title('Training Set Metrics per Epoch')
    plt.xlabel('Epoch')
    plt.ylabel('Score')
    plt.xticks(np.arange(0, NUM_EPOCHS + 1, step=1))
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(output_dir, 'figure_1_train_metrics.png'))
    plt.close()
    print("圖 1 (Training Set 四大指標) 已儲存。")
    
    # 圖 2: Loss Curve
    plt.figure(figsize=(12, 7))
    plt.plot(df_loss['epoch'], df_loss['train_loss'], marker='o', label='Training Loss')
    plt.plot(df_loss['epoch'], df_loss['eval_loss'], marker='s', label='Validation Loss')
    plt.title('Training and Validation Loss Curve')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.xticks(np.arange(0, NUM_EPOCHS + 1, step=1))
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(output_dir, 'figure_2_loss_curve.png'))
    plt.close()
    print("圖 2 (Loss Curve) 已儲存。")
    
    # 圖 3: F1-Score 比較圖
    plt.figure(figsize=(12, 7))
    for split in ['train', 'validation']:
        subset = df_metrics[df_metrics['split'] == split]
        plt.plot(subset['epoch'], subset['f1'], marker='o', linestyle='--', label=f'{split} F1-Score')
    plt.title('F1-Score Comparison (Train vs Validation)')
    plt.xlabel('Epoch')
    plt.ylabel('F1-Score')
    plt.xticks(np.arange(0, NUM_EPOCHS + 1, step=1))
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(output_dir, 'figure_3_f1_comparison.png'))
    plt.close()
    print("圖 3 (F1-Score 比較圖) 已儲存。")
    
    # 表格 4: 最終結果
    best_epoch_row = df_metrics.loc[df_metrics[df_metrics['split'] == 'validation']['f1'].idxmax()]
    best_epoch = int(best_epoch_row['epoch'])
    
    print(f"\n最佳模型是在 Epoch: {best_epoch} 找到的 (Validation F1-Score 最高)。")
    
    best_train_metrics = df_metrics[(df_metrics['epoch'] == best_epoch) & (df_metrics['split'] == 'train')]
    best_test_metrics = df_metrics[(df_metrics['epoch'] == best_epoch) & (df_metrics['split'] == 'test')]
    
    final_table_data = []
    if not best_train_metrics.empty:
        final_table_data.append(best_train_metrics.iloc[0])
    if not best_test_metrics.empty:
        final_table_data.append(best_test_metrics.iloc[0])

    if final_table_data:
        final_table = pd.DataFrame(final_table_data)
        final_table = final_table[['split', 'precision', 'recall', 'f1', 'accuracy', 'support']]
        final_table.rename(columns={'f1': 'f1-score'}, inplace=True)
    
        print("\n--- 最終結果表格 (在最佳 Epoch) ---")
        print(final_table.to_string(index=False))
        final_table.to_csv(os.path.join(output_dir, 'table_4_final_results.csv'), index=False)
        print("\n表格 4 (最終結果) 已儲存。")
    
    print("\n✅ 所有流程執行完畢！")

if __name__ == '__main__':
    main()
