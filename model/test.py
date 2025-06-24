import torch
from torch.utils.data import DataLoader, Subset
import json
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from model import ChessMoveClassifier
from dataset import ChessMoveDataset
import random

# ========= CONFIG =========
TEST_PATH = "model/test_cleaned.csv"
VOCAB_PATH = "model/move_vocab.json"
MODEL_PATH = "model/chess_model_new.pt"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 64
TOP_K = [1, 3, 5]
NUM_WORKERS = 2

# ========= LOAD VOCAB =========
with open(VOCAB_PATH) as f:
    move_to_idx = json.load(f)
    if "move_to_idx" in move_to_idx:
        move_to_idx = move_to_idx["move_to_idx"]
idx_to_move = {v: k for k, v in move_to_idx.items()}

# ========= LOAD & DOWNSAMPLE TEST DATA =========
test_dataset_full = ChessMoveDataset(TEST_PATH, move_to_idx)
total_samples = len(test_dataset_full)
subset_indices = random.sample(range(total_samples), min(200000, total_samples))
test_dataset = Subset(test_dataset_full, subset_indices)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False,
                         num_workers=NUM_WORKERS, pin_memory=True)

# ========= LOAD MODEL =========
model = ChessMoveClassifier(num_classes=len(move_to_idx)).to(DEVICE)
model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
model.eval()

# ========= EVALUATION =========
y_true, y_pred, y_proba = [], [], []

with torch.no_grad():
    for batch_idx, (x, y) in enumerate(test_loader):
        x, y = x.to(DEVICE), y.to(DEVICE)
        outputs = model(x)
        probs = torch.softmax(outputs, dim=1)
        preds = torch.argmax(probs, dim=1)

        y_true.extend(y.cpu().numpy())
        y_pred.extend(preds.cpu().numpy())
        y_proba.extend(probs.cpu().numpy())

        if (batch_idx + 1) % 100 == 0:
            print(f"✅ Processed {(batch_idx + 1) * BATCH_SIZE} samples...")

# ========= METRICS =========
print("\n===== EVALUATION METRICS =====")
print("Accuracy:", accuracy_score(y_true, y_pred))

print("\nTop-K Accuracies:")
y_true_np = np.array(y_true)
y_proba_np = np.array(y_proba)

for k in TOP_K:
    topk_preds = np.argsort(y_proba_np, axis=1)[:, -k:]
    match_array = [true in topk for true, topk in zip(y_true_np, topk_preds)]
    topk_acc = np.mean(match_array)
    print(f"Top-{k} Accuracy: {topk_acc:.4f}")

# ========= SAMPLE REPORT =========
print("\n📊 Sampling 10,000 examples for classification report and confusion matrix...")
sample_size = min(10000, len(y_true))
indices = random.sample(range(len(y_true)), sample_size)
sample_y_true = [y_true[i] for i in indices]
sample_y_pred = [y_pred[i] for i in indices]

print("\nClassification Report (sampled):")
print(classification_report(
    sample_y_true, sample_y_pred,
    labels=list(range(len(move_to_idx))),
    target_names=[idx_to_move.get(i, str(i)) for i in range(len(move_to_idx))],
    zero_division=0
))

# ========= CONFUSION MATRIX =========
print("\n📉 Plotting Confusion Matrix on 10,000-sample subset...")
cm = confusion_matrix(sample_y_true, sample_y_pred)

plt.figure(figsize=(10, 8))
sns.heatmap(cm, cmap="Blues", xticklabels=False, yticklabels=False)
plt.title("Confusion Matrix (Sampled 10k)")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.tight_layout()
plt.savefig("confusion_matrix_sampled.png")
plt.show()

print("\n✅ Evaluation complete. Confusion matrix saved as 'confusion_matrix_sampled.png'")
