import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from dataset import ChessMoveDataset 
from model import ChessMoveClassifier 
from utils import encode_fen
import json
import os


# ====== CONFIGURATION ======
BATCH_SIZE = 64
EPOCHS = 10
LEARNING_RATE = 1e-3
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


DATA_PATH = "model/train_cleaned.csv"  
VOCAB_PATH = "model/move_vocab.json"
SAVE_PATH = "model/chess_model_new.pt"

# ====== LOAD VOCABULARY ======
with open(VOCAB_PATH) as f:
    vocab = json.load(f)
    move_to_idx = vocab["move_to_idx"]


# ====== DATASET AND LOADER ======
dataset = ChessMoveDataset(DATA_PATH, move_to_idx)
dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)


# ====== MODEL SETUP ======
model = ChessMoveClassifier(num_classes=len(move_to_idx)).to(DEVICE)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)


invalid_count = 0
for fen, move in dataset.data.values:
    if move not in move_to_idx:
        print(f"Missing move: {move}")
        invalid_count += 1

print(f"Total missing moves: {invalid_count}")


# ====== TRAINING LOOP ======
for epoch in range(EPOCHS):
    model.train()
    running_loss = 0.0 
    for batch_idx, (x, y) in enumerate(dataloader): 
        x, y = x.to(DEVICE), y.to(DEVICE) 



        optimizer.zero_grad()  
        outputs = model(x) 
        loss = criterion(outputs, y)  
        loss.backward() 
        optimizer.step() 

        running_loss += loss.item() 

    avg_loss = running_loss / len(dataloader) 
    print(f"Epoch [{epoch+1}/{EPOCHS}], Loss: {avg_loss:.4f}") 

# ====== SAVE MODEL ======
torch.save(model.state_dict(), SAVE_PATH)
print(f"Model saved to {SAVE_PATH}")
