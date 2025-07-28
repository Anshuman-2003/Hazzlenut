import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import torch
import json
from model.model import ChessMoveClassifier

MODEL_PATH = "model/chess_model_new.pt"
VOCAB_PATH = "model/move_vocab.json"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

with open(VOCAB_PATH) as f:
    vocab = json.load(f)
    move_vocab = vocab["move_to_idx"]
    inv_move_vocab = {v: k for k, v in move_vocab.items()}

model = ChessMoveClassifier(num_classes=len(move_vocab)).to(DEVICE)
model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
model.eval()