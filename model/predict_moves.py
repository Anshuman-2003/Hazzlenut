import torch
import json
import numpy as np
from model import ChessMoveClassifier
from utils import encode_fen

# ====== Load Model & Vocab ======
MODEL_PATH = "model/chess_model_new.pt"
VOCAB_PATH = "model/move_vocab.json"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load vocabulary
with open(VOCAB_PATH) as f:
    vocab = json.load(f)
    move_to_idx = vocab["move_to_idx"]
    idx_to_move = {v: k for k, v in move_to_idx.items()}

# Load model
model = ChessMoveClassifier(num_classes=len(move_to_idx)).to(DEVICE)
model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
model.eval()

def predict_moves(fen, top_k=3):
    """
    Given a FEN string, return top K predicted UCI moves using trained model.
    """
    # Encode board position
    input_tensor = encode_fen(fen).unsqueeze(0).to(DEVICE)  # shape: (1, 13, 8, 8)

    with torch.no_grad():
        output = model(input_tensor)
        probs = torch.softmax(output, dim=1)
        topk = torch.topk(probs, k=top_k)

    top_indices = topk.indices[0].tolist()
    top_moves = [idx_to_move[i] for i in top_indices]
    top_scores = [probs[0][i].item() for i in top_indices]

    return list(zip(top_moves, top_scores))  # [(move1, prob1), (move2, prob2), ...]

# ====== Example Usage ======
if __name__ == "__main__":
    test_fen = "r1bqkbnr/pppppppp/n7/8/4P3/5N2/PPPP1PPP/RNBQKB1R b KQkq - 1 2"
    predictions = predict_moves(test_fen)
    print("Top predicted moves:")
    for move, score in predictions:
        print(f"{move}: {score:.4f}")
