import csv
import chess
import torch
import json
from model import ChessMoveClassifier
from utils import encode_fen
from stockfish import Stockfish

# ==== Configs ====
CSV_PATH = "data/processed/oldDS/random_fens_1000.csv"  # Your test CSV
MODEL_PATH = "model/chess_model_new.pt"
VOCAB_PATH = "model/move_vocab.json"
STOCKFISH_PATH = "/opt/homebrew/bin/stockfish"  # Adjust if needed

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
stockfish = Stockfish(path=STOCKFISH_PATH, parameters={"Threads": 2, "Minimum Thinking Time": 30})


# ==== Load Model & Vocab ====
with open(VOCAB_PATH) as f:
    vocab = json.load(f)
    move_vocab = vocab["move_to_idx"]
    inv_move_vocab = {v: k for k, v in move_vocab.items()}

model = ChessMoveClassifier(num_classes=len(move_vocab)).to(DEVICE)
model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
model.eval()


# ==== Get Top Prediction ====
def get_top_prediction(fen):
    input_tensor = encode_fen(fen).unsqueeze(0).to(DEVICE)
    board = chess.Board(fen)
    legal_moves = {move.uci() for move in board.legal_moves}
    with torch.no_grad():
        output = model(input_tensor)
        probs = torch.softmax(output, dim=1)
        sorted_indices = torch.argsort(probs, dim=1, descending=True)[0].tolist()

    for idx in sorted_indices:
        move = inv_move_vocab.get(idx)
        if move in legal_moves:
            return move
    return None


# ==== Evaluation Script ====
def evaluate_model_on_fens(csv_path):
    total = 0
    valid = 0
    eval_drops = []

    with open(csv_path, 'r') as f:
        reader = csv.reader(f)
        for row in reader:
            if not row: continue
            fen = row[0].strip()
            board = chess.Board(fen)

            # Get model move
            predicted_move_uci = get_top_prediction(fen)
            if not predicted_move_uci:
                continue

            # Eval before
            stockfish.set_fen_position(fen)
            eval_before = stockfish.get_evaluation()
            if eval_before['type'] != 'cp':
                continue  # Skip mate/stalemate etc.

            # Apply model move
            try:
                board.push_uci(predicted_move_uci)
            except:
                continue

            # Eval after
            stockfish.set_fen_position(board.fen())
            eval_after = stockfish.get_evaluation()
            if eval_after['type'] != 'cp':
                continue

            drop = eval_before['value'] - eval_after['value']
            eval_drops.append(drop)
            valid += 1

            # === 🧠 Logging per move ===
            print("\n---------------------------")
            print(f"FEN: {fen}")
            print(f"Model move: {predicted_move_uci}")
            print(f"Stockfish eval before: {eval_before['value']} cp")
            print(f"Stockfish eval after : {eval_after['value']} cp")
            print(f"Eval drop: {drop} cp")

            if total % 100 == 0:
                print(f"\nProcessed: {total} positions")
            total += 1


    # ==== Metrics ====
    if not eval_drops:
        print("No valid evaluations collected.")
        return

    avg_drop = sum(eval_drops) / len(eval_drops)
    within_50 = sum(1 for d in eval_drops if d <= 50) / len(eval_drops) * 100
    within_100 = sum(1 for d in eval_drops if d <= 100) / len(eval_drops) * 100

    print("\n========== Model Evaluation Summary ==========")
    print(f"Total Positions Evaluated: {valid}")
    print(f"Average Eval Drop: {avg_drop:.2f} centipawns")
    print(f"Moves within 50 cp drop: {within_50:.2f}%")
    print(f"Moves within 100 cp drop: {within_100:.2f}%")

if __name__ == "__main__":
    evaluate_model_on_fens(CSV_PATH)
