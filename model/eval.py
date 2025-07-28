import csv
import chess
import torch
import json
import matplotlib.pyplot as plt
import numpy as np
from model import ChessMoveClassifier
from utils import encode_fen
from stockfish import Stockfish

# ==== Configs ====
CSV_PATH = "data/processed/oldDS/random_fens_1000.csv"
MODEL_PATH = "model/chess_model_new.pt"
VOCAB_PATH = "model/move_vocab.json"
STOCKFISH_PATH = "/opt/homebrew/bin/stockfish"
LOG_FILE_PATH = "model_eval_log.txt"
HISTOGRAM_PATH = "eval_drop_histogram.png"

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

    with open(LOG_FILE_PATH, "w") as log_file:
        log_file.write("========== Model Evaluation Log ==========\n")

        with open(csv_path, 'r') as f:
            reader = csv.reader(f)
            for row in reader:
                if not row: continue
                fen = row[0].strip()
                board = chess.Board(fen)

                predicted_move_uci = get_top_prediction(fen)
                if not predicted_move_uci:
                    continue

                stockfish.set_fen_position(fen)
                eval_before = stockfish.get_evaluation()
                if eval_before['type'] != 'cp':
                    continue

                try:
                    board.push_uci(predicted_move_uci)
                except:
                    continue

                stockfish.set_fen_position(board.fen())
                eval_after = stockfish.get_evaluation()
                if eval_after['type'] != 'cp':
                    continue

                drop = eval_before['value'] - eval_after['value']
                eval_drops.append(drop)
                valid += 1

                # Per-move log
                log_file.write("\n---------------------------\n")
                log_file.write(f"FEN: {fen}\n")
                log_file.write(f"Model move: {predicted_move_uci}\n")
                log_file.write(f"Stockfish eval before: {eval_before['value']} cp\n")
                log_file.write(f"Stockfish eval after : {eval_after['value']} cp\n")
                log_file.write(f"Eval drop: {drop} cp\n")

                if total % 100 == 0:
                    print(f"Processed: {total} positions")
                total += 1

        if not eval_drops:
            print("No valid evaluations collected.")
            return

        # === Metrics ===
        avg_drop = np.mean(eval_drops)
        within_50 = np.sum(np.array(eval_drops) <= 50) / len(eval_drops) * 100
        within_100 = np.sum(np.array(eval_drops) <= 100) / len(eval_drops) * 100
        worst_drop = np.max(eval_drops)
        best_drop = np.min(eval_drops)
        std_dev = np.std(eval_drops)

        summary = (
            "\n========== Model Evaluation Summary ==========\n"
            f"Total Positions Evaluated: {valid}\n"
            f"Average Eval Drop: {avg_drop:.2f} centipawns\n"
            f"Moves within 50 cp drop: {within_50:.2f}%\n"
            f"Moves within 100 cp drop: {within_100:.2f}%\n"
            f"Worst Eval Drop: {worst_drop:.2f} cp\n"
            f"Best Eval Drop: {best_drop:.2f} cp\n"
            f"Standard Deviation: {std_dev:.2f} cp\n"
        )

        print(summary)
        log_file.write(summary)

        # === Histogram ===
        plt.figure(figsize=(10, 6))
        plt.hist(eval_drops, bins=40, color="skyblue", edgecolor="black")
        plt.title("Distribution of Evaluation Drop (in centipawns)")
        plt.xlabel("Eval Drop (cp)")
        plt.ylabel("Number of Positions")
        plt.axvline(avg_drop, color='red', linestyle='dashed', linewidth=1.5, label=f"Avg: {avg_drop:.2f} cp")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(HISTOGRAM_PATH)
        print(f"Saved histogram to: {HISTOGRAM_PATH}")
        log_file.write(f"\nSaved histogram to: {HISTOGRAM_PATH}\n")

if __name__ == "__main__":
    evaluate_model_on_fens(CSV_PATH)
