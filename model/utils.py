import torch
import chess
import json

PIECE_TO_IDX = {
    "P": 0, "N": 1, "B": 2, "R": 3, "Q": 4, "K": 5,
    "p": 6, "n": 7, "b": 8, "r": 9, "q": 10, "k": 11
}

def encode_fen(fen: str) -> torch.Tensor:
    board_tensor = torch.zeros((13, 8, 8), dtype=torch.float32)  # 12 pieces + 1 for turn

    # Parse board
    fen_parts = fen.split()
    rows = fen_parts[0].split("/")
    turn = fen_parts[1] 

    for row_idx, row in enumerate(rows):
        col_idx = 0
        for char in row:
            if char.isdigit():
                col_idx += int(char)
            else:
                idx = PIECE_TO_IDX[char]
                board_tensor[idx, row_idx, col_idx] = 1
                col_idx += 1


    board_tensor[12] = 1.0 if turn == 'w' else 0.0

    return board_tensor



def generate_move_vocab():
    move_to_idx = {}
    idx_to_move = {}

    idx = 0
    squares = [chess.square(file, rank) for file in range(8) for rank in range(8)]

    for from_sq in squares:
        for to_sq in squares:
            if from_sq == to_sq:
                continue  # skip null moves

            uci = chess.SQUARE_NAMES[from_sq] + chess.SQUARE_NAMES[to_sq]
            move_to_idx[uci] = idx
            idx_to_move[idx] = uci
            idx += 1

            # Handle promotions
            if chess.square_rank(from_sq) == 6 and chess.square_rank(to_sq) == 7:
                for promo in ['q', 'r', 'b', 'n']:
                    promo_uci = uci + promo
                    move_to_idx[promo_uci] = idx
                    idx_to_move[idx] = promo_uci
                    idx += 1
            elif chess.square_rank(from_sq) == 1 and chess.square_rank(to_sq) == 0:
                for promo in ['q', 'r', 'b', 'n']:
                    promo_uci = uci + promo
                    move_to_idx[promo_uci] = idx
                    idx_to_move[idx] = promo_uci
                    idx += 1

    # Manually add castling moves
    for move in ["e1g1", "e1c1", "e8g8", "e8c8"]:
        if move not in move_to_idx:
            move_to_idx[move] = idx
            idx_to_move[idx] = move
            idx += 1

    print(f"Total move classes: {len(move_to_idx)}")
    return move_to_idx, idx_to_move

def save_vocab_json(path="model/move_vocab.json"):
    move_to_idx, idx_to_move = generate_move_vocab()
    with open(path, "w") as f:
        json.dump({"move_to_idx": move_to_idx, "idx_to_move": idx_to_move}, f)

if __name__ == "__main__":
    save_vocab_json()