import chess
import torch
import json
from model import ChessMoveClassifier
from utils import encode_fen

# ====== Load Model & Vocab ======
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

# --- Helpers ---
def piece_value(ptype):
    return {
        chess.PAWN: 100,
        chess.KNIGHT: 320,
        chess.BISHOP: 330,
        chess.ROOK: 500,
        chess.QUEEN: 900,
        chess.KING: 20000
    }.get(ptype, 0)

def score_move(board, move):
    score = 0
    moved = board.piece_at(move.from_square)
    captured = board.piece_at(move.to_square)
    if captured:
        score += 10 * piece_value(captured.piece_type)
        if moved:
            score -= piece_value(moved.piece_type)
    if move.promotion:
        score += piece_value(move.promotion)
    return score

# --- Predict Top N Moves ---
def get_model_top_moves(fen, top_k=5):
    input_tensor = encode_fen(fen).unsqueeze(0).to(DEVICE)
    board = chess.Board(fen)
    legal_moves = {move.uci() for move in board.legal_moves}
    with torch.no_grad():
        output = model(input_tensor)
        probs = torch.softmax(output, dim=1)
        sorted_indices = torch.argsort(probs, dim=1, descending=True)[0].tolist()

    legal_predictions = []
    for idx in sorted_indices:
        move = inv_move_vocab.get(idx)
        if move in legal_moves:
            legal_predictions.append((move, probs[0][idx].item()))
        if len(legal_predictions) >= top_k:
            break
    return legal_predictions

# --- Evaluation Function ---
def evaluate_board(board):
    if board.is_checkmate():
        return -9999 if board.turn else 9999
    if board.is_stalemate() or board.is_insufficient_material():
        return 0
    score = 0
    for ptype in range(1, 7):
        score += len(board.pieces(ptype, chess.WHITE)) * piece_value(ptype)
        score -= len(board.pieces(ptype, chess.BLACK)) * piece_value(ptype)
    return score if board.turn == chess.WHITE else -score

# --- Quiescence Search ---
def quiescence(board, alpha, beta):
    stand_pat = evaluate_board(board)
    if stand_pat >= beta:
        return beta
    if alpha < stand_pat:
        alpha = stand_pat

    for move in board.legal_moves:
        if board.is_capture(move):
            board.push(move)
            score = -quiescence(board, -beta, -alpha)
            board.pop()
            if score >= beta:
                return beta
            if score > alpha:
                alpha = score
    return alpha

# --- Move Selection Logic ---
def capture_moves_only(board):
    return [m for m in board.legal_moves if board.is_capture(m)]

def has_good_capture(board):
    for move in board.legal_moves:
        if board.is_capture(move):
            return True  # Now includes all captures, including pawn captures
    return False

def get_filtered_moves(board):
    if board.is_check():
        return list(board.legal_moves)
    captures = capture_moves_only(board)
    if captures:
        # Accept all captures, including pawn captures
        return sorted(captures, key=lambda m: score_move(board, m), reverse=True)
    return [chess.Move.from_uci(mv) for mv, _ in get_model_top_moves(board.fen(), top_k=5)]

# --- Minimax with Alpha-Beta ---
def minimax(board, depth, alpha, beta, is_maximizing):
    if depth == 0 or board.is_game_over():
        return quiescence(board, alpha, beta), None

    moves = get_filtered_moves(board)
    best_move = None

    if is_maximizing:
        max_eval = -float('inf')
        for move in moves:
            board.push(move)
            eval, _ = minimax(board, depth - 1, alpha, beta, False)
            board.pop()
            if eval > max_eval:
                max_eval = eval
                best_move = move
            alpha = max(alpha, eval)
            if beta <= alpha:
                break
        return max_eval, best_move
    else:
        min_eval = float('inf')
        for move in moves:
            board.push(move)
            eval, _ = minimax(board, depth - 1, alpha, beta, True)
            board.pop()
            if eval < min_eval:
                min_eval = eval
                best_move = move
            beta = min(beta, eval)
            if beta <= alpha:
                break
        return min_eval, best_move

# --- Main Game Loop ---
def play():
    board = chess.Board()
    while not board.is_game_over():
        print(board)
        print("FEN:", board.fen())
        if board.turn == chess.WHITE:
            move_input = input("Your move (e.g. e2e4): ")
            try:
                board.push_uci(move_input)
            except:
                print("Invalid move. Try again.")
                continue
        else:
            depth = 7 if not board.is_check() else 8
            print(f"Thinking... depth {depth}")
            _, move = minimax(board, depth, -float('inf'), float('inf'), True)
            print(f"Bot plays: {move}")
            board.push(move)

    print("\nGame Over!")
    print(board.result())

if __name__ == "__main__":
    play()