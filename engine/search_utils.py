import chess
import torch
from model_loader import model, move_vocab, inv_move_vocab, DEVICE
from model.utils import encode_fen
from eval_static import evaluate_material


# --- Get Top K Legal Model Moves ---
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


def quiescence(board: chess.Board, depth: int, alpha: int, beta: int) -> int:
    stand_pat = evaluate_material(board)

    if board.is_game_over() or depth == 0:
        return stand_pat

    if stand_pat >= beta:
        return beta
    if alpha < stand_pat:
        alpha = stand_pat

    # Evaluate all captures (and optionally checks or promotions)
    for move in board.generate_legal_captures():
        board.push(move)
        score = -quiescence(board, depth - 1, -beta, -alpha)
        board.pop()

        if score >= beta:
            return beta
        if score > alpha:
            alpha = score

    return alpha

def minimax(board: chess.Board, depth: int, alpha: int, beta: int, maximizing: bool) -> int:
    if depth == 0 or board.is_game_over():
        return evaluate_material(board)

    if maximizing:
        max_eval = -float('inf')
        for move in board.legal_moves:
            board.push(move)
            eval = minimax(board, depth - 1, alpha, beta, False)
            board.pop()
            max_eval = max(max_eval, eval)
            alpha = max(alpha, eval)
            if beta <= alpha:
                break
        return max_eval
    else:
        min_eval = float('inf')
        for move in board.legal_moves:
            board.push(move)
            eval = minimax(board, depth - 1, alpha, beta, True)
            board.pop()
            min_eval = min(min_eval, eval)
            beta = min(beta, eval)
            if beta <= alpha:
                break
        return min_eval


def get_best_move(fen: str, capture_depth: int = 5, fallback_depth: int = 4, fallback_topk: int = 5):
    board = chess.Board(fen)
    best_move = None

    current_score = evaluate_material(board)
    best_score = current_score

    legal_moves = list(board.legal_moves)
    capturing_moves = [m for m in legal_moves if board.is_capture(m)]

    capture_improved = False

    # --- Phase 1: Evaluate Capture Moves using Quiescence ---
    for move in capturing_moves:
        try:
            board.push(move)
            score = -quiescence(board, capture_depth - 1, -float('inf'), float('inf'))
            board.pop()
            if score > best_score:
                best_score = score
                best_move = move
                capture_improved = True
        except:
            if board.is_legal(move):
                board.pop()

    # --- Phase 2: Fallback to model-suggested moves ---
    if not capture_improved:
        print("Switching to model moves")
        fallback_moves = get_model_top_moves(fen, top_k=fallback_topk)
        first_legal_model_move = None

        for move_str, _ in fallback_moves:
            try:
                move = chess.Move.from_uci(move_str)
                if move not in board.legal_moves:
                    continue

                if first_legal_model_move is None:
                    first_legal_model_move = move

                board.push(move)
                score = minimax(board, fallback_depth - 1, -float('inf'), float('inf'), False)
                board.pop()

                if score > best_score:
                    best_score = score
                    best_move = move

            except:
                if move in board.move_stack:
                    board.pop()

        if best_move is None and first_legal_model_move is not None:
            best_move = first_legal_model_move

    return best_move.uci() if best_move else None