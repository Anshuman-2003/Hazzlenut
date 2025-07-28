import chess
from search_utils import get_best_move

def play_game():
    board = chess.Board()
    print("Initial Position:")
    print(board)

    # Choose player color
    while True:
        player_color = input("Do you want to play as white or black? (w/b): ").strip().lower()
        if player_color in ["w", "b"]:
            break
        print("Invalid input. Enter 'w' or 'b'.")

    is_player_white = player_color == "w"

    while not board.is_game_over():
        print("\nCurrent position (FEN):", board.fen())
        print(board)

        if (board.turn == chess.WHITE and is_player_white) or (board.turn == chess.BLACK and not is_player_white):
            while True:
                try:
                    user_move = input("Your move (in UCI format, e.g., e2e4): ").strip()
                    move = chess.Move.from_uci(user_move)
                    if move in board.legal_moves:
                        board.push(move)
                        break
                    else:
                        print("Illegal move. Try again.")
                except:
                    print("Invalid move format. Try again.")
        else:
            fen = board.fen()
            best_move_uci = get_best_move(fen)
            print("AI plays:", best_move_uci)
            move = chess.Move.from_uci(best_move_uci)
            if move in board.legal_moves:
                board.push(move)
            else:
                print("AI suggested illegal move! Exiting.")
                break

    print("\nFinal position:")
    print(board)
    print("Game Over:", board.result(), "-", board.outcome().termination.name)

if __name__ == "__main__":
    play_game()
