import chess.pgn
import os
from tqdm import tqdm

import chess.pgn
import os
from tqdm import tqdm

def extract_fen_move_pairs(pgn_path, output_path, max_games=10000, min_elo=2000):
    with open(pgn_path) as pgn, open(output_path, "w") as out:
        out.write("fen,move\n")  
        game_count = 0
        total_seen = 0

        while game_count < max_games:
            game = chess.pgn.read_game(pgn)
            if game is None:
                break

            total_seen += 1

            try:
                white_elo = int(game.headers.get("WhiteElo", 0))
                black_elo = int(game.headers.get("BlackElo", 0))
            except ValueError:
                continue  # skip malformed ratings

            if white_elo < min_elo or black_elo < min_elo:
                continue  

            board = game.board()
            for move in game.mainline_moves():
                fen = board.fen()
                board.push(move)
                uci = move.uci()
                out.write(f"{fen},{uci}\n")

            game_count += 1
            if game_count % 100 == 0:
                print(f"Processed {game_count} games (from {total_seen} seen)")

    print(f"\n✅ Finished: {game_count} high-quality games written.")


def extract_all_fen_move_pairs(pgn_path, output_path, min_elo=2000):
    with open(pgn_path) as pgn, open(output_path, "w") as out:
        out.write("fen,move\n")  # Header

        game_count = 0
        total_seen = 0

        while True:
            game = chess.pgn.read_game(pgn)
            if game is None:
                break  # End of file

            total_seen += 1

            try:
                white_elo = int(game.headers.get("WhiteElo", 0))
                black_elo = int(game.headers.get("BlackElo", 0))
            except ValueError:
                continue  # Skip if malformed ELO

            if white_elo < min_elo or black_elo < min_elo:
                continue  # Skip low-rated games

            board = game.board()
            for move in game.mainline_moves():
                fen = board.fen()
                board.push(move)
                uci = move.uci()
                out.write(f"{fen},{uci}\n")

            game_count += 1
            if game_count % 100 == 0:
                print(f"✅ Processed {game_count} high-rated games (from {total_seen} total seen)")

    print(f"\n🎉 Finished! Extracted {game_count} games to {output_path}")

if __name__ == "__main__":
    extract_all_fen_move_pairs(
        pgn_path="data/raw/2600-.pgn",
        output_path="data/processed/data1.csv",
        min_elo=2000
    )