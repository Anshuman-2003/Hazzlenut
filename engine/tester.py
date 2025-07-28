from search_utils import get_best_move
import chess

fen = "r3k2r/ppp2ppp/2p2n2/8/4P3/N2P1N1P/bP3PP1/R1BR2K1 w kq - 0 12"
best_move = get_best_move(fen)
print("Suggested Move:", best_move)