import zstandard as zstd

def make_pgn(input_path, output_path):
    with open(input_path, 'rb') as compressed, open(output_path, 'wb') as out:
        dctx = zstd.ZstdDecompressor()
        dctx.copy_stream(compressed, out)

if __name__ == "__main__":
    make_pgn("data/raw/lichess_db_broadcast_2025-06.pgn.zst", "data/raw/data.pgn")