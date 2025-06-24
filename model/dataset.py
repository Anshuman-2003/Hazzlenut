# model/dataset.py
import torch
from torch.utils.data import Dataset
import pandas as pd
from utils import encode_fen

#Class used for data processing
class ChessMoveDataset(Dataset):
    def __init__(self, csv_path, move_to_idx, max_samples=None): #Constuctor innitialized with csv file used and move_to_idx dictionary
        self.data = pd.read_csv(csv_path, names=["fen", "uci"]) #Dataframe consisting two columns fen and corresponding uci
        if max_samples:
            self.data = self.data[:max_samples] #Reduces the dataframe to given limit
        self.move_to_idx = move_to_idx #A dictionary which will be used to map given uci(move) to a single tensor(number)

    def __len__(self): #Fun used by pytorch to get dataframe size
        return len(self.data)

    def __getitem__(self, idx): #for a given idx in dataframe it returns tensor x(input) and y(output) which can be processed by model
        fen = self.data.iloc[idx]["fen"]
        uci = self.data.iloc[idx]["uci"]

        x = encode_fen(fen)                # shape: (12,8,8)
        y = self.move_to_idx.get(uci, -1)  # target class index

        return x, torch.tensor(y)
