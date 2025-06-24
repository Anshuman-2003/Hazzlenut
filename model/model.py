import torch.nn as nn


class ChessMoveClassifier(nn.Module):
    def __init__(self, num_classes): 
        super().__init__()
        self.model = nn.Sequential( 
            nn.Flatten(), 
            nn.Linear(832, 512), 
            nn.ReLU(), 
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, num_classes)
        )

    def forward(self, x): 
        return self.model(x)

