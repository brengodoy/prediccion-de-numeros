from torch import nn

class RedNeuronal(nn.Module):
    def __init__(self):
        super().__init__()
        
        self.aplanar = nn.Flatten()
        self.red = nn.Sequential(
            nn.Linear(28*28, 15),  
            nn.ReLU(), 
            nn.Linear(15, 10) 
        )

    def forward(self, x):
        x = self.aplanar(x)
        logits = self.red(x)
        return logits
