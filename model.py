from torch import nn

class NeuralNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        
        self.flatten = nn.Flatten()
        self.network = nn.Sequential(
            nn.Linear(28*28, 15),  
            nn.ReLU(), 
            nn.Linear(15, 10) 
        )

    def forward(self, x):
        x = self.flatten(x)
        logits = self.network(x)
        return logits