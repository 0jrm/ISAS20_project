import torch.nn as nn
import torch.nn.functional as F
from base import BaseModel


class FFNN(BaseModel):
    def __init__(self, input_dim=1, layers_config=[512, 256], output_dim=30, dropout_prob = 0.5, activation = nn.ReLU()):
        super(FFNN, self).__init__()
        
        # Construct layers based on the given configuration
        layers = []
        prev_dim = input_dim
        for neurons in layers_config:
            layers.append(nn.Linear(prev_dim, neurons))
            layers.append(activation) # can be changed to an array, just as layers_config
            if dropout_prob > 0:
                layers.append(nn.Dropout(dropout_prob)) # added dropout
            prev_dim = neurons
        layers.append(nn.Linear(prev_dim, output_dim))
        
        self.model = nn.Sequential(*layers)
    
        def forward(self, x):
            return self.model(x)
    