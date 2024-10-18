# mem_gru.py
import torch
import torch.nn as nn

class MemGRU(nn.Module):
    """GRU model class for regression."""
    def __init__(self, encoder, input_size, hidden_dim, num_layers=1, bidirectional=False, dropout=0.0):
        super(MemGRU, self).__init__()
        self.encoder = encoder
        for param in self.encoder.parameters():
            param.requires_grad = False
        
        self.gru = nn.GRU(input_size=input_size, hidden_size=hidden_dim, num_layers=num_layers, 
                          batch_first=True, bidirectional=bidirectional, dropout=dropout if num_layers > 1 else 0.0)
        
        fc_input_features = hidden_dim * 2 if bidirectional else hidden_dim
        self.fc = nn.Linear(fc_input_features, 1)
        
    def forward(self, x):
        if len(x.size()) == 5:
            batch_size, seq_len, C, H, W = x.size()
            x = x.view(batch_size * seq_len, C, H, W)
        else:
            raise ValueError(f"Unexpected input shape: {x.size()}")

        encoded_images = self.encoder(x)
        encoded_images = torch.flatten(encoded_images, start_dim=1)
        encoded_images = encoded_images.view(batch_size, seq_len, -1)
        gru_out, _ = self.gru(encoded_images)
        gru_out = gru_out[:, -1, :]
        output = self.fc(gru_out)
        return output
