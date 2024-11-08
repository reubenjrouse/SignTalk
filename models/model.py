import torch
import torch.nn as nn
import torch.nn.functional as F

class AttentionLayer(nn.Module):
    def __init__(self, hidden_size):
        super(AttentionLayer, self).__init__()
        self.attention = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size),  # *2 for bidirectional
            nn.Tanh(),
            nn.Linear(hidden_size, 1)
        )
        
    def forward(self, hidden_states):
        attention_weights = self.attention(hidden_states)
        attention_weights = F.softmax(attention_weights, dim=1)
        attended = torch.sum(attention_weights * hidden_states, dim=1)
        return attended, attention_weights

class SignLanguageModel(nn.Module):
    def __init__(self, input_size=108, hidden_size=256, num_layers=2, num_classes=13, dropout=0.5):
        super(SignLanguageModel, self).__init__()
        
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        # Layer Normalization
        self.layer_norm = nn.LayerNorm(input_size)
        
        # BiLSTM layers
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0
        )
        
        # Attention layer
        self.attention = AttentionLayer(hidden_size)
        
        # Final classifier
        self.classifier = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size),  # *2 for bidirectional
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, num_classes)
        )
        
    def forward(self, x):
        # Layer normalization
        x = self.layer_norm(x)
        
        # BiLSTM
        lstm_out, _ = self.lstm(x)
        
        # Attention
        attended, attention_weights = self.attention(lstm_out)
        
        # Classification
        output = self.classifier(attended)
        
        return output, attention_weights