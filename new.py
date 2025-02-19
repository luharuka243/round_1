import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torchtext.data.utils import get_tokenizer
import math
from tqdm import tqdm
import collections

# Device configuration
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class CrimeDataset(Dataset):
    def __init__(self, features, labels):
        self.features = features
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, item):
        return self.features[item], self.labels[item]

def load_and_preprocess_data(train_path, test_path, max_seq_len=100, vocab_size=15000):
    # Read CSV files
    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)
    
    # Get unique categories and create label mapping
    categories = sorted(train_df['category'].unique())
    category_to_id = {cat: idx for idx, cat in enumerate(categories)}
    
    # Convert labels to tensor
    y_train = torch.tensor([category_to_id[cat] for cat in train_df['category']])
    y_test = torch.tensor([category_to_id[cat] for cat in test_df['category']])
    
    # Initialize tokenizer
    tokenizer = get_tokenizer('basic_english')
    
    # Tokenize crime info
    x_train_texts = [tokenizer(text.lower())[:max_seq_len] 
                     for text in train_df['crime_info']]
    x_test_texts = [tokenizer(text.lower())[:max_seq_len] 
                    for text in test_df['crime_info']]
    
    # Build vocabulary
    counter = collections.Counter()
    for text in x_train_texts:
        counter.update(text)
    
    most_common_words = np.array(counter.most_common(vocab_size - 2))
    vocab = most_common_words[:,0]
    
    # Special tokens
    PAD = 0
    UNK = 1
    word_to_id = {vocab[i]: i + 2 for i in range(len(vocab))}
    
    # Convert words to integers
    x_train = [torch.tensor([word_to_id.get(word, UNK) for word in text])
               for text in x_train_texts]
    x_test = [torch.tensor([word_to_id.get(word, UNK) for word in text])
              for text in x_test_texts]
    
    # Pad test sequences
    x_test = torch.nn.utils.rnn.pad_sequence(x_test, 
                                           batch_first=True, 
                                           padding_value=PAD)
    
    return (CrimeDataset(x_train, y_train), 
            CrimeDataset(x_test, y_test), 
            vocab_size, 
            categories,
            PAD)

class MultiHeadedAttention(nn.Module):
    def __init__(self, h, d_embed, dropout=0.0):
        super(MultiHeadedAttention, self).__init__()
        assert d_embed % h == 0
        self.d_k = d_embed//h
        self.d_embed = d_embed
        self.h = h
        self.WQ = nn.Linear(d_embed, d_embed)
        self.WK = nn.Linear(d_embed, d_embed)
        self.WV = nn.Linear(d_embed, d_embed)
        self.linear = nn.Linear(d_embed, d_embed)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x_query, x_key, x_value, mask=None):
        nbatch = x_query.size(0)
        query = self.WQ(x_query).view(nbatch, -1, self.h, self.d_k).transpose(1,2)
        key = self.WK(x_key).view(nbatch, -1, self.h, self.d_k).transpose(1,2)
        value = self.WV(x_value).view(nbatch, -1, self.h, self.d_k).transpose(1,2)
        
        scores = torch.matmul(query, key.transpose(-2, -1))/math.sqrt(self.d_k)
        if mask is not None:
            scores = scores.masked_fill(mask, float('-inf'))
        
        p_atten = torch.nn.functional.softmax(scores, dim=-1)
        p_atten = self.dropout(p_atten)
        x = torch.matmul(p_atten, value)
        x = x.transpose(1, 2).contiguous().view(nbatch, -1, self.d_embed)
        return self.linear(x)

class ResidualConnection(nn.Module):
    def __init__(self, dim, dropout):
        super().__init__()
        self.drop = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(dim)

    def forward(self, x, sublayer):
        return x + self.drop(sublayer(self.norm(x)))

class EncoderBlock(nn.Module):
    def __init__(self, config):
        super(EncoderBlock, self).__init__()
        self.atten = MultiHeadedAttention(config.h, config.d_embed, config.dropout)
        self.feed_forward = nn.Sequential(
            nn.Linear(config.d_embed, config.d_ff),
            nn.ReLU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.d_ff, config.d_embed)
        )
        self.residual1 = ResidualConnection(config.d_embed, config.dropout)
        self.residual2 = ResidualConnection(config.d_embed, config.dropout)

    def forward(self, x, mask=None):
        x = self.residual1(x, lambda x: self.atten(x, x, x, mask=mask))
        return self.residual2(x, self.feed_forward)

class Encoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.d_embed = config.d_embed
        self.tok_embed = nn.Embedding(config.encoder_vocab_size, config.d_embed)
        self.pos_embed = nn.Parameter(torch.zeros(1, config.max_seq_len, config.d_embed))
        self.encoder_blocks = nn.ModuleList([EncoderBlock(config) for _ in range(config.N_encoder)])
        self.dropout = nn.Dropout(config.dropout)
        self.norm = nn.LayerNorm(config.d_embed)

    def forward(self, input, mask=None):
        x = self.tok_embed(input)
        x_pos = self.pos_embed[:, :x.size(1), :]
        x = self.dropout(x + x_pos)
        for layer in self.encoder_blocks:
            x = layer(x, mask)
        return self.norm(x)

class CrimeTransformer(nn.Module):
    def __init__(self, config, num_classes):
        super().__init__()
        self.encoder = Encoder(config)
        self.linear = nn.Linear(config.d_embed, num_classes)

    def forward(self, x, pad_mask=None):
        x = self.encoder(x, pad_mask)
        return self.linear(torch.mean(x, -2))

def train_model(model, train_loader, test_loader, epochs, optimizer, loss_fn, pad_token):
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        correct = 0
        total = 0
        
        pbar = tqdm(enumerate(train_loader), total=len(train_loader))
        for idx, (x, y) in pbar:
            optimizer.zero_grad()
            
            x = x.to(DEVICE)
            y = y.to(DEVICE)
            
            pad_mask = (x == pad_token).unsqueeze(1).unsqueeze(2)
            outputs = model(x, pad_mask)
            
            loss = loss_fn(outputs, y)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            _, predicted = outputs.max(1)
            total += y.size(0)
            correct += predicted.eq(y).sum().item()
            
            if idx > 0 and idx % 50 == 0:
                pbar.set_description(
                    f'Epoch [{epoch+1}/{epochs}] Loss: {total_loss/total:.4f} '
                    f'Acc: {100.*correct/total:.2f}%')
        
        # Validation
        model.eval()
        val_correct = 0
        val_total = 0
        with torch.no_grad():
            for x, y in test_loader:
                x = x.to(DEVICE)
                y = y.to(DEVICE)
                pad_mask = (x == pad_token).unsqueeze(1).unsqueeze(2)
                outputs = model(x, pad_mask)
                _, predicted = outputs.max(1)
                val_total += y.size(0)
                val_correct += predicted.eq(y).sum().item()
        
        print(f'Validation Accuracy: {100.*val_correct/val_total:.2f}%')

# Example usage:
def main():
    # Load and preprocess data
    train_dataset, test_dataset, vocab_size, categories, PAD = load_and_preprocess_data(
        'train.csv', 'test.csv')
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True,
                            collate_fn=lambda b: pad_sequence(b, PAD))
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False,
                            collate_fn=lambda b: pad_sequence(b, PAD))
    
    # Model configuration
    class Config:
        def __init__(self):
            self.encoder_vocab_size = vocab_size
            self.d_embed = 64
            self.d_ff = 256
            self.h = 4
            self.N_encoder = 3
            self.max_seq_len = 100
            self.dropout = 0.1
    
    config = Config()
    model = CrimeTransformer(config, len(categories)).to(DEVICE)
    
    # Training setup
    optimizer = torch.optim.Adam(model.parameters())
    loss_fn = nn.CrossEntropyLoss()
    
    # Train the model
    train_model(model, train_loader, test_loader, epochs=5, 
                optimizer=optimizer, loss_fn=loss_fn, pad_token=PAD)

def pad_sequence(batch, pad_token):
    texts = [item[0] for item in batch]
    labels = torch.stack([item[1] for item in batch])
    padded_texts = torch.nn.utils.rnn.pad_sequence(texts, 
                                                 batch_first=True, 
                                                 padding_value=pad_token)
    return padded_texts, labels

if __name__ == "__main__":
    main()
