"""
-- Created by: Ashok Kumar Pant
-- Email: asokpant@gmail.com
-- Created on: 03/04/2025
"""

import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm


class Vocabulary:
    """Handles text processing and vocabulary mapping."""

    def __init__(self):
        self.token_to_idx = {"<PAD>": 0}  # Add padding token
        self.idx_to_token = ["<PAD>"]

    def add_token(self, token):
        if token not in self.token_to_idx:
            self.token_to_idx[token] = len(self.idx_to_token)
            self.idx_to_token.append(token)
        return self.token_to_idx[token]

    def __len__(self):
        return len(self.idx_to_token)


class SurnameDataset(Dataset):
    """Custom dataset for name classification."""

    def __init__(self, data_path, max_length=15):
        self.data = pd.read_csv(data_path)
        self.vocab = Vocabulary()
        self.nationality_vocab = Vocabulary()  # LabelEncoder
        self.max_length = max_length
        self._build_vocab()

    def _build_vocab(self):
        for name in self.data['name']:
            for char in name:
                self.vocab.add_token(char)
        for nationality in self.data['nationality']:
            self.nationality_vocab.add_token(nationality)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        name = self.data.iloc[idx]['name']
        nationality = self.data.iloc[idx]['nationality']
        nationality_idx = self.nationality_vocab.token_to_idx[nationality]

        # Convert name to index sequence
        name_indices = [self.vocab.token_to_idx[char] for char in name]

        # Pad or truncate
        name_indices = name_indices[:self.max_length]  # Truncate
        name_indices += [0] * (self.max_length - len(name_indices))  # Pad

        name_tensor = torch.tensor(name_indices, dtype=torch.long)

        return name_tensor, torch.tensor(nationality_idx, dtype=torch.long)


class SurnameClassifier(nn.Module):
    """RNN-based classifier."""

    def __init__(self, vocab_size, embed_dim, hidden_dim, output_dim):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)  # Use padding_idx
        self.rnn = nn.RNN(embed_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        embedded = self.embedding(x)
        _, hidden = self.rnn(embedded)
        return self.fc(hidden[-1])  # Use last hidden state


class SurnameTrainer:
    """Handles model training and evaluation."""

    def __init__(self, model, dataloader, device, epochs=10, lr=0.001, model_path="model.pth"):
        self.model = model.to(device)
        self.dataloader = dataloader
        self.device = device
        self.epochs = epochs
        self.loss = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(model.parameters(), lr=lr)
        self.model_path = model_path

    def train(self):
        self.model.train()
        best_loss = float('inf')
        patience, counter = 20, 0
        for epoch in range(self.epochs):
            epoch_loss = 0
            for inputs, labels in tqdm(self.dataloader, desc=f"Epoch {epoch + 1}"):
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                self.optimizer.zero_grad()
                outputs = self.model(inputs)
                loss = self.loss(outputs, labels)
                loss.backward()
                self.optimizer.step()
                loss_value = loss.item()
                epoch_loss += loss_value
            print(f"Epoch {epoch + 1}, Loss: {epoch_loss / len(self.dataloader):.4f}")

            if epoch_loss < best_loss:
                best_loss = epoch_loss
                counter = 0
            else:
                counter += 1
                if counter >= patience:
                    print("Early stopping")
                    break
        self.save_model()

    def save_model(self):
        torch.save(self.model.state_dict(), self.model_path)
        print(f"Model saved to {self.model_path}")

    def load_model(self):
        self.model.load_state_dict(torch.load(self.model_path))
        self.model.eval()
        print(f"Model loaded from {self.model_path}")


class SurnameInference:
    """Handles model inference."""

    def __init__(self, model, vocab, max_length=15, model_path="model.pth", device="cpu"):
        self.model = model.to(device)
        self.vocab = vocab
        self.max_length = max_length
        self.device = device
        self.model_path = model_path
        self.load_model()

    def load_model(self):
        self.model.load_state_dict(torch.load(self.model_path, map_location=self.device))
        self.model.eval()
        print(f"Model loaded for inference from {self.model_path}")

    def predict(self, name):
        name_indices = [self.vocab.token_to_idx.get(char, 0) for char in name]
        name_indices = name_indices[:self.max_length]  # Truncate
        name_indices += [0] * (self.max_length - len(name_indices))  # Pad

        name_tensor = torch.tensor([name_indices], dtype=torch.long).to(self.device)
        with torch.no_grad():
            output = self.model(name_tensor)
        return torch.argmax(output, dim=1).item()


def training_example():
    data_path = './data/surnames-by-nationality.csv'  # Update with actual path
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dataset = SurnameDataset(data_path)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

    model = SurnameClassifier(len(dataset.vocab), embed_dim=32, hidden_dim=32,
                              output_dim=len(dataset.nationality_vocab))
    trainer = SurnameTrainer(model, dataloader, device, epochs=100, lr=0.001, model_path="./models/surname-model.bin")
    trainer.train()


def inference_example():
    data_path = './data/surnames-by-nationality.csv'  # Update with actual path
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dataset = SurnameDataset(data_path)
    model = SurnameClassifier(len(dataset.vocab), embed_dim=32, hidden_dim=32,
                              output_dim=len(dataset.nationality_vocab))
    inference = SurnameInference(model, dataset.vocab, model_path="./models/surname-model.bin", device=device)
    for name in ['McMahan', 'Nakamoto', 'Wan', 'Cho', "Pant", "aayush", "Ansan"]:
        prediction = inference.predict(name)
        prediction_label = dataset.nationality_vocab.idx_to_token[prediction]
        print(f"Predicted class for '{name}': {prediction_label}")


if __name__ == '__main__':
    training_example()
    # inference_example()
