import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
from tqdm import tqdm

class TextDataset(Dataset):
    def __init__(self, filename):
        with open(filename, 'r') as f:
            self.data = [line.strip().split() for line in f]

        self.vocabulary = "0123456789abcd"
        self.char2idx = {char: idx for idx, char in enumerate(self.vocabulary)}
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        text, label = self.data[idx]
        # use ord(c) for simple embedding
        x = torch.tensor([self.char2idx[c] for c in text], dtype=torch.long)
        y = torch.tensor(int(label), dtype=torch.float)
        return x, y

def collate_fn(batch):
    # pad sequences to same length
    sequences, labels = zip(*batch)
    lengths = torch.tensor([len(seq) for seq in sequences])
    padded_seqs = nn.utils.rnn.pad_sequence(sequences, batch_first=True)
    return padded_seqs, torch.stack(labels), lengths

########################################
# I'm going off of Colah's blog because 
# it has helpful pictures 
# https://colah.github.io/posts/2015-08-Understanding-LSTMs/
########################################

class LSTMCell(nn.Module):
    def __init__(self, input_size, hidden_size):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        concat_size = input_size + hidden_size
        
        # forget gate
        self.ff_f = nn.Linear(concat_size, hidden_size)
        # input gate
        self.ff_i = nn.Linear(concat_size, hidden_size)
        # cell gate
        self.ff_c = nn.Linear(concat_size, hidden_size)
        # output gate
        self.ff_o = nn.Linear(concat_size, hidden_size)
    
    def forward(self, x, cell_state, hidden):
        gate_inputs = torch.concat((x, hidden), dim=1)

        f = torch.sigmoid(self.ff_f(gate_inputs))
        i = torch.sigmoid(self.ff_i(gate_inputs))
        c_hat = torch.tanh(self.ff_c(gate_inputs))
        o = torch.sigmoid(self.ff_o(gate_inputs))
        
        c_next = f * cell_state + i * c_hat
        h_next = o * torch.tanh(c_next)
        
        return c_next, h_next

class LSTMClassifier(nn.Module):
    def __init__(self, input_size, hidden_size):
        super().__init__()
        self.hidden_size = hidden_size
        self.embedding = nn.Embedding(14, input_size)
        self.lstm_cell = LSTMCell(input_size, hidden_size)
        self.fc = nn.Linear(hidden_size, 1)
    
    def forward(self, x, lengths):
        batch_size = x.size(0)
        seq_len = x.size(1)

        # get embeddings
        x = self.embedding(x.squeeze())
        
        # one initialization for each layer
        c = torch.zeros(batch_size, self.hidden_size).to(x.device)
        h = torch.zeros(batch_size, self.hidden_size).to(x.device)
        
        hidden_outputs = [h]
        for t in range(seq_len):
            x_t = x[:, t, :] # the whole batch, sequence item t, the whole input vector at t
            c, h = self.lstm_cell(
                x_t,
                c,
                hidden_outputs[-1]
            )
            hidden_outputs.append(h)
        
        # get last hidden state before padding
        last_hidden = torch.zeros(batch_size, self.hidden_size)
        for i in range(batch_size):
            last_hidden[i] = hidden_outputs[lengths[i]][i]
        
        out = torch.sigmoid(self.fc(last_hidden))
        return out

def train():
    input_size = 1
    hidden_size = 32
    batch_size = 32
    num_epochs = 5
    learning_rate = 0.01
    
    train_dataset = TextDataset('train.txt')
    test_dataset = TextDataset('test.txt')
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, 
                             shuffle=True, collate_fn=collate_fn)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, 
                            shuffle=False, collate_fn=collate_fn)
    
    # I'm trying to keep this runable on colab
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = LSTMClassifier(input_size, hidden_size).to(device)
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    # begin train loop
    for epoch in range(num_epochs):
        print(f"epoch {epoch}:")
        model.train()
        for batch in tqdm(train_loader):
            x, y, lengths = batch
            x, y = x.unsqueeze(-1).to(device), y.to(device)
            
            optimizer.zero_grad()
            outputs = model(x, lengths).squeeze()
            loss = criterion(outputs, y)
            loss.backward()
            optimizer.step()
        print("end of epoch loss:", loss.item())
        
        # val
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for batch in test_loader:
                x, y, lengths = batch
                x, y = x.unsqueeze(-1).to(device), y.to(device)
                outputs = model(x, lengths)
                predicted = outputs.squeeze() > 0.5
                total += y.size(0)
                correct += (predicted == y).sum().item()
        
        print(f'Epoch {epoch+1}, Accuracy: {100 * correct / total:.2f}%')

if __name__ == '__main__':
    train()