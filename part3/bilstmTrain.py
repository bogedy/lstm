import argparse
import json
from collections import defaultdict

import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset

import time

class LSTMCell(nn.Module):
    def __init__(self, input_size, hidden_size):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        concat_size = input_size + hidden_size
        self.ff_f = nn.Linear(concat_size, hidden_size)
        self.ff_i = nn.Linear(concat_size, hidden_size)
        self.ff_c = nn.Linear(concat_size, hidden_size)
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
    
    def process_sequence(self, sequence):
        """
        process a whole sequence and get back the final hidden
        """
        if sequence.dim() == 2:
            # add batch dimension if not present, useful when testing on overfitting on one sample.
            sequence = sequence.unsqueeze(0)
            squeeze_output = True
        else:
            squeeze_output = False
            
        batch_size, seq_len, _ = sequence.shape
        device = sequence.device
        
        hidden = torch.zeros(batch_size, self.hidden_size, device=device)            
        cell = torch.zeros(batch_size, self.hidden_size, device=device)
        
        outputs = []
        for t in range(seq_len):
            cell, hidden = self.forward(sequence[:, t, :], cell, hidden)
            outputs.append(hidden)
        
        if squeeze_output:
            outputs = [h.squeeze(0) for h in outputs]
            hidden = hidden.squeeze(0)
            cell = cell.squeeze(0)
            
        return outputs, hidden, cell

class BiLSTMTagger(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.repr_type = config['repr_type']
        self.hidden_dim = config['hidden_dim']
        self.embedding_dim = config['embedding_dim']
        self.dropout = nn.Dropout(0.5)
        
        # word embeddings for reprs a, c, d
        if self.repr_type in ['a', 'c', 'd']:
            self.word_embeddings = nn.Embedding(config['vocab_size'], self.embedding_dim, padding_idx=0)
        # character lstm for reprs b, d
        if self.repr_type in ['b', 'd']:
            self.char_embeddings = nn.Embedding(config['char_vocab_size'], config['char_embedding_dim'], padding_idx=0)
            self.char_lstm = LSTMCell(config['char_embedding_dim'], config['char_hidden_dim'])
        # subword embeddings for repr c
        if self.repr_type == 'c':
            self.prefix_embeddings = nn.Embedding(config['prefix_vocab_size'], self.embedding_dim, padding_idx=0)
            self.suffix_embeddings = nn.Embedding(config['suffix_vocab_size'], self.embedding_dim, padding_idx=0)
        
        # set input dim correctly and initalize the linear layer for d
        if self.repr_type in ['a', 'c']:
            lstm_input_dim = self.embedding_dim
        elif self.repr_type == 'b':
            lstm_input_dim = config['char_hidden_dim']
        elif self.repr_type == 'd':
            lstm_input_dim = self.embedding_dim
            self.combine_linear = nn.Linear(self.embedding_dim + config['char_hidden_dim'], self.embedding_dim) #for simplicty the we take embedding dim + char hidden dim -> embedding dim

        self.bilstm_layer1_fwd = LSTMCell(lstm_input_dim, self.hidden_dim)
        self.bilstm_layer1_bwd = LSTMCell(lstm_input_dim, self.hidden_dim)
        self.bilstm_layer2_fwd = LSTMCell(2 * self.hidden_dim, self.hidden_dim)
        self.bilstm_layer2_bwd = LSTMCell(2 * self.hidden_dim, self.hidden_dim)

        self.hidden2tag = nn.Linear(2 * self.hidden_dim, config['tagset_size'])

    def _process_char_representations(self, char_indices):
        """char lstm encodings for b and d"""
        batch_size, seq_len, max_word_len = char_indices.shape
        
        # reshape to process all chars at once
        char_flat = char_indices.view(-1, max_word_len)  # (batch_size * seq_len, max_word_len)
        char_embeds = self.char_embeddings(char_flat)  # (batch_size * seq_len, max_word_len, char_embedding_dim)
        
        # get lengths for each word before padding
        # so that we can find the correct hidden output later
        actual_lengths = (char_flat != 0).sum(dim=1)  # (batch_size * seq_len,)
        
        _, final_hiddens, _ = self.char_lstm.process_sequence(char_embeds)  # final_hiddens: (batch_size * seq_len, char_hidden_dim)
        
        # handle empty words
        empty_word_mask = (actual_lengths == 0)
        final_hiddens[empty_word_mask] = 0
        
        # reshape back (batch_size, seq_len, char_hidden_dim)
        char_repr = final_hiddens.view(batch_size, seq_len, -1)
        return char_repr

    def forward(self, batch):
        word_indices = batch['words']

        if self.repr_type == 'a':
            embeds = self.word_embeddings(word_indices)
        
        elif self.repr_type == 'b':
            char_indices = batch['chars']  # (batch_size, seq_len, max_word_len)
            embeds = self._process_char_representations(char_indices)
        
        elif self.repr_type == 'c':
            word_embeds = self.word_embeddings(word_indices)
            prefix_embeds = self.prefix_embeddings(batch['prefixes'])
            suffix_embeds = self.suffix_embeddings(batch['suffixes'])
            embeds = word_embeds + prefix_embeds + suffix_embeds
        
        elif self.repr_type == 'd':
            word_embeds = self.word_embeddings(word_indices)
            char_indices = batch['chars']
            char_repr = self._process_char_representations(char_indices)
            combined = torch.cat([word_embeds, char_repr], dim=2)
            embeds = self.combine_linear(combined)

        embeds = self.dropout(embeds)

        # first Bilstm layer
        outputs_fwd1, _, _ = self.bilstm_layer1_fwd.process_sequence(embeds)
        
        embeds_reversed = torch.flip(embeds, dims=[1])
        outputs_bwd1_rev, _, _ = self.bilstm_layer1_bwd.process_sequence(embeds_reversed)
        outputs_bwd1 = list(reversed(outputs_bwd1_rev)) # backwards outputs in the forwards order
        
        layer1_output = torch.stack([torch.cat([hf, hb], dim=1) for hf, hb in zip(outputs_fwd1, outputs_bwd1)], dim=1)
        layer1_output = self.dropout(layer1_output)

        # second bilstm layer
        outputs_fwd2, _, _ = self.bilstm_layer2_fwd.process_sequence(layer1_output)
        
        layer1_reversed = torch.flip(layer1_output, dims=[1])
        outputs_bwd2_rev, _, _ = self.bilstm_layer2_bwd.process_sequence(layer1_reversed)
        outputs_bwd2 = list(reversed(outputs_bwd2_rev))

        layer2_output = torch.stack([torch.cat([hf, hb], dim=1) for hf, hb in zip(outputs_fwd2, outputs_bwd2)], dim=1)
        layer2_output = self.dropout(layer2_output)

        # projection and output
        logits = self.hidden2tag(layer2_output.reshape(-1, 2 * self.hidden_dim))
        return logits
    
####################################
# BELOW begins all the utility code
####################################

def read_data(file_path):
    data = []
    current_words, current_tags = [], []
    
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                if len(line.split()) == 2:
                    word, tag = line.split()
                    if word != "-DOCSTART-":
                        current_words.append(word.lower())
                        current_tags.append(tag)

                if len(line.split()) == 1: # test
                    current_words.append(word.lower())
            else:
                if current_words:
                    data.append({'words': current_words, 'tags': current_tags})
                    current_words, current_tags = [], []
        
        if current_words:
            data.append({'words': current_words, 'tags': current_tags})
            
    return pd.DataFrame(data)

def build_vocabs(train_df):
    word_counts = defaultdict(int)
    char_set = set()
    tag_set = set()
    prefix_set = set()
    suffix_set = set()
    
    for _, row in train_df.iterrows():
        for word in row['words']:
            word_counts[word] += 1
            for char in word:
                char_set.add(char)
            if len(word) >= 3:
                prefix_set.add(word[:3])
                suffix_set.add(word[-3:])
        for tag in row['tags']:
            tag_set.add(tag)
    
    # build vocabs
    words = ['<PAD>', '<UNK>'] + [w for w, _ in word_counts.items()]
    chars = ['<PAD>', '<UNK>'] + list(char_set)
    tags = ['<PAD>'] + list(tag_set)
    prefixes = ['<PAD>', '<UNK>'] + list(prefix_set)
    suffixes = ['<PAD>', '<UNK>'] + list(suffix_set)
    
    word2idx = {w: i for i, w in enumerate(words)}
    char2idx = {c: i for i, c in enumerate(chars)}
    tag2idx = {t: i for i, t in enumerate(tags)}
    prefix2idx = {p: i for i, p in enumerate(prefixes)}
    suffix2idx = {s: i for i, s in enumerate(suffixes)}
    
    return word2idx, char2idx, tag2idx, prefix2idx, suffix2idx

class SequenceDataset(Dataset):
    def __init__(self, df, word2idx, char2idx, tag2idx, prefix2idx, suffix2idx, repr_type):
        self.df = df
        self.word2idx = word2idx
        self.char2idx = char2idx
        self.tag2idx = tag2idx
        self.prefix2idx = prefix2idx
        self.suffix2idx = suffix2idx
        self.repr_type = repr_type
        self.max_word_len = 20

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        words = row['words']
        tags = row['tags']
        
        word_indices = [self.word2idx.get(w, self.word2idx['<UNK>']) for w in words]
        tag_indices = [self.tag2idx[t] for t in tags]
        
        item = {
            'words': torch.tensor(word_indices, dtype=torch.long),
            'tags': torch.tensor(tag_indices, dtype=torch.long)
        }
        
        if self.repr_type in ['b', 'd']:
            char_matrix = []
            for word in words:
                char_ids = [self.char2idx.get(c, self.char2idx['<UNK>']) for c in word[:self.max_word_len]]
                char_ids += [self.char2idx['<PAD>']] * (self.max_word_len - len(char_ids))
                char_matrix.append(char_ids)
            item['chars'] = torch.tensor(char_matrix, dtype=torch.long)
        
        if self.repr_type == 'c':
            prefix_ids = []
            suffix_ids = []
            for word in words:
                if len(word) >= 3:
                    prefix_ids.append(self.prefix2idx.get(word[:3], self.prefix2idx['<UNK>']))
                    suffix_ids.append(self.suffix2idx.get(word[-3:], self.suffix2idx['<UNK>']))
                else:
                    prefix_ids.append(self.prefix2idx['<PAD>'])
                    suffix_ids.append(self.suffix2idx['<PAD>'])
            item['prefixes'] = torch.tensor(prefix_ids, dtype=torch.long)
            item['suffixes'] = torch.tensor(suffix_ids, dtype=torch.long)
        
        return item

def collate_fn(batch):
    batch.sort(key=lambda x: len(x['words']), reverse=True)
    
    max_len = len(batch[0]['words'])
    batch_size = len(batch)
    
    # Pad sequences
    padded_words = torch.full((batch_size, max_len), 0, dtype=torch.long)  # 0 is <PAD>
    padded_tags = torch.full((batch_size, max_len), 0, dtype=torch.long)   # 0 is <PAD>
    
    for i, item in enumerate(batch):
        seq_len = len(item['words'])
        padded_words[i, :seq_len] = item['words']
        padded_tags[i, :seq_len] = item['tags']
    
    result = {'words': padded_words, 'tags': padded_tags}
    
    # Handle character data
    if 'chars' in batch[0]:
        max_word_len = batch[0]['chars'].size(1)
        padded_chars = torch.full((batch_size, max_len, max_word_len), 0, dtype=torch.long)
        for i, item in enumerate(batch):
            seq_len = item['chars'].size(0)
            padded_chars[i, :seq_len] = item['chars']
        result['chars'] = padded_chars
    
    # Handle subword data
    if 'prefixes' in batch[0]:
        padded_prefixes = torch.full((batch_size, max_len), 0, dtype=torch.long)
        padded_suffixes = torch.full((batch_size, max_len), 0, dtype=torch.long)
        for i, item in enumerate(batch):
            seq_len = len(item['prefixes'])
            padded_prefixes[i, :seq_len] = item['prefixes']
            padded_suffixes[i, :seq_len] = item['suffixes']
        result['prefixes'] = padded_prefixes
        result['suffixes'] = padded_suffixes
    
    return result

def evaluate(model, data_loader, device, tag2idx):
    model.eval()
    total_correct = 0
    total_samples = 0
    
    with torch.no_grad():
        for batch in data_loader:
            for key, value in batch.items():
                batch[key] = value.to(device)
            
            # Forward pass
            logits = model(batch)
            targets = batch['tags'].view(-1)
            
            # Print first sequence's tags and predictions
            
            # Mask calculations
            non_pad_mask = (targets != tag2idx['<PAD>'])            
            if 'O' in tag2idx:  # NER task
                o_idx = tag2idx['O']
                o_mask = (targets == o_idx) & logits.argmax(dim=1) == o_idx
                eval_mask = non_pad_mask & ~o_mask
            else:  # POS task
                eval_mask = non_pad_mask
                
            
            active_logits = logits[eval_mask]
            active_targets = targets[eval_mask]
            
            if len(active_targets) > 0:
                predictions = active_logits.argmax(dim=1)
                total_correct += (predictions == active_targets).sum().item()
                total_samples += len(active_targets)
            else:
                print("WARNING: No active targets in this batch")
                
    return total_correct / total_samples if total_samples > 0 else 0.0

##############
# main / train
##############

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('repr', choices=['a', 'b', 'c', 'd'], help='Representation type')
    parser.add_argument('--task', choices=['pos', 'ner'], default='pos', help='Task type (default: %(default)s)')
    parser.add_argument('--embedding_dim', type=int, default=50, help='Embedding dimension (default: %(default)s)')
    parser.add_argument('--char_embedding_dim', type=int, default=32, help='Character embedding dimension (default: %(default)s)')
    parser.add_argument('--char_hidden_dim', type=int, default=64, help='Character hidden dimension (default: %(default)s)')
    parser.add_argument('--hidden_dim', type=int, default=128, help='Hidden dimension (default: %(default)s)')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size (default: %(default)s)')
    parser.add_argument('--learning_rate', type=float, default=0.001, help='Learning rate (default: %(default)s)')
    parser.add_argument('--num_epochs', type=int, default=5, help='Number of epochs (default: %(default)s)')
    parser.add_argument('--grad_clip_norm', type=float, default=1.0, help='Gradient clipping max norm (default: %(default)s)')
    parser.add_argument('--debug', action='store_true', help='Flag to overfit on one example for debugging (default: False)')
    parser.add_argument('--scheduler', action='store_true', help='Use a learning rate scheduler to halve the lr when dev accuracy plateaus (default: False)')
    parser.add_argument('--eval_freq', type=int, default=500, help='How often to eval (number of sentences) (default: %(default)s)')
    parser.add_argument('--force_cpu', action='store_true', help='Force CPU even when cuda is available (default: False)')
    parser.add_argument('--weight_noise', type=float, default=0.0, 
                       help='Amount of Gaussian noise to add to weights (default: %(default)s)')
    parser.add_argument('--label_smoothing', type=float, default=0.0, help='Amount of label smoothing to use in the loss (default: %(default)s)')
    parser.add_argument('--weight_decay', type=float, default=0.0, help='Amount of weight decay to use with AdamW (default: %(default)s)')


    args = parser.parse_args()
    
    # gen filename
    model_file = f"blstm_{args.repr}_{args.task}_{args.embedding_dim}_{args.hidden_dim}.pt"
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if args.force_cpu:
        device = torch.device('cpu')
    print(f"Using device: {device}")
    
    train_df = read_data(f"../data/{args.task}/train")
    if args.debug:
        train_df = read_data(f"../data/{args.task}/train").iloc[:1]
    dev_df = read_data(f"../data/{args.task}/dev")
    word2idx, char2idx, tag2idx, prefix2idx, suffix2idx = build_vocabs(train_df)
    
    train_dataset = SequenceDataset(train_df, word2idx, char2idx, tag2idx, prefix2idx, suffix2idx, args.repr)
    dev_dataset = SequenceDataset(dev_df, word2idx, char2idx, tag2idx, prefix2idx, suffix2idx, args.repr)
    
    train_loader = DataLoader(train_dataset, batch_size=1 if args.debug else args.batch_size, shuffle=True, collate_fn=collate_fn)
    dev_loader = DataLoader(dev_dataset, batch_size=args.batch_size, collate_fn=collate_fn)
    
    config = {
        'repr_type': args.repr,
        'vocab_size': len(word2idx),
        'char_vocab_size': len(char2idx),
        'tagset_size': len(tag2idx),
        'prefix_vocab_size': len(prefix2idx),
        'suffix_vocab_size': len(suffix2idx),
        'embedding_dim': args.embedding_dim,
        'char_embedding_dim': args.char_embedding_dim,
        'char_hidden_dim': args.char_hidden_dim,
        'hidden_dim': args.hidden_dim
    }
    
    model = BiLSTMTagger(config).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    if args.scheduler:
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', patience=0, factor=0.5)
    criterion = nn.CrossEntropyLoss(ignore_index=tag2idx['<PAD>'], label_smoothing=args.label_smoothing)
    
    # training loop starts here
    best_dev_acc = 0
    dev_accuracies = []
    sentences_seen = 0
    
    num_epochs = args.num_epochs
    if args.debug:
        num_epochs = 51200
    for epoch in range(num_epochs):
        start_time = time.time()
        print(f"Epoch: {epoch}")
        model.train()
        
        for batch_idx, batch in enumerate(train_loader):
            optimizer.zero_grad()
            
            for key, value in batch.items():
                batch[key] = value.to(device)
            
            logits = model(batch)
            targets = batch['tags'].view(-1)
            loss = criterion(logits, targets)
            
            if args.weight_noise > 0:
                for param in model.parameters():
                    if param is not None:
                        noise = torch.randn_like(param) * args.weight_noise
                        param.data.add_(noise)

            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=args.grad_clip_norm)
            optimizer.step()
            
            sentences_seen += batch['words'].size(0)
            
            # evaluate every 500 sentences
            iter_freq = int(args.eval_freq / batch['words'].size(0))
            if iter_freq==0 or batch_idx % iter_freq == 0:
                print(f"{batch_idx:0{len(str(len(train_loader)))}d}/{len(train_loader)}: train batch loss {loss.item():.4f}")
                if not args.debug:
                    dev_acc = evaluate(model, dev_loader, device, tag2idx)
                    if args.scheduler:
                        scheduler.step(dev_acc)
                    dev_accuracies.append((sentences_seen // 100, dev_acc))
                    print(f"Sentences: {sentences_seen}, Dev Acc: {dev_acc:.4f}")
                    
                    if dev_acc > best_dev_acc:
                        best_dev_acc = dev_acc
                        torch.save({
                            'model_state_dict': model.state_dict(),
                            'config': config,
                            'word2idx': word2idx,
                            'char2idx': char2idx,
                            'tag2idx': tag2idx,
                            'prefix2idx': prefix2idx,
                            'suffix2idx': suffix2idx
                        }, model_file)
                
                model.train()
        print(f"ellapsed time: {time.time() - start_time:.2f}")
    
    # save learning curve
    curve_file = model_file.replace('.pt', f'.{args.task}.{args.repr}.curve.json')
    with open(curve_file, 'w') as f:
        json.dump(dev_accuracies, f)
    
    print(f"Training complete. Best dev accuracy: {best_dev_acc:.4f}")
    print(f"Model saved to: {model_file}")
    print(f"Learning curve saved to: {curve_file}")

if __name__ == '__main__':
    main()