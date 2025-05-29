import json
import glob
import matplotlib.pyplot as plt
import torch
import pandas as pd
from collections import defaultdict
import argparse

# reuse code where i can but I redefine some of these below for 
# predictions instead of training.
from bilstmTrain import BiLSTMTagger, SequenceDataset, collate_fn
from torch.utils.data import DataLoader, Dataset

def read_data(file_path):
    data = []
    current_words, current_tags = [], []
    
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                parts = line.split()
                if len(parts) == 2:  # training/dev data with word and tag
                    word, tag = parts
                    if word != "-DOCSTART-":
                        current_words.append(word.lower())
                        current_tags.append(tag)
                elif len(parts) == 1:  # test data with only word
                    word = parts[0]
                    if word != "-DOCSTART-":
                        current_words.append(word.lower())
                        # for test data, we'll add dummy tags that will be ignored, just to 
                        # avoid refactoring
                        current_tags.append('O' if current_tags or len(current_words) == 1 else 'O')
            else:
                if current_words:
                    data.append({'words': current_words, 'tags': current_tags})
                    current_words, current_tags = [], []
        
        if current_words:
            data.append({'words': current_words, 'tags': current_tags})
            
    return pd.DataFrame(data)

class SequenceDataset(Dataset):
    def __init__(self, df, word2idx, char2idx, tag2idx, prefix2idx, suffix2idx, repr_type, is_test=False):
        self.df = df
        self.word2idx = word2idx
        self.char2idx = char2idx
        self.tag2idx = tag2idx
        self.prefix2idx = prefix2idx
        self.suffix2idx = suffix2idx
        self.repr_type = repr_type
        self.max_word_len = 20
        self.is_test = is_test

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        words = row['words']
        tags = row['tags'] if not self.is_test else ['O'] * len(words)
        
        word_indices = [self.word2idx.get(w, self.word2idx['<UNK>']) for w in words]
        tag_indices = [self.tag2idx[t] for t in tags] if not self.is_test else [self.tag2idx.get('O', 0)] * len(words)
        
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
    max_len = max(len(item['words']) for item in batch)
    batch_size = len(batch)
    
    padded_words = torch.full((batch_size, max_len), 0, dtype=torch.long)
    padded_tags = torch.full((batch_size, max_len), 0, dtype=torch.long)
    
    for i, item in enumerate(batch):
        seq_len = len(item['words'])
        padded_words[i, :seq_len] = item['words']
        padded_tags[i, :seq_len] = item['tags']
    
    result = {'words': padded_words, 'tags': padded_tags}
    
    if 'chars' in batch[0]:
        max_word_len = batch[0]['chars'].size(1)
        padded_chars = torch.full((batch_size, max_len, max_word_len), 0, dtype=torch.long)
        for i, item in enumerate(batch):
            seq_len = item['chars'].size(0)
            padded_chars[i, :seq_len] = item['chars']
        result['chars'] = padded_chars
    
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


def load_learning_curves():
    """Load all learning curve JSON files and organize by task and representation."""
    curves = defaultdict(dict)
    
    # find all curve files
    curve_files = glob.glob("*.curve.json")
    
    for file in curve_files:
        # parse filename to extract task and representation
        # expected format: blstm_{repr}_{task}_{embedding_dim}_{hidden_dim}.{task}.{repr}.curve.json
        parts = file.split('.')
        if len(parts) >= 4 and parts[-1] == 'json' and parts[-2] == 'curve':
            task = parts[-4]  # pos or ner
            repr_type = parts[-3]  # a, b, c, or d
            
            with open(file, 'r') as f:
                data = json.load(f)
                curves[task][repr_type] = data
    
    return curves

def plot_learning_curves(curves):
    """Create plots for each task showing all representations."""
    for task in ['pos', 'ner']:
        if task not in curves or not curves[task]:
            print(f"No data found for {task} task")
            continue
            
        plt.figure(figsize=(10, 6))
        
        for repr_type in ['a', 'b', 'c', 'd']:
            if repr_type in curves[task]:
                data = curves[task][repr_type]
                sentences = [point[0] * 100 for point in data]  # convert back from hundreds
                accuracies = [point[1] for point in data]
                plt.plot(sentences, accuracies, label=f'Representation {repr_type}', marker='o', markersize=3)
        
        plt.xlabel('Sentences Seen')
        plt.ylabel('Dev Accuracy')
        plt.title(f'{task.upper()} Task - Learning Curves')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        plt.savefig(f'{task}_learning_curves.png', dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Saved {task}_learning_curves.png")

def find_best_models(curves):
    """Find the best model for each task based on final dev accuracy."""
    best_models = {}
    
    for task in ['pos', 'ner']:
        if task not in curves or not curves[task]:
            continue
            
        best_acc = 0
        best_repr = None
        
        for repr_type, data in curves[task].items():
            if data:  # check if data is not empty
                final_acc = data[-1][1]  # last acc val
                print(f"{task} {repr_type}: final dev acc = {final_acc:.4f}")
                
                if final_acc > best_acc:
                    best_acc = final_acc
                    best_repr = repr_type
        
        if best_repr:
            best_models[task] = (best_repr, best_acc)
            print(f"Best model for {task}: representation {best_repr} with accuracy {best_acc:.4f}")
    
    return best_models

def load_model_and_vocabs(task, repr_type):
    """Load the saved model and vocabularies."""
    model_files = glob.glob(f"blstm_{repr_type}_{task}_*.pt")
    
    if not model_files:
        raise FileNotFoundError(f"No model file found for {task} task, representation {repr_type}")
    
    model_file = model_files[0]
    print(f"Loading model from {model_file}")
    
    checkpoint = torch.load(model_file, map_location='cpu')
    
    config = checkpoint['config']
    word2idx = checkpoint['word2idx']
    char2idx = checkpoint['char2idx']
    tag2idx = checkpoint['tag2idx']
    prefix2idx = checkpoint['prefix2idx']
    suffix2idx = checkpoint['suffix2idx']
    
    model = BiLSTMTagger(config)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    return model, word2idx, char2idx, tag2idx, prefix2idx, suffix2idx

def generate_predictions(task, repr_type):
    model, word2idx, char2idx, tag2idx, prefix2idx, suffix2idx = load_model_and_vocabs(task, repr_type)
    idx2tag = {i: tag for tag, i in tag2idx.items()}
    test_df = read_data(f"../data/{task}/test")
    test_dataset = SequenceDataset(test_df, word2idx, char2idx, tag2idx, prefix2idx, suffix2idx, repr_type, is_test=True)
    test_loader = DataLoader(test_dataset, batch_size=32, collate_fn=collate_fn)
    
    all_predictions = []
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    
    with torch.no_grad():
        for batch in test_loader:
            for key, value in batch.items():
                batch[key] = value.to(device)
            
            logits = model(batch)
            predictions = logits.argmax(dim=1)
            
            batch_size, seq_len = batch['words'].shape
            predictions = predictions.view(batch_size, seq_len)
            
            for i in range(batch_size):
                seq_len_actual = (batch['words'][i] != 0).sum().item()
                seq_predictions = predictions[i][:seq_len_actual]
                all_predictions.extend([idx2tag[pred.item()] for pred in seq_predictions])
    
    output_file = f"test4.{task}"
    with open(output_file, 'w') as f:
        word_idx = 0
        for _, row in test_df.iterrows():
            for word in row['words']:
                f.write(f"{word} {all_predictions[word_idx]}\n")
                word_idx += 1
            f.write("\n")


def main():
    print("Loading learning curves...")
    curves = load_learning_curves()
    
    print("Creating plots...")
    plot_learning_curves(curves)
    
    print("Finding best models...")
    best_models = find_best_models(curves)
    
    print("Generating predictions...")
    for task, (repr_type, acc) in best_models.items():
        print(f"Generating predictions for {task} using representation {repr_type}")
        generate_predictions(task, repr_type)
    
    print("Analysis complete!")

if __name__ == '__main__':
    main()