# bilstmPredict.py

import argparse
import json
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm


# --- LSTMCell (Copied from bilstmTrain.py or a shared util) ---
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

# --- BiLSTMTagger Model (Copied from bilstmTrain.py or a shared util) ---
class BiLSTMTagger(nn.Module):
    def __init__(self, model_config):
        super().__init__()
        self.repr_type = model_config['repr_type']
        self.hidden_dim = model_config['hidden_dim']
        self.embedding_dim = model_config['embedding_dim']
        
        if self.repr_type in ['a', 'c', 'd']:
            if model_config.get('pretrained_embeddings') is not None and not isinstance(model_config.get('pretrained_embeddings'), bool): # check if actual embeddings array passed
                self.word_embeddings = nn.Embedding.from_pretrained(
                    torch.FloatTensor(model_config['pretrained_embeddings']),
                    freeze=model_config.get('fixed_embeddings', False)
                )
                # Update embedding_dim from pretrained if it changed (should be consistent with saved config)
                self.embedding_dim = model_config['pretrained_embeddings'].shape[1] 
            else: # No pretrained or dummy bool was passed
                self.word_embeddings = nn.Embedding(model_config['vocab_size'], self.embedding_dim)
        
        current_input_dim = 0
        if self.repr_type in ['b', 'd']:
            self.char_vocab_size = model_config['char_vocab_size']
            self.char_embedding_dim = model_config['char_embedding_dim']
            self.char_hidden_dim = model_config['char_hidden_dim']
            self.char_embeddings = nn.Embedding(self.char_vocab_size, self.char_embedding_dim, padding_idx=model_config['CHAR_PAD_IDX'])
            self.char_lstm = nn.LSTM(self.char_embedding_dim, self.char_hidden_dim, batch_first=True, bidirectional=False)
            if self.repr_type == 'b':
                current_input_dim = self.char_hidden_dim
        
        if self.repr_type == 'c':
            self.prefix_embeddings = nn.Embedding(model_config['prefix_vocab_size'], self.embedding_dim, padding_idx=model_config['PREFIX_PAD_IDX'])
            self.suffix_embeddings = nn.Embedding(model_config['suffix_vocab_size'], self.embedding_dim, padding_idx=model_config['SUFFIX_PAD_IDX'])
            current_input_dim = self.embedding_dim

        if self.repr_type == 'a':
            current_input_dim = self.embedding_dim
        
        if self.repr_type == 'd':
            concat_dim_d = self.embedding_dim + self.char_hidden_dim
            self.repr_d_linear = nn.Linear(concat_dim_d, self.embedding_dim) 
            current_input_dim = self.embedding_dim

        self.actual_input_dim_to_lstm = current_input_dim
        self.bilstm_layer1_fwd = LSTMCell(self.actual_input_dim_to_lstm, self.hidden_dim)
        self.bilstm_layer1_bwd = LSTMCell(self.actual_input_dim_to_lstm, self.hidden_dim)
        self.bilstm_layer2_fwd = LSTMCell(2 * self.hidden_dim, self.hidden_dim)
        self.bilstm_layer2_bwd = LSTMCell(2 * self.hidden_dim, self.hidden_dim)
        self.hidden2tag = nn.Linear(2 * self.hidden_dim, model_config['tagset_size'])

    def _init_hidden_cell(self, batch_size, device):
        return (torch.zeros(batch_size, self.hidden_dim).to(device),
                torch.zeros(batch_size, self.hidden_dim).to(device))

    def forward(self, batch): # Same forward pass logic as in train
        sentences_word_indices = batch['words']
        lengths = batch['lengths'] 
        device = sentences_word_indices.device
        batch_size = sentences_word_indices.size(0)
        seq_len = sentences_word_indices.size(1)
        embeds = None
        if self.repr_type == 'a':
            embeds = self.word_embeddings(sentences_word_indices) 
        elif self.repr_type == 'b':
            sentences_char_indices = batch['chars']
            char_input_flat = sentences_char_indices.view(-1, sentences_char_indices.size(2))
            char_embeds_flat = self.char_embeddings(char_input_flat)
            _, (h_n_char, _) = self.char_lstm(char_embeds_flat)
            word_repr_from_chars = h_n_char.squeeze(0)
            embeds = word_repr_from_chars.view(batch_size, seq_len, self.char_hidden_dim)
        elif self.repr_type == 'c':
            sentences_prefix_indices = batch['prefixes']
            sentences_suffix_indices = batch['suffixes']
            word_e = self.word_embeddings(sentences_word_indices)
            prefix_e = self.prefix_embeddings(sentences_prefix_indices)
            suffix_e = self.suffix_embeddings(sentences_suffix_indices)
            embeds = word_e + prefix_e + suffix_e
        elif self.repr_type == 'd':
            sentences_char_indices = batch['chars']
            word_e = self.word_embeddings(sentences_word_indices)
            char_input_flat = sentences_char_indices.view(-1, sentences_char_indices.size(2))
            char_embeds_flat = self.char_embeddings(char_input_flat)
            _, (h_n_char, _) = self.char_lstm(char_embeds_flat)
            word_repr_from_chars = h_n_char.squeeze(0).view(batch_size, seq_len, self.char_hidden_dim)
            concatenated_repr = torch.cat((word_e, word_repr_from_chars), dim=2)
            embeds = self.repr_d_linear(concatenated_repr.view(-1, concatenated_repr.size(2)))
            embeds = embeds.view(batch_size, seq_len, self.embedding_dim)

        h_fwd1, c_fwd1 = self._init_hidden_cell(batch_size, device)
        h_bwd1, c_bwd1 = self._init_hidden_cell(batch_size, device)
        outputs_fwd1, outputs_bwd1 = [], []
        for t in range(seq_len):
            c_fwd1, h_fwd1 = self.bilstm_layer1_fwd(embeds[:, t, :], c_fwd1, h_fwd1)
            outputs_fwd1.append(h_fwd1)
        for t in range(seq_len - 1, -1, -1):
            c_bwd1, h_bwd1 = self.bilstm_layer1_bwd(embeds[:, t, :], c_bwd1, h_bwd1)
            outputs_bwd1.append(h_bwd1)
        outputs_bwd1.reverse()
        hiddens_layer1_stacked = torch.stack([torch.cat((hf, hb), dim=1) for hf, hb in zip(outputs_fwd1, outputs_bwd1)], dim=1)

        h_fwd2, c_fwd2 = self._init_hidden_cell(batch_size, device)
        h_bwd2, c_bwd2 = self._init_hidden_cell(batch_size, device)
        outputs_fwd2, outputs_bwd2 = [], []
        for t in range(seq_len):
            c_fwd2, h_fwd2 = self.bilstm_layer2_fwd(hiddens_layer1_stacked[:, t, :], c_fwd2, h_fwd2)
            outputs_fwd2.append(h_fwd2)
        for t in range(seq_len - 1, -1, -1):
            c_bwd2, h_bwd2 = self.bilstm_layer2_bwd(hiddens_layer1_stacked[:, t, :], c_bwd2, h_bwd2)
            outputs_bwd2.append(h_bwd2)
        outputs_bwd2.reverse()
        hiddens_layer2_stacked = torch.stack([torch.cat((hf, hb), dim=1) for hf, hb in zip(outputs_fwd2, outputs_bwd2)], dim=1)
        
        lstm_feats = hiddens_layer2_stacked.reshape(-1, 2 * self.hidden_dim)
        logits = self.hidden2tag(lstm_feats)
        return logits


# --- Data Handling (Copied from bilstmTrain.py or a shared util) ---
def clean_word(word):
    return word.lower()

def read_data(file_path, is_test=False): # Same as in train
    data = []
    sentence_id_counter = 0
    current_words, current_tags = [], []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                if is_test:
                    word = line
                    current_words.append(clean_word(word))
                else: # Not typically used for predict, but good to have
                    parts = line.split()
                    if len(parts) == 2:
                        word, tag = parts
                        if word == "-DOCSTART-": 
                            if current_words: 
                                data.append({'sentence_id': sentence_id_counter, 'words': list(current_words), 'tags': list(current_tags) if not is_test else []})
                                sentence_id_counter += 1
                                current_words, current_tags = [], []
                            continue 
                        current_words.append(clean_word(word))
                        current_tags.append(tag)
                    else: 
                        print(f"Skipping malformed line: {line}")
            else: 
                if current_words: 
                    data.append({'sentence_id': sentence_id_counter, 'words': list(current_words), 'tags': list(current_tags) if not is_test else []})
                    sentence_id_counter += 1
                    current_words, current_tags = [], []
        if current_words:
            data.append({'sentence_id': sentence_id_counter, 'words': list(current_words), 'tags': list(current_tags) if not is_test else []})
    return pd.DataFrame(data)


class SequenceDataset(Dataset): # Same as in train
    def __init__(self, df, word2idx, tag2idx, char2idx, prefix2idx, suffix2idx, config, is_train=True):
        self.df = df
        self.word2idx = word2idx
        self.tag2idx = tag2idx
        self.char2idx = char2idx
        self.prefix2idx = prefix2idx
        self.suffix2idx = suffix2idx
        self.repr_type = config['repr_type']
        self.max_word_len = config['max_word_len']
        self.is_train = is_train

        self.WORD_PAD_IDX = config['WORD_PAD_IDX']
        self.TAG_PAD_IDX = config.get('TAG_PAD_IDX',0) # May not exist if predict on unlabeled
        self.CHAR_PAD_IDX = config['CHAR_PAD_IDX']
        self.PREFIX_PAD_IDX = config['PREFIX_PAD_IDX']
        self.SUFFIX_PAD_IDX = config['SUFFIX_PAD_IDX']
        
        self.WORD_UNK_IDX = self.word2idx.get('<UNK>',0)
        self.CHAR_UNK_IDX = self.char2idx.get('<UNK_CHAR>',0)
        self.PREFIX_UNK_IDX = self.prefix2idx.get('<UNK_SUB>',0)
        self.SUFFIX_UNK_IDX = self.suffix2idx.get('<UNK_SUB>',0)


    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        words = row['words']
        word_indices = [self.word2idx.get(w, self.WORD_UNK_IDX) for w in words]
        item = {"words": torch.tensor(word_indices, dtype=torch.long), "lengths": len(words)}

        if self.is_train and 'tags' in row and row['tags']: # For dev/test with labels
            tag_indices = [self.tag2idx[t] for t in row['tags']]
            item["tags"] = torch.tensor(tag_indices, dtype=torch.long)

        if self.repr_type in ['b', 'd']:
            sentence_chars_indices = []
            for word in words:
                char_ids = [self.char2idx.get(c, self.CHAR_UNK_IDX) for c in word]
                char_ids = char_ids[:self.max_word_len] + \
                           [self.CHAR_PAD_IDX] * (self.max_word_len - len(char_ids))
                sentence_chars_indices.append(torch.tensor(char_ids, dtype=torch.long))
            item["chars"] = torch.stack(sentence_chars_indices) if sentence_chars_indices else torch.empty(0, self.max_word_len, dtype=torch.long)
        
        if self.repr_type == 'c':
            p_indices = [self.prefix2idx.get(word[:3], self.PREFIX_UNK_IDX) if len(word) >= 3 else self.PREFIX_PAD_IDX for word in words]
            s_indices = [self.suffix2idx.get(word[-3:], self.SUFFIX_UNK_IDX) if len(word) >= 3 else self.SUFFIX_PAD_IDX for word in words]
            item["prefixes"] = torch.tensor(p_indices, dtype=torch.long)
            item["suffixes"] = torch.tensor(s_indices, dtype=torch.long)
        return item

def collate_fn_wrapper(word_pad_idx, tag_pad_idx, char_pad_idx, prefix_pad_idx, suffix_pad_idx, repr_type, max_word_len): # Same as in train
    def collate_fn(batch):
        batch.sort(key=lambda x: x["lengths"], reverse=True)
        lengths = torch.tensor([item["lengths"] for item in batch], dtype=torch.long)
        max_seq_len = lengths.max().item()
        padded_words = torch.full((len(batch), max_seq_len), word_pad_idx, dtype=torch.long)
        for i, item in enumerate(batch):
            seq_len = item["words"].size(0)
            padded_words[i, :seq_len] = item["words"]
        collated_batch = {"words": padded_words, "lengths": lengths}
        if "tags" in batch[0] and batch[0]["tags"] is not None and batch[0]["tags"].numel() > 0 :
            padded_tags = torch.full((len(batch), max_seq_len), tag_pad_idx, dtype=torch.long)
            for i, item in enumerate(batch):
                if item["tags"].numel() > 0:
                    seq_len = item["tags"].size(0)
                    padded_tags[i, :seq_len] = item["tags"]
            collated_batch["tags"] = padded_tags
        if repr_type in ['b', 'd']:
            padded_chars = torch.full((len(batch), max_seq_len, max_word_len), char_pad_idx, dtype=torch.long)
            for i, item in enumerate(batch):
                if item["chars"].numel() > 0:
                    seq_len, current_max_word_len = item["chars"].shape
                    padded_chars[i, :seq_len, :current_max_word_len] = item["chars"]
            collated_batch["chars"] = padded_chars
        if repr_type == 'c':
            padded_prefixes = torch.full((len(batch), max_seq_len), prefix_pad_idx, dtype=torch.long)
            padded_suffixes = torch.full((len(batch), max_seq_len), suffix_pad_idx, dtype=torch.long)
            for i, item in enumerate(batch):
                if item["prefixes"].numel() > 0:
                    seq_len = item["prefixes"].size(0)
                    padded_prefixes[i, :seq_len] = item["prefixes"]
                    padded_suffixes[i, :seq_len] = item["suffixes"]
            collated_batch["prefixes"] = padded_prefixes
            collated_batch["suffixes"] = padded_suffixes
        return collated_batch
    return collate_fn


# --- Prediction and Plotting ---
def predict_tags(model, data_loader, device, idx2tag):
    model.eval()
    all_predictions = []
    all_original_words = [] # For output file writing

    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm(data_loader, desc="Predicting")):
            # Store original words before moving to device
            # This part assumes `read_data` for test files stores words in a way that can be retrieved by SequenceDataset
            # The current SequenceDataset doesn't store original words. Let's assume test_df has 'words' column accessible.
            # This needs to be handled carefully for output generation. For now, let's focus on tag prediction.

            original_batch_words = []
            for i in range(batch['words'].size(0)): # Iterate through sentences in batch
                 # Get original words for this sentence based on length
                 sent_len = batch['lengths'][i].item()
                 # This step needs access to the original words from the dataframe,
                 # which means the dataloader or dataset should somehow provide them or indices to them.
                 # For simplicity, we will reconstruct this from test_df later.
            
            for key, value in batch.items():
                if isinstance(value, torch.Tensor):
                    batch[key] = value.to(device)

            logits = model(batch) # (bs * seq_len, num_tags)
            predicted_indices_flat = torch.max(logits, 1)[1] # (bs * seq_len)
            
            # Reshape and collect predictions per sentence
            start_idx = 0
            for i in range(batch['words'].size(0)): # Iterate through sentences in batch
                sent_len = batch['lengths'][i].item()
                preds_for_sent = predicted_indices_flat[start_idx : start_idx + sent_len]
                all_predictions.append([idx2tag[idx.item()] for idx in preds_for_sent])
                start_idx += sent_len
                
    return all_predictions

def generate_plots(task_mode, model_file_path_used_for_predict):
    # Path logic: model_file_path_used_for_predict is like "mymodels/a_pos.pt"
    # Curve files are like "mymodels/a_pos.pos.a.curve.json"
    # We need to find all curve files for the given task in the same directory.
    
    model_dir = os.path.dirname(model_file_path_used_for_predict)
    base_name_parts = os.path.basename(model_file_path_used_for_predict).split('.') # e.g. ['a_pos', 'pt']
    #This assumes model file is named like "repr_task.pt"
    #Example: model_file = "models/a_pos.pt"
    #curve_file name: models/a_pos.pos.a.curve.json (current format)
    #Need to make this robust.
    
    print(f"Generating plots for {task_mode.upper()} task...")
    plt.figure(figsize=(10, 6))
    
    found_curves = False
    for repr_choice in ['a', 'b', 'c', 'd']:
        # Construct expected curve file name based on convention from bilstmTrain.py
        # If model_file was "mymodels/model_a_pos.pt", curve file is "mymodels/model_a_pos.pos.a.curve.json"
        # A bit tricky. Let's simplify: assume curve files are named based on task and repr directly.
        # e.g. curve_dir/pos.a.curve.json, curve_dir/pos.b.curve.json
        # For this, we need to pass a directory where curve files are stored.
        # Let's assume the model_file_path's directory contains relevant curve files.
        # And curve files are named like: {original_model_name}.{task}.{repr}.curve.json
        
        # Try to find a curve file that matches the repr_choice and task
        # This is heuristic if model names are not strictly repr_task.pt
        # A more robust way: bilstmTrain saves as {model_base_name_from_arg}.{task}.{repr}.curve.json
        # bilstmPredict then uses this base name pattern.
        
        # Let's assume model files are simply named like "a.pt", "b.pt" in a task-specific folder or with task in name.
        # e.g., model_file = /path/to/models_pos/a.pt
        # curve_file = /path/to/models_pos/a.pos.a.curve.json (this is based on my bilstmTrain save format)
        
        # Simplified: assume model_file is like "basename_repr_task.pt"
        # and curves are "basename_repr_task.task.repr.curve.json"
        # The user provides one model file, e.g., "mymodel_a_pos.pt". task is "pos".
        # We need "mymodel_a_pos.pos.a.curve.json", "mymodel_b_pos.pos.b.curve.json" etc.
        # This means the base part of the name "mymodel" needs to be identified.
        
        # Let's try to match based on the directory of `model_file_path_used_for_predict`
        # and look for any file ending with .{task_mode}.{repr_choice}.curve.json
        curve_file_path = None
        for f_name in os.listdir(model_dir):
            if f_name.endswith(f".{task_mode}.{repr_choice}.curve.json"):
                curve_file_path = os.path.join(model_dir, f_name)
                break
        
        if curve_file_path and os.path.exists(curve_file_path):
            try:
                with open(curve_file_path, 'r') as f:
                    data = json.load(f) # List of [sentences_seen/100, accuracy]
                
                if data:
                    found_curves = True
                    steps, accuracies = zip(*data)
                    plt.plot(steps, accuracies, marker='o', linestyle='-', label=f'Repr {repr_choice.upper()}')
            except Exception as e:
                print(f"Could not load or parse curve file {curve_file_path}: {e}")
        else:
            print(f"Curve file for repr {repr_choice}, task {task_mode} not found at expected path: {curve_file_path if curve_file_path else 'pattern not matched'}")

    if not found_curves:
        print("No learning curve data found to plot.")
        return

    plt.title(f'Learning Curves for {task_mode.upper()} Task (Dev Accuracy)')
    plt.xlabel('Number of Sentences Seen / 100')
    plt.ylabel('Dev Accuracy')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    
    plot_filename = f'learning_curves_{task_mode}.png'
    plt.savefig(plot_filename)
    print(f"Learning curve plot saved to {plot_filename}")
    plt.close()


# --- Main Execution ---
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='BiLSTM Tagger Prediction and Plotting')
    parser.add_argument('repr', choices=['a', 'b', 'c', 'd'], help='Input representation type of the model to use for prediction')
    parser.add_argument('model_file', type=str, help='Path to the trained model file (.pt)')
    parser.add_argument('input_file', type=str, help='Path to the input data file (blind test data)')
    parser.add_argument('--task', type=str, required=True, choices=['pos', 'ner'], help='Tagging task (pos or ner) for plotting context')
    parser.add_argument('--output_file', type=str, default='predictions.txt', help='Path to save the predictions')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size for prediction')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')

    args = parser.parse_args()

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load model configuration and state
    print(f"Loading model from {args.model_file}...")
    if not os.path.exists(args.model_file):
        print(f"Error: Model file {args.model_file} not found.")
        exit(1)
        
    checkpoint = torch.load(args.model_file, map_location=device)
    
    # The checkpoint itself is the model_config dictionary with 'model_state_dict'
    model_config = checkpoint 
    
    # Handle case where pretrained_embeddings might be a path or ndarray in config
    # For prediction, we don't need to load the actual array if it's just for nn.Embedding init size.
    # If nn.Embedding.from_pretrained was used, it's part of state_dict.
    # The BiLSTMTagger class needs to handle this. Let's ensure it uses vocab_size if pretrained_embeddings is not an array.
    if 'pretrained_embeddings' in model_config and not isinstance(model_config['pretrained_embeddings'], np.ndarray):
        model_config['pretrained_embeddings'] = None # Don't try to load from path here if it wasn't saved as array

    model = BiLSTMTagger(model_config)
    model.load_state_dict(model_config['model_state_dict'])
    model.to(device)
    model.eval()

    # Extract necessary vocabs and pad indices from loaded config
    word2idx = model_config['word2idx']
    tag2idx = model_config['tag2idx']
    char2idx = model_config['char2idx']
    prefix2idx = model_config['prefix2idx']
    suffix2idx = model_config['suffix2idx']
    idx2tag = {idx: tag for tag, idx in tag2idx.items()}

    # Load test data
    print(f"Loading input data from {args.input_file}...")
    test_df_raw = read_data(args.input_file, is_test=True)
    
    # Create dataset and dataloader
    test_dataset = SequenceDataset(test_df_raw, word2idx, tag2idx, char2idx, prefix2idx, suffix2idx, model_config, is_train=False)
    
    # Make sure pad indices for collate_fn are from the loaded model_config
    custom_collate_fn = collate_fn_wrapper(model_config['WORD_PAD_IDX'], 
                                           model_config.get('TAG_PAD_IDX',0), # TAG_PAD_IDX might not be needed if no true tags
                                           model_config['CHAR_PAD_IDX'], 
                                           model_config['PREFIX_PAD_IDX'], 
                                           model_config['SUFFIX_PAD_IDX'], 
                                           model.repr_type, # Use repr from loaded model
                                           model_config['max_word_len'])

    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, collate_fn=custom_collate_fn)

    # Predict tags
    print("Predicting tags...")
    predicted_tags_per_sentence = predict_tags(model, test_loader, device, idx2tag)

    # Save predictions
    print(f"Saving predictions to {args.output_file}...")
    with open(args.output_file, 'w', encoding='utf-8') as f:
        for i, sentence_words in enumerate(test_df_raw['words']):
            predicted_tags_for_this_sentence = predicted_tags_per_sentence[i]
            if len(sentence_words) != len(predicted_tags_for_this_sentence):
                print(f"Warning: Mismatch in length for sentence {i}. Words: {len(sentence_words)}, Tags: {len(predicted_tags_for_this_sentence)}")
                # Fallback or error handling, e.g. truncate/pad tags or skip sentence
                # For now, we'll write what we have, possibly misaligned
                min_len = min(len(sentence_words), len(predicted_tags_for_this_sentence))
                for k in range(min_len):
                     f.write(f"{sentence_words[k]}\t{predicted_tags_for_this_sentence[k]}\n")
            else:
                for word, tag in zip(sentence_words, predicted_tags_for_this_sentence):
                    f.write(f"{word}\t{tag}\n")
            f.write("\n") # End of sentence

    print("Predictions saved.")

    # Generate and save plots
    # The prompt asks for two graphs: one for POS, one for NER.
    # This script is run with a --task argument. It generates ONE graph for that task.
    # To get both, run this script twice: once with --task pos, once with --task ner.
    # (Assuming curve data for both tasks is available)
    
    # The current script generates one plot based on the --task argument.
    # If the prompt means this single script invocation should somehow create two distinct graph files (pos_curves.png, ner_curves.png),
    # then it would need to load curve data for *both* tasks, irrespective of the --task arg used for prediction context.
    # Let's assume it generates one graph file per run, corresponding to the specified --task.
    
    # Plotting for the specified task
    generate_plots(args.task, args.model_file)
    
    # If you need to generate both plots (POS and NER) in one go, assuming all data is available:
    # print("\n--- Generating POS Plot ---")
    # generate_plots('pos', args.model_file) # Assumes models/curve files for POS exist
    # print("\n--- Generating NER Plot ---")
    # generate_plots('ner', args.model_file) # Assumes models/curve files for NER exist
    
    print("Process finished.")