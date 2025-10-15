import torch
from torch.utils.data import Dataset, DataLoader
from datasets import load_dataset
from collections import Counter
import pytorch_lightning as pl
import pickle
import os


class Tokenizer:
    def __init__(self, vocab_size=10000):
        self.vocab_size = vocab_size
        self.word2idx = {"<PAD>": 0, "<UNK>": 1, "<EOS>": 2}
        self.idx2word = {0: "<PAD>", 1: "<UNK>", 2: "<EOS>"}
        self.vocab_built = False
    
    def build_vocab(self, texts):
        word_counts = Counter()
        for text in texts:
            words = text.lower().split()
            word_counts.update(words)
        
        most_common = word_counts.most_common(self.vocab_size - 3)
        
        for idx, (word, _) in enumerate(most_common, start=3):
            self.word2idx[word] = idx
            self.idx2word[idx] = word
        
        self.vocab_built = True
    
    def encode(self, text):
        words = text.lower().split()
        return [self.word2idx.get(word, self.word2idx["<UNK>"]) for word in words]
    
    def decode(self, indices):
        return " ".join([self.idx2word.get(idx, "<UNK>") for idx in indices])
    
    def save(self, path):
        with open(path, 'wb') as f:
            pickle.dump({
                'word2idx': self.word2idx,
                'idx2word': self.idx2word,
                'vocab_size': self.vocab_size
            }, f)
    
    def load(self, path):
        with open(path, 'rb') as f:
            data = pickle.load(f)
            self.word2idx = data['word2idx']
            self.idx2word = data['idx2word']
            self.vocab_size = data['vocab_size']
            self.vocab_built = True


class WikiTextDataset(Dataset):
    def __init__(self, texts, tokenizer, seq_length=50):
        self.tokenizer = tokenizer
        self.seq_length = seq_length
        self.data = []
        
        for text in texts:
            if text.strip():
                encoded = tokenizer.encode(text)
                if len(encoded) > seq_length:
                    for i in range(0, len(encoded) - seq_length, seq_length // 2):
                        sequence = encoded[i:i + seq_length + 1]
                        if len(sequence) == seq_length + 1:
                            self.data.append(sequence)
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        sequence = self.data[idx]
        x = torch.tensor(sequence[:-1], dtype=torch.long)
        y = torch.tensor(sequence[1:], dtype=torch.long)
        return x, y


class WikiTextDataModule(pl.LightningDataModule):
    def __init__(self, vocab_size=10000, seq_length=50, batch_size=64, num_workers=4):
        super().__init__()
        self.vocab_size = vocab_size
        self.seq_length = seq_length
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.tokenizer = Tokenizer(vocab_size)
        
    def prepare_data(self):
        load_dataset("wikitext", "wikitext-2-raw-v1")
    
    def setup(self, stage=None):
        dataset = load_dataset("wikitext", "wikitext-2-raw-v1")
        
        if not self.tokenizer.vocab_built:
            self.tokenizer.build_vocab(dataset['train']['text'])
            os.makedirs('models', exist_ok=True)
            self.tokenizer.save('models/tokenizer.pkl')
        
        if stage == 'fit' or stage is None:
            self.train_dataset = WikiTextDataset(
                dataset['train']['text'], 
                self.tokenizer, 
                self.seq_length
            )
            self.val_dataset = WikiTextDataset(
                dataset['validation']['text'], 
                self.tokenizer, 
                self.seq_length
            )
        
        if stage == 'test' or stage is None:
            self.test_dataset = WikiTextDataset(
                dataset['test']['text'], 
                self.tokenizer, 
                self.seq_length
            )
    
    def train_dataloader(self):
        return DataLoader(
            self.train_dataset, 
            batch_size=self.batch_size, 
            shuffle=True, 
            num_workers=self.num_workers
        )
    
    def val_dataloader(self):
        return DataLoader(
            self.val_dataset, 
            batch_size=self.batch_size, 
            num_workers=self.num_workers
        )
    
    def test_dataloader(self):
        return DataLoader(
            self.test_dataset, 
            batch_size=self.batch_size, 
            num_workers=self.num_workers
        )