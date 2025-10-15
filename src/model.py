import torch
import torch.nn as nn
import pytorch_lightning as pl


class NextWordLSTM(pl.LightningModule):
    def __init__(
        self, 
        vocab_size, 
        embedding_dim=256, 
        hidden_dim=512, 
        num_layers=2, 
        dropout=0.3,
        learning_rate=0.001
    ):
        super().__init__()
        self.save_hyperparameters()
        
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(
            embedding_dim, 
            hidden_dim, 
            num_layers, 
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_dim, vocab_size)
        
    def forward(self, x):
        embedded = self.embedding(x)
        lstm_out, _ = self.lstm(embedded)
        lstm_out = self.dropout(lstm_out)
        output = self.fc(lstm_out)
        return output
    
    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = nn.functional.cross_entropy(
            logits.view(-1, logits.size(-1)), 
            y.view(-1)
        )
        self.log('train_loss', loss, prog_bar=True)
        return loss
    
    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = nn.functional.cross_entropy(
            logits.view(-1, logits.size(-1)), 
            y.view(-1)
        )
        self.log('val_loss', loss, prog_bar=True)
        return loss
    
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.hparams.learning_rate)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, 
            mode='min', 
            factor=0.5, 
            patience=2
        )
        return {
            'optimizer': optimizer,
            'lr_scheduler': {
                'scheduler': scheduler,
                'monitor': 'val_loss'
            }
        }
    
    def predict_next_word(self, text, tokenizer, top_k=5):
        self.eval()
        with torch.no_grad():
            tokens = tokenizer.encode(text)
            x = torch.tensor([tokens], dtype=torch.long)
            
            logits = self(x)
            last_logits = logits[0, -1, :]
            
            probs = torch.softmax(last_logits, dim=0)
            top_probs, top_indices = torch.topk(probs, top_k)
            
            predictions = []
            for prob, idx in zip(top_probs, top_indices):
                word = tokenizer.idx2word.get(idx.item(), "<UNK>")
                predictions.append((word, prob.item()))
            
            return predictions
    
    def generate_text(self, seed_text, tokenizer, max_length=50, temperature=1.0):
        self.eval()
        with torch.no_grad():
            tokens = tokenizer.encode(seed_text)
            generated = tokens.copy()
            
            for _ in range(max_length):
                x = torch.tensor([generated], dtype=torch.long)
                logits = self(x)
                last_logits = logits[0, -1, :] / temperature
                
                probs = torch.softmax(last_logits, dim=0)
                next_token = torch.multinomial(probs, 1).item()
                
                if next_token == tokenizer.word2idx.get("<EOS>", 2):
                    break
                
                generated.append(next_token)
            
            return tokenizer.decode(generated)