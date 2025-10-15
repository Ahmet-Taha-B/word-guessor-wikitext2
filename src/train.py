import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pytorch_lightning.loggers import TensorBoardLogger
from src.model import NextWordLSTM
from src.dataset import WikiTextDataModule
import torch


def train_model(
    vocab_size=10000,
    seq_length=50,
    batch_size=64,
    embedding_dim=256,
    hidden_dim=512,
    num_layers=2,
    dropout=0.3,
    learning_rate=0.001,
    max_epochs=20,
    num_workers=0
):
    pl.seed_everything(42)
    
    data_module = WikiTextDataModule(
        vocab_size=vocab_size,
        seq_length=seq_length,
        batch_size=batch_size,
        num_workers=num_workers
    )
    
    data_module.setup('fit')
    actual_vocab_size = len(data_module.tokenizer.word2idx)
    
    model = NextWordLSTM(
        vocab_size=actual_vocab_size,
        embedding_dim=embedding_dim,
        hidden_dim=hidden_dim,
        num_layers=num_layers,
        dropout=dropout,
        learning_rate=learning_rate
    )
    
    checkpoint_callback = ModelCheckpoint(
        dirpath='models/checkpoints',
        filename='next-word-{epoch:02d}-{val_loss:.2f}',
        monitor='val_loss',
        mode='min',
        save_top_k=3,
        save_last=True
    )
    
    early_stop_callback = EarlyStopping(
        monitor='val_loss',
        patience=5,
        mode='min'
    )
    
    logger = TensorBoardLogger('logs', name='next_word_model')
    
    trainer = pl.Trainer(
        max_epochs=max_epochs,
        callbacks=[checkpoint_callback, early_stop_callback],
        logger=logger,
        accelerator='auto',
        devices=1,
        log_every_n_steps=10,
        gradient_clip_val=1.0
    )
    
    trainer.fit(model, data_module)
    
    return model, data_module, trainer


if __name__ == '__main__':
    print("Starting training...")
    model, data_module, trainer = train_model(
        vocab_size=10000,
        seq_length=50,
        batch_size=64,
        max_epochs=20,
        num_workers=0
    )
    
    print("\nTraining complete!")
    print(f"Best model saved to: {trainer.checkpoint_callback.best_model_path}")
    
    test_text = "the president of the"
    predictions = model.predict_next_word(test_text, data_module.tokenizer, top_k=5)
    print(f"\nTest predictions for '{test_text}':")
    for word, prob in predictions:
        print(f"  {word}: {prob:.4f}")
    
    generated = model.generate_text(test_text, data_module.tokenizer, max_length=20)
    print(f"\nGenerated text: {generated}")