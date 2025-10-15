"""
Next Word Prediction Package

This package contains:
- NextWordLSTM: LSTM-based language model
- WikiTextDataModule: Data handling with PyTorch Lightning
- Tokenizer: Text tokenization and vocabulary management
"""

from .model import NextWordLSTM
from .dataset import WikiTextDataModule, Tokenizer

__version__ = "1.0.0"
__author__ = "Ahmet Taha Berberoğlu"
__all__ = ["NextWordLSTM", "WikiTextDataModule", "Tokenizer"]