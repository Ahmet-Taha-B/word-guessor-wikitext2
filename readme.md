# 🤖 Next Word Prediction Model

A neural language model that predicts the next word in a sentence using LSTM architecture, trained on the Wikitext-2 dataset.

## ✨ Features

- **Next Word Prediction**: Get top predictions for the next word
- **Text Generation**: Generate creative text from seed phrases
- **Interactive Web Interface**: Built with Streamlit
- **Smart Filtering**: Automatically removes unknown tokens
- **Adjustable Creativity**: Control randomness with temperature slider

## 📊 Model Details

- **Architecture**: 2-layer LSTM
- **Parameters**: ~42M
- **Vocabulary**: 50,000 words
- **Training Dataset**: Wikitext-2
- **Framework**: PyTorch Lightning

## 🚀 Quick Start

### Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/word-guessor.git
cd word-guessor

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Download Model

Download the trained model and place it in:
- `models/tokenizer.pkl`
- `models/checkpoints/best_model.ckpt`

### Run the App

```bash
streamlit run app.py
```

Open your browser at `http://localhost:8501`

## 🎯 Usage Tips

- **Temperature 0.5-0.7**: Safe, predictable text
- **Temperature 0.8-1.2**: Balanced creativity
- **Temperature 1.3-2.0**: More creative, risky
- **Max words 20-30**: Best for coherent output

## 🛠️ Technologies

- PyTorch
- PyTorch Lightning
- Streamlit
- Hugging Face Datasets
- Python 3.8+

## 📁 Project Structure

```
word-guessor/
├── src/
│   ├── __init__.py
│   ├── model.py          # LSTM model
│   ├── dataset.py        # Data handling
│   └── train.py          # Training script
├── models/
│   ├── tokenizer.pkl
│   └── checkpoints/
│       └── best_model.ckpt
├── app.py                # Streamlit app
├── requirements.txt
└── README.md
```

## 📝 License

MIT License

## 👤 Author

Ahmet Taha Berberoğlu

---

Made with using PyTorch Lightning & Streamlit
