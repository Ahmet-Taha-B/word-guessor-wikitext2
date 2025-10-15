import streamlit as st
import torch
from src.model import NextWordLSTM
from src.dataset import Tokenizer
from huggingface_hub import hf_hub_download
import re

REPO_ID = "AhmetTaha07/next-word-predictor-wikitext2"

@st.cache_resource
def load_model():
    try:
        tokenizer_path = hf_hub_download(
            repo_id=REPO_ID,
            filename="tokenizer.pkl"
        )
        
        model_path = hf_hub_download(
            repo_id=REPO_ID,
            filename="best_model.ckpt"
        )
        
        tokenizer = Tokenizer()
        tokenizer.load(tokenizer_path)
        
        model = NextWordLSTM.load_from_checkpoint(
            model_path,
            vocab_size=len(tokenizer.word2idx)
        )
        model.eval()
        
        return model, tokenizer
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        raise

def fix_text_spacing(text):
    text = re.sub(r'\s+([.,!?;:])', r'\1', text)
    text = re.sub(r'\s+(n\'t|\'re|\'ve|\'ll|\'d|\'m|\'s)', r'\1', text)
    text = re.sub(r'(\w)\s+\'(\w)', r"\1'\2", text)
    text = re.sub(r'@-@', '-', text)
    text = re.sub(r'=\s*=\s*=', '', text)
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

def predict_next_word_filtered(model, text, tokenizer, top_k=5):
    predictions = model.predict_next_word(text, tokenizer, top_k=top_k * 3)
    filtered = [(word, prob) for word, prob in predictions if word != "<UNK>"]
    return filtered[:top_k]

def generate_text_filtered(model, seed_text, tokenizer, max_length=50, temperature=1.0):
    model.eval()
    with torch.no_grad():
        tokens = tokenizer.encode(seed_text)
        generated = tokens.copy()
        
        for _ in range(max_length):
            x = torch.tensor([generated], dtype=torch.long)
            logits = model(x)
            last_logits = logits[0, -1, :] / temperature
            
            unk_idx = tokenizer.word2idx.get("<UNK>", 1)
            last_logits[unk_idx] = float('-inf')
            
            probs = torch.softmax(last_logits, dim=0)
            next_token = torch.multinomial(probs, 1).item()
            
            if next_token == tokenizer.word2idx.get("<EOS>", 2):
                break
            
            generated.append(next_token)
        
        decoded = tokenizer.decode(generated)
        return fix_text_spacing(decoded)

st.set_page_config(page_title="Next Word Predictor", page_icon="🤖", layout="wide")

st.title("🤖 Next Word Prediction Model")
st.markdown("Enter some text and let the AI predict what comes next!")

model, tokenizer = load_model()

tab1, tab2 = st.tabs(["📝 Next Word Prediction", "✨ Text Generation"])

with tab1:
    st.header("Predict Next Word")
    
    input_text = st.text_input(
        "Enter your text:",
        value="the president of the",
        help="Type some words and the model will predict what comes next"
    )
    
    top_k = st.slider("Number of predictions:", min_value=1, max_value=10, value=5)
    
    if st.button("🔮 Predict", type="primary"):
        if input_text.strip():
            with st.spinner("Thinking..."):
                predictions = predict_next_word_filtered(model, input_text, tokenizer, top_k=top_k)
            
            if predictions:
                st.success("Predictions:")
                
                cols = st.columns(2)
                for i, (word, prob) in enumerate(predictions):
                    col_idx = i % 2
                    word_display = fix_text_spacing(f"{input_text} {word}")
                    next_word = word_display.split()[-1]
                    with cols[col_idx]:
                        st.metric(
                            label=f"#{i+1}",
                            value=next_word,
                            delta=f"{prob*100:.2f}%"
                        )
            else:
                st.warning("No valid predictions found. Try different text!")
        else:
            st.warning("Please enter some text!")

with tab2:
    st.header("Generate Text")
    
    seed_text = st.text_input(
        "Seed text:",
        value="once upon a time",
        help="Start with a few words and let the model continue the story"
    )
    
    col1, col2 = st.columns(2)
    with col1:
        max_length = st.slider("Maximum words to generate:", min_value=10, max_value=100, value=50)
    with col2:
        temperature = st.slider(
            "Creativity (temperature):", 
            min_value=0.1, 
            max_value=2.0, 
            value=1.0,
            help="Lower = more predictable, Higher = more creative"
        )
    
    if st.button("✨ Generate", type="primary"):
        if seed_text.strip():
            with st.spinner("Generating..."):
                generated = generate_text_filtered(
                    model,
                    seed_text, 
                    tokenizer, 
                    max_length=max_length, 
                    temperature=temperature
                )
            
            st.success("Generated text:")
            st.markdown(f"### {generated}")
            
            word_count = len(generated.split())
            st.info(f"📊 Generated {word_count} words total")
        else:
            st.warning("Please enter seed text!")

st.sidebar.title("ℹ️ About")
st.sidebar.info(
    f"""
    This model was trained on the **Wikitext-2** dataset using an LSTM neural network.
    
    **Model Details:**
    - {sum(p.numel() for p in model.parameters()):,} parameters
    - 2-layer LSTM
    - Vocabulary: {len(tokenizer.word2idx):,} words
    
    **Tips:**
    - Try different seed texts
    - Adjust temperature for creativity
    - Lower temperature = safer predictions
    - Higher temperature = more variety
    
    Note: Unknown words are automatically filtered out!
    """
)

st.sidebar.markdown("---")
st.sidebar.markdown("Made with ❤️ using PyTorch Lightning & Streamlit")
