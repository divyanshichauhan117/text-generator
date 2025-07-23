from flask import Flask, render_template, request, jsonify
import torch
import torch.nn as nn
import torch.optim as optim
import random
import string
import numpy as np
from collections import Counter
import warnings
import os
warnings.filterwarnings('ignore')

app = Flask(__name__)

# Global variables for model and data
model = None
char_to_idx = {}
idx_to_char = {}
vocab_size = 0
sequence_length = 10

class SimpleTextGenerator(nn.Module):
    def __init__(self, vocab_size, embedding_dim=50, hidden_dim=128, num_layers=2):
        super(SimpleTextGenerator, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        
        # Embedding layer
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        
        # LSTM layer
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, num_layers, batch_first=True, dropout=0.3)
        
        # Output layer
        self.fc = nn.Linear(hidden_dim, vocab_size)
        self.dropout = nn.Dropout(0.5)
        
    def forward(self, x, hidden=None):
        # Embedding
        embedded = self.embedding(x)
        
        # LSTM
        lstm_out, hidden = self.lstm(embedded, hidden)
        
        # Take the last output
        lstm_out = self.dropout(lstm_out[:, -1, :])
        
        # Final layer
        output = self.fc(lstm_out)
        return output, hidden

def initialize_model():
    """Initialize and train the model"""
    global model, char_to_idx, idx_to_char, vocab_size
    
    # Sample training data
    sample_texts = [
        "The quick brown fox jumps over the lazy dog.",
        "Artificial intelligence is transforming the world in amazing ways.",
        "Once upon a time, there was a magical kingdom filled with wonder.",
        "Python is a powerful programming language for data science and AI.",
        "The future holds incredible possibilities for technology and humanity.",
        "Machine learning algorithms can learn patterns from data automatically.",
        "In a galaxy far far away, adventures await brave explorers.",
        "Deep learning networks process information in layers like the human brain.",
        "Creativity and innovation drive progress in science and technology.",
        "The ocean waves crash against the shore under the moonlit sky.",
        "Data science is revolutionizing how we understand complex problems.",
        "Natural language processing enables computers to understand human speech.",
        "Computer vision allows machines to interpret and analyze visual information.",
        "Robotics combines engineering with artificial intelligence for automation.",
        "Cloud computing provides scalable resources for modern applications."
    ]
    
    # Combine all texts
    full_text = " ".join(sample_texts).lower()
    
    # Create character vocabulary
    chars = sorted(list(set(full_text)))
    vocab_size = len(chars)
    
    # Create mappings
    char_to_idx = {ch: i for i, ch in enumerate(chars)}
    idx_to_char = {i: ch for i, ch in enumerate(chars)}
    
    # Create training sequences
    def create_sequences(text, seq_length=10):
        sequences = []
        targets = []
        
        for i in range(len(text) - seq_length):
            seq = text[i:i + seq_length]
            target = text[i + seq_length]
            
            seq_indices = [char_to_idx[ch] for ch in seq]
            target_idx = char_to_idx[target]
            
            sequences.append(seq_indices)
            targets.append(target_idx)
        
        return torch.tensor(sequences), torch.tensor(targets)
    
    X, y = create_sequences(full_text, sequence_length)
    
    # Initialize model
    model = SimpleTextGenerator(vocab_size)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.01)
    
    # Train the model
    model.train()
    epochs = 100
    batch_size = 32
    
    for epoch in range(epochs):
        total_loss = 0
        indices = torch.randperm(len(X))
        
        for i in range(0, len(X), batch_size):
            batch_indices = indices[i:i + batch_size]
            batch_X = X[batch_indices]
            batch_y = y[batch_indices]
            
            optimizer.zero_grad()
            outputs, _ = model(batch_X)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
    
    print(f"Model trained successfully! Vocabulary size: {vocab_size}")

def generate_text(seed_text, length=100, temperature=1.0):
    """Generate text using the trained model"""
    if model is None:
        return "Model not initialized"
    
    model.eval()
    
    # Prepare seed
    if len(seed_text) < sequence_length:
        seed_text = seed_text + " " * (sequence_length - len(seed_text))
    
    seed_text = seed_text.lower()
    generated = seed_text
    
    # Convert seed to indices
    current_seq = [char_to_idx.get(ch, 0) for ch in seed_text[-sequence_length:]]
    
    with torch.no_grad():
        for _ in range(length):
            x = torch.tensor([current_seq]).long()
            output, _ = model(x)
            
            # Apply temperature
            output = output / temperature
            probabilities = torch.softmax(output, dim=1)
            
            # Sample next character
            next_char_idx = torch.multinomial(probabilities, 1).item()
            next_char = idx_to_char[next_char_idx]
            
            generated += next_char
            current_seq = current_seq[1:] + [next_char_idx]
    
    return generated

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/generate', methods=['POST'])
def generate():
    try:
        data = request.get_json()
        prompt = data.get('prompt', '')
        length = int(data.get('length', 80))
        temperature = float(data.get('temperature', 1.0))
        
        # Validate inputs
        if not prompt:
            return jsonify({'error': 'Please provide a prompt'})
        
        if length < 10 or length > 500:
            return jsonify({'error': 'Length must be between 10 and 500'})
        
        if temperature < 0.1 or temperature > 2.0:
            return jsonify({'error': 'Temperature must be between 0.1 and 2.0'})
        
        # Generate text
        generated = generate_text(prompt, length, temperature)
        
        return jsonify({
            'success': True,
            'generated_text': generated,
            'prompt': prompt,
            'length': length,
            'temperature': temperature
        })
        
    except Exception as e:
        return jsonify({'error': f'Generation failed: {str(e)}'})

@app.route('/model_info')
def model_info():
    if model is None:
        return jsonify({'error': 'Model not initialized'})
    
    return jsonify({
        'vocab_size': vocab_size,
        'model_parameters': sum(p.numel() for p in model.parameters()),
        'sequence_length': sequence_length,
        'characters': ''.join(sorted(char_to_idx.keys()))
    })

@app.route('/health')
def health():
    return jsonify({'status': 'healthy', 'model_loaded': model is not None})

if __name__ == '__main__':
    print("Initializing model...")
    initialize_model()
    print("Starting Flask app...")
    app.run(debug=True, host='0.0.0.0', port=int(os.environ.get('PORT', 5000)))