#  AI Text Generator Web App

A simple but powerful neural network-based text generator built with Flask and PyTorch. This application uses an LSTM (Long Short-Term Memory) neural network to generate creative text based on user prompts.

##  Features

- **Neural Network Text Generation**: Uses LSTM architecture for character-level text generation
- **Interactive Web Interface**: Beautiful, responsive design with real-time controls
- **Customizable Parameters**: 
  - Adjustable text length (20-300 characters)
  - Temperature control for creativity (0.1-2.0)
- **Example Prompts**: Quick-start examples to try different text generation styles
- **Model Statistics**: Real-time display of model information
- **Mobile Responsive**: Works perfectly on all devices

##  How It Works

### Model Architecture

The text generator uses a simple but effective LSTM neural network:

- **Embedding Layer**: Converts characters to dense vectors (50 dimensions)
- **LSTM Layers**: 2 layers with 128 hidden units each, includes dropout (0.3)
- **Output Layer**: Maps hidden states back to character probabilities
- **Total Parameters**: ~100K parameters for efficient training

### Training Process

1. **Data Preparation**: Uses sample texts covering various domains (AI, science, literature)
2. **Sequence Creation**: Creates character-level sequences of length 10
3. **Training**: 100 epochs with Adam optimizer (learning rate: 0.01)
4. **Character-level Generation**: Predicts next character based on previous sequence

### Text Generation

The model generates text using:
- **Temperature Sampling**: Controls randomness/creativity
- **Sliding Window**: Maintains context using the last 10 characters
- **Character-by-character**: Builds text one character at a time

### Best Practices

- **Start with meaningful prompts**: "The future of" works better than "xyz"
- **Use appropriate length**: 50-150 characters often give best results
- **Experiment with temperature**: Try different values for various effects
- **Context matters**: The model performs better with coherent starting text


### Dependencies

- **Flask 2.3.3**: Web framework
- **PyTorch 2.0.1**: Neural network framework
- **NumPy 1.24.3**: Numerical computations
- **Gunicorn 21.2.0**: Production WSGI server

### API Endpoints

- `GET /`: Main web interface
- `POST /generate`: Text generation endpoint
- `GET /model_info`: Model statistics
- `GET /health`: Health check endpoint

### Model Training

The model is trained on startup with a curated dataset including:
- Technical content (AI, programming, data science)
- Creative content (stories, descriptions)
- General knowledge content

