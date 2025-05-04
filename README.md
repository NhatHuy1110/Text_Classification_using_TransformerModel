# Text Classification using Transformer Model

- This repository demonstrates the implementation of a text classification model using Transformer architectures.
- The model leverages the power of multi-head attention mechanisms, position encodings, and transformer blocks to process and classify text data. 
- It is implemented in PyTorch and provides modularity for different components such as multi-head attention, transformer blocks, and custom attention mechanisms.
  
# Project Overview

This project applies various transformer-based techniques to classify text into different categories, such as positive or negative sentiment. The implementation uses several key building blocks from the Transformer model architecture, including multi-head attention, feedforward layers, and residual connections. The goal is to show how Transformer models can be adapted for efficient text classification tasks.

# Dataset

The dataset used for training the model is [IMDB dataset](https://www.kaggle.com/datasets/lakshmi25npathi/imdb-dataset-of-50k-movie-reviews?datasetId=134715&sortBy=dateRun&tab=profile) with 50,000 movie review for sentiment analysis.
- Consist of:
  + 25.000 movie review for training.
  + 25.000 movie review for testing.
- Label: Positive - Negative.

## File Structure

Text_Classification_using_TransformerModel/

├── MaskedAttention.py

├── MultiheadAttention.py

├── MultiheadCrossAttention.py

├── TextClassification.py

├── Transformer_Block.py

├── Text_Classification_using_Transformer2.ipynb

├── text-classification-using-transformer.ipynb

└── README.md

- `MaskedAttention.py`: This script demonstrates the application of masked multi-head attention, where the attention mechanism is restricted based on a causal mask.
- `MultiheadAttention.py`: This file defines a standard multi-head attention mechanism and demonstrates how to apply attention to a tensor of text sequences.
- `MultiheadCrossAttention.py`: This script implements multi-head cross-attention, where the attention weights are calculated between different sequences (cross-attention).
- `Transformer_Block.py`: This file defines a custom Transformer block which includes multi-head attention, feedforward layers, and layer normalization with residual connections.
- `Text_Classification_using_Transformer2.ipynb`: A Notebook for training a Transformer Model using 1-head self-attention.
- `text-classification-using-transformer.ipynb`: A Notebook for training a Transformer Model using only attention layer and remove positional embedding.

## Requirement

- You can run code on googlecollab, jupyter or Kaggle notebooks, ... If you don't have a virtual enviroment, manually include the nessesary libraries, like:
pytorch
torchtext
numpy
matplotlib
pandas
