## Project Overview

This is a deep learning project that focuses on building a neural machine translation (NMT) system specifically designed for Japanese-Vietnamese language pairs. The system uses the Transformer architecture as introduced in the "Attention Is All You Need" paper, with custom modifications to fit the specific characteristics of these languages.

## Key Components

### Data Processing
- Custom tokenization using SentencePiece models (separate models for Japanese and Vietnamese)
- Dataset creation with appropriate padding and attention masks
- Train/validation split management

### Model Architecture
- Transformer-based encoder-decoder architecture
- Positional encoding for sequence information
- Custom vocabulary sizes (2500 for Japanese, 900 for Vietnamese)
- Optimized hyperparameters for this specific language pair

### Training Pipeline
- AdamW optimizer with learning rate scheduling
- Label smoothing and gradient accumulation
- BLEU score evaluation during training
- Checkpoint saving for model persistence

### Analysis Tools
- Real-time visualization of training metrics
- BLEU score calculation and tracking
- Custom analysis of translation quality

## Training Performance

The model was trained for 10 epochs, showing consistent improvement in both loss metrics and BLEU scores:
- Starting with a training loss of ~5.79 and validation loss of ~5.04
- Finishing with a training loss of ~1.68 and validation loss of ~1.52
- BLEU score improved from near zero to 0.6429 by the final epoch

## Files in the Project

- preprocesstest.py: Handles data preprocessing and tokenization
- translator.py: Contains the main model architecture and training logic
- analysis.py: Provides tools for analyzing translation performance
- redraw_bleu.py: Utility script to regenerate BLEU score visualizations
- training_log.txt: Log of the training process with metrics for each epoch
- plots: Directory containing visualization outputs

## Results

The training process demonstrated steady improvement in translation quality, with the BLEU score increasing from effectively zero to over 0.64. The model achieves reasonable translation quality between Japanese and Vietnamese, two languages with significantly different structures.

## Training Duration
The complete training process took approximately 1.63 hours, with individual epochs ranging from 8.3 to 13.8 minutes.

## Next Steps
Potential improvements could include:

Exploring larger vocabulary sizes
Testing different model architectures or hyperparameters
Implementing techniques like back-translation for data augmentation
Fine-tuning on domain-specific data

## Requirements

- PyTorch
- SentencePiece
- NLTK
- Matplotlib
- Pandas
- scikit-learn

## Usage

The entire training pipeline can be executed by running:
```bash
python translator.py
```

For analysis of results after training:
```bash
python analysis.py
```

To regenerate BLEU score visualization:
```bash
python redraw_bleu.py
```
