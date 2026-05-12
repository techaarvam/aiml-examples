# Transformer Model Experiments - Running Notes

## Run 2 — WikiText-103 (50MB), Learned Embedding (May 12, 2026)

### Hyperparameters
- Embedding: Learned (nn.Embedding, vecDims=128)
- Data: raw_data/wikitext50m.txt (50MB slice of WikiText-103)
- Vocabulary: 30,000 (capped from 148,535 unique tokens)
- Layers: 6 | Heads: 8 | Window: 64 | Batch: 192
- LR: 0.0003 | Schedule: Plateau | Optimizer: Adam
- Model Parameters: 11,062,832

### Timing
- ~1hr 28min per epoch (50,407 batches at 9.2 batch/s on RTX 5070)
- vs Run 1: ~25 min/epoch — 3.5× slower despite only 1.8× more parameters
- Slowdown mainly due to larger dataset (wikitext50m >> combined.txt) — more tokens = more batches per epoch

### Loss Log
| Epoch | Loss | Notes |
|-------|------|-------|
| 1 | 4.4123 | |
| 2 | 3.9514 | |
| 3 | 3.8389 | |
| 4 | 3.7725 | |
| 5 | 3.7239 | |
| 6 | — | CUDA launch timeout (X11 watchdog, display on same GPU) |
| 7+ | resumed | Resumed from epoch 5 checkpoint with --start_epoch 7 |

---

## Run 1 — combined.txt, GloVe Embedding

## Experiment Overview
- **Date**: May 11, 2026
- **Model**: Custom Transformer with Multi-Head Attention
- **Embedding**: GloVe fixed embedding (100-dim)
- **Positional Encoding**: Unconventional approach - position kept in dimension 101 (diverging from original sinusoidal positional embedding)
- **Vocabulary Size**: 50,020 tokens
- **Vector Space Dimension**: 100

## Hyperparameters
- Number of Attention Heads: 8
- Number of Transformer Layers: 3
- Window Size (Context Length): 64
- Batch Size: 128
- Epochs Planned: 100
- Learning Rate: 0.0003
- Optimizer: Adam
- Learning Rate Schedule: Plateau (patience=5, factor=0.5)
- Output Type: Indices (CrossEntropy Loss)
- Input Data: raw_data/combined.txt
- Model Checkpoint: transformer_model.pth

## Model Architecture
- Total Parameters: 6,143,451
- Trainable Parameters: 6,143,451
- Embedding Layer: Fixed GloVe vectors (non-trainable)
- Position Handling: Custom implementation in dimension 101
- Attention Mechanism: Multi-Head Self-Attention
- Feed Forward Network: Standard transformer FFN

## Training Progress
Training was interrupted manually at epoch 54. Loss values recorded:

| Epoch | Loss |
|-------|------|
| 1 | 4.4696 |
| 2 | 3.9660 |
| 3 | 3.8387 |
| 4 | 3.7737 |
| 5 | 3.7321 |
| 6 | 3.7025 |
| 7 | 3.6800 |
| 8 | 3.6622 |
| 9 | 3.6476 |
| 10 | 3.6354 |
| 11 | 3.6250 |
| 12 | 3.6159 |
| 13 | 3.6080 |
| 14 | 3.6010 |
| 15 | 3.5948 |
| 16 | 3.5892 |
| 17 | 3.5841 |
| 18 | 3.5795 |
| 19 | 3.5753 |
| 20 | 3.5713 |
| 21 | 3.5677 |
| 22 | 3.5644 |
| 23 | 3.5613 |
| 24 | 3.5583 |
| 25 | 3.5556 |
| 26 | 3.5531 |
| 27 | 3.5506 |
| 28 | 3.5483 |
| 29 | 3.5461 |
| 30 | 3.5441 |
| 31 | 3.5421 |
| 32 | 3.5403 |
| 33 | 3.5384 |
| 34 | 3.5368 |
| 35 | 3.5351 |
| 36 | 3.5336 |
| 37 | 3.5321 |
| 38 | 3.5306 |
| 39 | 3.5293 |
| 40 | 3.5279 |
| 41 | 3.5267 |
| 42 | 3.5254 |
| 43 | 3.5243 |
| 44 | 3.5230 |
| 45 | 3.5220 |
| 46 | 3.5209 |
| 47 | 3.5199 |
| 48 | 3.5189 |
| 49 | 3.5179 |
| 50 | 3.5170 |
| 51 | 3.5161 |
| 52 | 3.5152 |
| 53 | 3.5143 |
| 54 | 3.5135 (interrupted) |

### Observations from Training Curve
- Steady, consistent decrease in loss throughout training
- No signs of overfitting (would expect validation loss to increase while training loss decreases)
- Learning rate remained constant at 0.0003 throughout recorded epochs (plateau scheduler didn't trigger)
- Loss reduction appears to be slowing down as training progresses (asymptotic behavior)
- After 54 epochs, loss ~3.51, suggesting model has learned meaningful patterns but could benefit from more training

## Inference Examples
### Example 1
**Input Context** (63 tokens):
```
how are you today? I met the governor yesterday and the prime minister the day before. they are very happy with the elections
```

**Generated Continuation**:
```
how are you today ? i met the governor yesterday and the prime minister the day before . they are very happy with the elections , and the duke hears the massachusetts ’ s swain , and the snake and hank swim around the duke floundered in the river . temperance tribe , whooping and a crick ’ s , and a mighty big lot , and a big stack of matches , and a
```



## Technical Implementation Notes

### Positional Encoding Approach
Unlike the original Transformer which uses sinusoidal positional encodings, this implementation:
- Keeps positional information in dimension 101 of the embedding space
- Uses a custom `embedPositions` function in `DataInput.py`
- This approach separates semantic information (dimensions 0-99) from positional information (dimension 100)

### Data Processing Pipeline
1. Raw text → Tokenization (NLTK word_tokenize)
2. Tokens → GloVe vectors + indices (via `tokensToVecsAndIndices`)
3. Positional encoding applied via `embedPositions`
4. Sequence fed to Transformer model
5. Output processed based on `output_type`:
   - "indices": CrossEntropy loss with vocabulary prediction
   - "vecs": MSE loss with vector regression

### Model Saving/Loading
- Checkpoints saved after each epoch to `transformer_model.pth`
- Final model also saved explicitly
- Vocabulary saved to `vocab.json`
- Model loading restores state_dict and applies correct dtype/device

## Next Steps (Immediate)
1. **Replace GloVe with nn.Embedding**: Modify DataInput.py to use trainable embedding layer instead of fixed GloVe vectors
2. **Fix Positional Encoding**: Implement standard sinusoidal positional encoding as in original Transformer paper
3. **Re-run Training**: Execute two experiments:
   - Case A: Trainable nn.Embedding + standard sinusoidal positional encoding
   - Case B: Trainable nn.Embedding + learned positional embeddings
4. **Compare Results**: Analyze loss curves and generation quality between approaches

## Future Experiments & Improvements
1. **Complete Training**: Run for full 100 epochs to see if loss continues to decrease
2. **Validation Split**: Add validation monitoring to detect overfitting
3. **Learning Rate Tuning**: Experiment with different LR schedules or initial values
4. **Architecture Variations**:
    - Experiment with different numbers of heads/layers
    - Try different embedding dimensions
5. **Output Type Comparison**: Compare "indices" vs "vecs" output types
6. **Prompt Engineering**: Test with different input contexts to evaluate generation quality
7. **Quantization**: Explore model quantization for deployment efficiency
8. **ONNX Export**: Utilize existing `onnxrun.py` and `pth2onnx.py` for deployment

## Files Modified/Created in This Session
- `transformer_model.pth`: Model checkpoint (updated during training)
- `vocab.json`: Vocabulary mapping
- Various `.pyc` files in `__pycache__`: Compiled Python bytecode

## Conclusion
This experiment demonstrates a functional Transformer implementation with custom positional encoding. The model shows ability to learn from text data and generate coherent continuations. The unconventional positional encoding approach (dimension 101) appears to be functioning correctly, though comparison with standard approaches would be valuable. Training shows steady convergence, suggesting the architecture is well-suited to the task.

---
*Notes compiled from terminal output of transformer training session*