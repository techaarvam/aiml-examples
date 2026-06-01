import argparse

def parse_args():
    parser = argparse.ArgumentParser(description='Transformer Model Arguments')

    parser.add_argument('--num_heads', type=int, default=8, help='Number of attention heads')
    parser.add_argument('--num_layers', type=int, default=6, help='Number of transformer layers')
    parser.add_argument('--window_size', type=int, default=100, help='Window size for input sequences')
    parser.add_argument('--data_stride', type=int, default=1,
        help='Start-position stride between training windows. Default 1 preserves the old fully-overlapping dataset; set to window_size-1 for mostly non-overlapping next-token targets.')
    parser.add_argument('--batch_size', type=int, default=32, help='Training batch size')
    parser.add_argument('--epochs', type=int, default=10, help='Number of training epochs')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--optimizer', type=str, default='adam', choices=['adam', 'adam8', 'sgd'], help='Optimizer type')
    parser.add_argument('--qkv', type=str, default='fused', choices=['fused', 'unfused'], help='qkv fused/unfused. Also enabled flash attention')
    parser.add_argument('--data_dir', type=str, default='./data', help='Directory containing training data')
    parser.add_argument('--save_model', type=str, default='transformer_model.pth', help='Path to save trained model')
    parser.add_argument('--use_custom_norm', action='store_true', default=False, help='Use custom mean/std norm instead of nn.LayerNorm')
    parser.add_argument('--verbosity', type=int, default=1, help='Debug verbosity level (1=output, 2=error, 3=warn, 4=debug, 5=info)')
    parser.add_argument('--output_type', type=str, default='vecs', choices=['indices', 'vecs'], help='Output type: indices (linear projection to vocab, crossentropy loss) or vecs (linear projection to vecDims-1, MSE loss)')
    parser.add_argument('--input', type=str, default=None, help='Path to input text file (required for training)')
    parser.add_argument('--input_list', type=str, default=None, help='Comma-separated input files or path to a JSON file listing them. Files are cycled across epochs.')
    parser.add_argument('--tiktoken_encoding', type=str, default='cl100k_base', help='tiktoken encoding name (cl100k_base, p50k_base, o200k_base)')
    parser.add_argument('--seed', type=int, default=42, help='Random seed for reproducibility')
    parser.add_argument('--model_file', type=str, default=None, help='Path to load a saved model')
    parser.add_argument('--output_size', type=int, default=50, help='Number of tokens to generate during inference')
    parser.add_argument('--float_type', type=str, default='float32', choices=['float32', 'float16', 'bfloat16', 'float8'], help='Float precision type for model weights and computation')
    parser.add_argument('--lr_schedule', type=str, default='none', choices=['none', 'plateau', 'cosine'], help='LR scheduler: none, plateau (reduce on stall), cosine (anneal over epochs)')
    parser.add_argument('--infer_window_size', type=int, default=None, help='Context window size at inference time (defaults to window_size if not set)')
    parser.add_argument('--quantize', action='store_true', default=False, help='Quantize ONNX model to INT8 after export (pth2onnx only)')
    parser.add_argument('--vecDims', type=int, default=128, help='Word vector / embedding dimensions. default=128')
    parser.add_argument('--embedding_type', type=str, default='learned', choices=['glove-fixed', 'learned'], help='Embedding type: glove-fixed (pre-trained GloVe, frozen) or learned (random init, trained end-to-end)')
    parser.add_argument('--output', type=str, default=None, help='Output path for pth2onnx conversion (defaults to model_file with .onnx extension)')
    parser.add_argument('--resume', action='store_true', default=False, help='Resume training from --model_file checkpoint')
    parser.add_argument('--start_epoch', type=int, default=None, help='Manually override starting epoch when resuming (1-based). Required for old-format checkpoints.')
    parser.add_argument('--run_dir', type=str, default=None, help='Output directory for this run (default: runs/YYYYMMDD_HHMMSS). Set by runner.py.')
    parser.add_argument('--cache_file', type=str, default=None, help='Path to pickle cache of tokenized indices+vocab (skips word_tokenize on subsequent runs)')
    parser.add_argument('--validate', action='store_true', default=False, help='Validation mode: run forward pass over --input, print cross-entropy loss and perplexity, then exit')
    parser.add_argument('--grad_checkpoint', action='store_true', default=False,
        help='Enable gradient checkpointing per transformer layer (saves activation memory, costs ~30% extra compute)')
    parser.add_argument('--max_tokens', type=int, default=None,
        help='Cap dataset to first N tokens per shard (applied after cache load; cache always stores full data)')
    parser.add_argument('--sampler', type=str, default='top_p', choices=['min_p', 'top_k', 'top_p'],
        help='Sampling strategy for inference: top_p (default), min_p, top_k')
    parser.add_argument('--temperature', type=float, default=0.8,
        help='Sampling temperature — scales logits before sampling (default 0.8)')
    parser.add_argument('--top_k', type=int, default=50,
        help='k for top_k sampler (default 50)')
    parser.add_argument('--top_p', type=float, default=0.8,
        help='Nucleus probability for top_p sampler (default 0.8)')
    parser.add_argument('--min_p', type=float, default=0.05,
        help='Min-p threshold: fraction of top-token prob below which tokens are dropped (default 0.05)')
    parser.add_argument('--inner_dims', type=int, default=None,
        help='Expanded inner transformer dimension (d’). When set, frozen embedding/output adapters bridge vecDims→inner_dims. Use extend_dims.py to create the starting checkpoint.')
    parser.add_argument('--mlp_boost_old_d', type=int, default=0,
        help='Previous inner_dims before the most recent expansion. When > 0: gradient hooks multiply new-dim gradients by --mlp_boost to accelerate warm-up. Set to 0 to disable.')
    parser.add_argument('--mlp_boost', type=float, default=4.0,
        help='Gradient multiplier applied to new MLP dims when --mlp_boost_old_d is active (default 4.0).')
    parser.add_argument('--lr_warmup_target', type=float, default=0.0,
        help='ending LR after warmup. 10% of the epoch is the duration for the warmup hard coded currently')
    parser.add_argument('--reset_optimizer_every_epoch', action='store_true', default=False,
        help='Discard optimizer state at the start of each epoch after the first. Cold Adam restart each epoch.')
    parser.add_argument('--reset_adam_v_every_epoch', action='store_true', default=False,
        help='Zero exp_avg_sq (v_t) at the start of each epoch after the first, keeping exp_avg (m_t). Clears stale scale estimates while preserving directional momentum.')
    parser.add_argument('--dataloader_workers', type=int, default=2,
        help='Number of DataLoader worker processes for background batch prefetching (default 2)')
    parser.add_argument('--prefetch_factor', type=int, default=4,
        help='Batches to prefetch per DataLoader worker (default 4)')
    parser.add_argument('--shared_checkpoint_dir', type=str, default=None,
        help='If set, save model-only checkpoint here every --shared_checkpoint_every epochs (overwrites). Used by run_progressive.sh to bridge stages.')
    parser.add_argument('--shared_checkpoint_every', type=int, default=5,
        help='Save to --shared_checkpoint_dir every N epochs (default 5)')
    parser.add_argument('--entropy_csv', type=str, default=None,
        help='If set, append spectral entropy row to this CSV every --shared_checkpoint_every epochs.')
    return parser.parse_args()

args = parse_args()
