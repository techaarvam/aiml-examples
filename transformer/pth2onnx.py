
# --------------------------------------------------
# Tech Aarvam
# Copyright (c) 2026 Tech Aarvam.
# Author: Ram (Ramasubramanian B)
# --------------------------------------------------
# Converts a saved .pth checkpoint to ONNX format.
# Usage:
#   python pth2onnx.py --model_file transformer_model.pth \
#                      --vocab_file vocab.json --num_heads 8 --num_layers 3 \
#                      --window_size 64 --output_type indices
#   Add --quantize to also produce an INT8 quantized version
# --------------------------------------------------

import os
import torch
import DataInput
import common
import multihead
from argParser import *
from debug import *

set_verbosity(args.verbosity)

if not args.model_file:
    raise ValueError("--model_file is required for pth2onnx conversion")

output_path = args.output if hasattr(args, 'output') and args.output else args.model_file.replace('.pth', '.onnx').replace('.pt', '.onnx')

common.dtype = {'float32': torch.float32, 'float16': torch.float16,
                'bfloat16': torch.bfloat16, 'float8': torch.float8_e4m3fn}[args.float_type]

dIn = DataInput.DataInput()

transformer = multihead.MultiHead().to(common.device)
checkpoint = torch.load(args.model_file, map_location=common.device)
if isinstance(checkpoint, dict) and 'model' in checkpoint:
    state_dict = {k.replace('_orig_mod.', ''): v for k, v in checkpoint['model'].items()}
    transformer.load_state_dict(state_dict)
    dbg_output(f"Checkpoint epoch: {checkpoint.get('epoch', '?')}")
else:
    state_dict = {k.replace('_orig_mod.', ''): v for k, v in checkpoint.items()}
    transformer.load_state_dict(state_dict)
transformer = transformer.to(common.dtype)
transformer.eval()

dbg_output(f"Loaded model from {args.model_file}")

if args.embedding_type == "glove-fixed":
    dummy = torch.randn(1, args.window_size - 1, common.vecDims, dtype=common.dtype, device=common.device)
else:
    dummy = torch.zeros(1, args.window_size - 1, dtype=torch.long, device=common.device)

torch.onnx.export(
    transformer,
    dummy,
    output_path,
    input_names=["input"],
    output_names=["logits"],
    dynamic_axes={
        "input":  {0: "batch_size"},
        "logits": {0: "batch_size"},
    },
    opset_version=17,
    dynamo=False,
)

dbg_output(f"ONNX model saved to {output_path}")
if args.embedding_type == "glove-fixed":
    dbg_output(f"Input shape:  [batch, seq_len={args.window_size-1}, vecDims={common.vecDims}] (float)")
else:
    dbg_output(f"Input shape:  [batch, seq_len={args.window_size-1}] (long indices)")
dbg_output(f"Output shape: [batch, seq_len={args.window_size-1}, vocab={common.vocabSize}]")
dbg_output(f"Size: {os.path.getsize(output_path) / 1024 / 1024:.1f} MB")

if args.quantize:
    from onnxruntime.quantization import quantize_dynamic, QuantType
    quant_path = output_path.replace('.onnx', '_int8.onnx')
    quantize_dynamic(
        model_input=output_path,
        model_output=quant_path,
        weight_type=QuantType.QInt8,
    )
    dbg_output(f"INT8 quantized model saved to {quant_path}")
    dbg_output(f"Size: {os.path.getsize(quant_path) / 1024 / 1024:.1f} MB")
