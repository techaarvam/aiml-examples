python pth2onnx.py --embedding_type learned --vecDims 128 \
    --vocab_file vocab.json --output_type indices \
    --num_heads 8 --num_layers 6 --window_size 64 \
    --model_file transformer_model.pth
#   Add --quantize to also export an INT8 version
#   Add --output path/to/output.onnx to override the output path
