python trainer.py --embedding_type learned --vecDims 128 \
    --input raw_data/wikitext50m.txt --output_type indices \
    --num_heads 8 --num_layers 6 --window_size 64 --batch_size 192 \
    --epochs 100 --lr 0.0003 --lr_schedule plateau --model_file ./snapshot.pth --start_epoch 7 --resume
