#!/bin/bash
CUDA_VISIBLE_DEVICES=0 python fairseq_cli/train_lth.py  /tmp/data/wmt16_en_de_bpe32k/ --save-dir /tmp/checkpoints/ --arch pruned_transformer_vaswani_wmt_en_de_big --share-all-embeddings --optimizer adam --adam-betas '(0.9, 0.98)' --clip-norm 0.0 --lr 0.0007 --lr-scheduler inverse_sqrt --warmup-updates 4000 --warmup-init-lr 1e-07 --dropout 0.3 --weight-decay 0.0 --criterion label_smoothed_cross_entropy --label-smoothing 0.1 --max-tokens 10000 --max-epoch 3 --save-interval-updates 0 --tensorboard-logdir /raj-learn/logs/wmt16_en_de_big/ --no-progress-bar --log-interval 500 --num-workers 32 --disable-validation --final_sparsity 0.2 --n_lth_iterations 1
