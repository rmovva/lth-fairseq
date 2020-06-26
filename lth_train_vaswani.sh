#!/bin/bash
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python fairseq_cli/train_lth.py  /raid/data/wmt16_en_de_bpe32k/ --save-dir /raid/checkpoints/ --arch pruned_transformer_vaswani_wmt_en_de_big --share-all-embeddings --optimizer adam --adam-betas '(0.9, 0.98)' --clip-norm 0.0 --lr 0.0007 --lr-scheduler inverse_sqrt --warmup-updates 4000 --warmup-init-lr 1e-07 --dropout 0.3 --weight-decay 0.0 --criterion label_smoothed_cross_entropy --label-smoothing 0.1 --max-tokens 10000 --max-epoch 2 --save-interval-updates 0 --tensorboard-logdir /raj-learn/logs/wmt16_en_de_big/ --no-progress-bar --log-interval 500 --num-workers 32 --disable-validation --final_sparsity 0.5 --n_lth_iterations 4
