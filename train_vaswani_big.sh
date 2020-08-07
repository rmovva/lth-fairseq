#!/bin/bash
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python fairseq_cli/train.py  /raid/data/wmt16_en_de_bpe32k/     --arch transformer_vaswani_wmt_en_de_big    --share-all-embeddings     --optimizer adam --adam-betas '(0.9, 0.98)' --clip-norm 0.0     --lr 0.0007 --lr-scheduler inverse_sqrt --warmup-updates 4000 --warmup-init-lr 1e-07     --dropout 0.3 --weight-decay 0.0     --criterion label_smoothed_cross_entropy --label-smoothing 0.1     --max-tokens 10000 --save-dir /raid/checkpoints/ --max-update 250000 --save-interval-updates 0 --tensorboard-logdir /raj-learn/logs/wmt16_en_de_big/ --no-progress-bar --log-interval 500 --num-workers 32 --lth-rewind-iter 1000 --disable-validation

cp /raid/checkpoints/checkpoint_last.pt /raj-learn/checkpoints/wmt16_en_de_big/
for i in {120..130}
do
    cp /raid/checkpoints/checkpoint$i.pt /raj-learn/checkpoints/wmt16_en_de_big/
done
