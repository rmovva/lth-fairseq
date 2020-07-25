#!/bin/bash
python fairseq_cli/train_lth.py /raid/data/wmt16_en_de_bpe32k/ \
	--save-dir /raid/checkpoints/ \
	--restore-file /raid/checkpoints/checkpoint_LTH0_epoch60.pt \
	--arch pruned_transformer_vaswani_wmt_en_de_big \
	--share-all-embeddings \
	--prune-embeddings \
	--optimizer adam \
	--adam-betas '(0.9, 0.98)' \
	--clip-norm 0.0 \
	--lr 0.0007 \
	--lr-scheduler inverse_sqrt \
	--warmup-updates 4000 \
	--warmup-init-lr 1e-07 \
	--dropout 0.3 \
	--weight-decay 0.0 \
	--criterion label_smoothed_cross_entropy \
	--label-smoothing 0.1 \
	--max-tokens 10000 \
	--max-epoch 60 \
	--save-interval 1 \
	--save-interval-updates 0 \
	--no-progress-bar \
	--log-interval 500 \
	--num-workers 32 \
	--disable-validation \
	--final_sparsity 0.75 \
	--n_lth_iterations 6 \
	--lr-rewind
