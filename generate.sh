python fairseq_cli/generate.py \
	/raj-learn/data/wmt16_en_de_bpe32k/ \
	--cpu \
	--path /raj-learn/checkpoints/lr-rewind_0.75sparsity_0.2frac_30epochs/checkpoint_LTH9_epoch42_sparsity0.727.pt \
	--beam 4 --lenpen 0.6 --remove-bpe \
	--gen-subset test > /raj-learn/data/generated_translations/early_epoch_gen.txt
bash scripts/compound_split_bleu.sh /raj-learn/data/generated_translations/early_epoch_gen.txt
