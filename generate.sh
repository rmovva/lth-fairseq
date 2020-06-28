CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python fairseq_cli/generate.py \
	/raj-learn/data/wmt16_en_de_bpe32k/ \
	--path /raj-learn/checkpoints/wmt16_en_de_big/checkpoint74to78.pt \
	--beam 4 --lenpen 0.6 --remove-bpe \
	--gen-subset test > /raj-learn/data/generated_translations/vaswani_big_gen.txt
bash scripts/compound_split_bleu.sh /raj-learn/data/generated_translations/vaswani_big_gen.txt
