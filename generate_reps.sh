python3 fairseq_cli/generate_word_reps.py \
--path /raj-learn/checkpoints/lr-rewind_0.75sparsity_0.2frac_30epochs/checkpoint_LTH0_epoch60.pt /raj-learn/data/wmt16_en_de_bpe32k/ \
--source-lang en --target-lang de \
--tokenizer moses \
--bpe subword_nmt --bpe-codes /raj-learn/data/wmt16_en_de_bpe32k/bpe.32000 \
--max-tokens 4000 \
--input /raj-learn/data/probing_data/ccg/ccg_sentences_head100.txt

