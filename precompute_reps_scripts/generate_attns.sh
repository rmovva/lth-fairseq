python3 fairseq_cli/generate_attn_dists.py \
--path /raj-learn/checkpoints/lr-rewind_0.75sparsity_0.2frac_30epochs/checkpoint_LTH0_epoch60.pt /raj-learn/data/wmt16_en_de_bpe32k/ \
--source-lang en --target-lang de \
--tokenizer moses \
--bpe subword_nmt --bpe-codes /raj-learn/data/wmt16_en_de_bpe32k/bpe.32000 \
--max-tokens 4000 \
--input /raj-learn/data/probing_task_data/pos/wsj_sentences.txt \
--outfile /raj-learn/data/precomputed_attns/LTH0/ptb_pos.hdf5
