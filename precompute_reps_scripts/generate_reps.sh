python3 fairseq_cli/generate_word_reps_oneshot.py \
--path /raj-learn/checkpoints/lr-rewind_0.75sparsity_0.2frac_30epochs/checkpoint_LTH0_epoch60.pt /raj-learn/data/wmt16_en_de_bpe32k/ \
--source-lang en --target-lang de \
--tokenizer moses \
--bpe subword_nmt --bpe-codes /raj-learn/data/wmt16_en_de_bpe32k/bpe.32000 \
--max-tokens 4000 \
--cpu \
--input /raj-learn/data/probing_task_data/semantic_dependency/semantic_dependency_sentences.txt \
--outfile /raj-learn/data/precomputed_reps/LTH0/semantic_dependency.hdf5
