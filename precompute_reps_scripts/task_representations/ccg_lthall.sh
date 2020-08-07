#!/bin/bash
inputpath=/raj-learn/data/probing_task_data/ccg/ccg_sentences.txt
outname=ccg_wsj_sentences.hdf5
#for str in LTH0 LTH1 LTH2 LTH3 LTH4 LTH5 LTH6 LTH7
for str in LTH8
do
    echo $str
    if [ ! -d /raj-learn/data/precomputed_reps/$str ]; then
	newdir=/raj-learn/data/precomputed_reps/$str
        mkdir $newdir
        echo 'created precomputed reps dir '$newdir
    fi
    if [ -f /raj-learn/data/precomputed_reps/$str/$outname ]; then
        echo 'already found precomputed reps file'
	continue
    fi 
    MODELPATH=$(ls /raj-learn/checkpoints/lr-rewind_0.75sparsity_0.2frac_30epochs/checkpoint_${str}_epoch60*.pt)
    python3 /raj-learn/lth-fairseq/fairseq_cli/generate_word_reps_oneshot.py \
    --path $MODELPATH /raj-learn/data/wmt16_en_de_bpe32k/ \
    --source-lang en --target-lang de \
    --tokenizer moses \
    --bpe subword_nmt --bpe-codes /raj-learn/data/wmt16_en_de_bpe32k/bpe.32000 \
    --max-tokens 4000 \
    --input $inputpath \
    --outfile /raj-learn/data/precomputed_reps/$str/$outname
done
