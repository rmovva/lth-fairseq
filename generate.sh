CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python fairseq_cli/generate.py /raid/wmt16_en_de_bpe32k/ --path /raid/checkpoints_to_avg/checkpoint74to78.pt --beam 4 --lenpen 0.6 --remove-bpe
