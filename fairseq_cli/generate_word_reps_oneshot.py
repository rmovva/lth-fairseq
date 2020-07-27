#!/usr/bin/env python3 -u
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""
Translate raw text with a trained model. Batches data on-the-fly.
"""

from collections import namedtuple
import fileinput
import logging
import math
import sys
import os

import torch
import numpy as np
import h5py
import json

from fairseq import checkpoint_utils, options, tasks, utils
from fairseq.data import encoders
from fairseq.models import FairseqEncoder


logging.basicConfig(
    format='%(asctime)s | %(levelname)s | %(name)s | %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
    level=logging.INFO,
    stream=sys.stdout,
)
logger = logging.getLogger('fairseq_cli.interactive')


Batch = namedtuple('Batch', 'ids src_tokens src_lengths')
Translation = namedtuple('Translation', 'src_str hypos pos_scores alignments')


def make_hdf5_file(sentence_to_index, vectors, output_file_path):
    '''
    From Nelson Liu's contextual-repr-analysis repo.
    Creates hdf5 file in correct format for input to probing training.
    https://github.com/nelson-liu/contextual-repr-analysis
    '''
    with h5py.File(output_file_path, 'w') as fout:
        for key, embeddings in vectors.items():
            fout.create_dataset(
                str(key),
                embeddings.shape, dtype='float32',
                data=embeddings)
        sentence_index_dataset = fout.create_dataset(
            "sentence_to_index",
            (1,),
            dtype=h5py.special_dtype(vlen=str))
        sentence_index_dataset[0] = json.dumps(sentence_to_index)


def make_tokens(lines, task, encode_fn):
    encoded_inputs = []
    indices = []
    for src_str in lines:
        x, idxs = encode_fn(src_str)
        encoded_inputs.append(x)
        indices.append(idxs)
    tokens = [
        task.source_dictionary.encode_line(
            encoded_input, add_if_not_exist=False
        ).long()
        for encoded_input in encoded_inputs
    ]
    return tokens, indices


def make_batches(tokens, args, task, max_positions):
    # tokens = [
    #     task.source_dictionary.encode_line(
    #         encode_fn(src_str), add_if_not_exist=False
    #     ).long()
    #     for src_str in lines
    # ]
    lengths = [t.numel() for t in tokens]
    itr = task.get_batch_iterator(
        dataset=task.build_dataset_for_inference(tokens, lengths),
        max_tokens=args.max_tokens,
        max_sentences=args.max_sentences,
        max_positions=max_positions,
        ignore_invalid_inputs=args.skip_invalid_size_inputs_valid_test
    ).next_epoch_itr(shuffle=False)
    for batch in itr:
        yield Batch(
            ids=batch['id'],
            src_tokens=batch['net_input']['src_tokens'], src_lengths=batch['net_input']['src_lengths'],
        )


def map_rep_to_sentence(rep, indices, token_length):
    '''
    rep: MaxTokens x Dim
    indices: index mapping each token to a word index in the original sentence
    tokenlength: value corresponding to how many tokens the encoded input is, w/o padding 
    '''
    # get rid of padding
    rep = rep[-token_length :]
    # number of tokens in sentence: max index + 1 in index tracker
    sentence_length = max(indices) + 1

    sentence_rep = np.zeros((sentence_length, rep.shape[1]), dtype=np.float32)
    count_tokens_per_idx = {}
    for (i, map_idx) in enumerate(indices):
        sentence_rep[map_idx] += rep[i]
        if map_idx not in count_tokens_per_idx:
            count_tokens_per_idx[map_idx] = 0
        count_tokens_per_idx[map_idx] += 1

    # normalize for number of subtokens per word; i.e. take mean over BPE subtoken representations
    for map_idx in count_tokens_per_idx:
        sentence_rep[map_idx] *= (1.0 / count_tokens_per_idx[map_idx])

    return sentence_rep


def main(args):
    utils.import_user_module(args)

    if args.max_tokens is None and args.max_sentences is None:
        args.max_sentences = 1

    logger.info(args)

    use_cuda = torch.cuda.is_available() and not args.cpu

    # Setup task, e.g., translation
    task = tasks.setup_task(args)

    # Load ensemble
    logger.info('loading model(s) from {}'.format(args.path))
    models, _model_args = checkpoint_utils.load_model_ensemble(
        args.path.split(os.pathsep),
        arg_overrides=eval(args.model_overrides),
        task=task,
    )
    model = models[0]
    model.cuda()
    model.eval()
    # encoder = FairseqEncoder()
    # encoder = checkpoint_utils.load_pretrained_component_from_model(encoder, args.path)
    encoder = model.encoder

    # Set dictionaries
    src_dict = task.source_dictionary
    tgt_dict = task.target_dictionary

    # Handle tokenization and BPE
    tokenizer = encoders.build_tokenizer(args)
    bpe = encoders.build_bpe(args)

    def encode_fn(x):
        if tokenizer is not None:
            x = tokenizer.encode(x)
        x, idx = bpe.encode_with_mapping(x)
        return x, idx

    # Load alignment dictionary for unknown word replacement
    # (None if no unknown word replacement, empty if no path to align dictionary)
    align_dict = utils.load_align_dict(args.replace_unk)

    max_positions = utils.resolve_max_positions(
        task.max_positions(),
        *[model.max_positions()]
    )

    outputs = []
    lines = open(args.input, 'r').readlines()
    print("Dataset size: %d sentences" % len(lines))
    tokens, indices = make_tokens(lines, task, encode_fn)
    n_batches = 0
    encoder_reps = {}
    for batch in make_batches(tokens, args, task, max_positions):
        src_tokens = batch.src_tokens
        src_lengths = batch.src_lengths
        if use_cuda:
            src_tokens = src_tokens.cuda()
            src_lengths = src_lengths.cuda()

        # print(src_tokens.shape)
        # print(src_lengths.shape)
        # print(src_tokens[-1])
        # print(src_lengths)

        enc_outputs = encoder.forward(src_tokens, src_lengths, return_all_hiddens=False)
        # encoder_data = [output.encoder_out for output in enc_outputs]
        final_reps = enc_outputs.encoder_out # MaxTokens x Batch x Dim
        final_reps = final_reps.transpose(0, 1) # Batch x MaxTokens x Dim

        for (i, id) in enumerate(batch.ids.tolist()):
            mapped_rep = map_rep_to_sentence(final_reps[i].cpu().detach().numpy(), 
                                             indices[id],
                                             src_lengths[i])
            encoder_reps[id] = mapped_rep
        n_batches += 1
    
    sentence_to_index = {}
    for (i, line) in enumerate(lines):
        sentence_to_index[str(i)] = line

    outfile = '/raj-learn/data/precomputed_reps/wsj_sentences_all.hdf5'
    make_hdf5_file(sentence_to_index, encoder_reps, outfile)    


def cli_main():
    parser = options.get_generation_parser(interactive=True)
    args = options.parse_args_and_arch(parser)
    main(args)


if __name__ == '__main__':
    cli_main()
