#!/usr/bin/env python3 -u
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""
Computes attention distributions for sentences in a given file.
Writes to hdf5.
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
import re
import time

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

TWO_CHAR_JOINS = [('&apos;', 's'),
                  ('\\', '*'),
                  ('&apos;', 're'),
                  ('&apos;', 've'),
                  ('n', '&apos;t'),
                  ('&apos;', 'm'),
                  ('&apos;', 'll'),
                  ('&apos;', 'd'),
                  ('&apos;', 'S'),
                  ('&apos;', 'RE'),
                  ('&apos;', 'VE'),
                  ('N', '&apos;T'),
                  ('&apos;', 'M'),
                  ('&apos;', 'LL'),
                  ('&apos;', 'D'),
                  ('Can', 'not'),
                  ('Gim', 'me'),
                  ('Gon', 'na'),
                  ('Lem', 'me'),
                  ('T', 'is'),
                  ('T', 'was'),
                  ('Wan', 'na'),
                  ('can', 'not'),
                  ('gim', 'me'),
                  ('gon', 'na'),
                  ('lem', 'me'),
                  ('t', 'is'),
                  ('t', 'was'),
                  ('wan', 'na'),
                 ]

CASUAL_WORDS = [('%', 'eh'),
                ('%', 'u@@'),
                ('%', 'um'),
                ('%', 'h@@'),
                ('%', 'ah')
               ]

TWO_CHAR_JOINS += CASUAL_WORDS

BRIDGE_CHARS = ['@', '_']

THREE_CHAR_JOINS = [('AT', '&amp;', 'T'),
                    ('PG', '&amp;', 'E')
                   ]

def adjust_indices(line, encoded, idxs):
    # Look for two-char sequences that should be joined
    encoded = encoded.split(' ')
    for i in range(0, len(encoded) - 1):
        for seq in TWO_CHAR_JOINS:
            if encoded[i] == seq[0] and encoded[i+1] == seq[1]:
                for j in range(i+1, len(encoded)):
                    idxs[j] = idxs[j] - 1

    # Look for abbreviated proper nouns like "Va .", "Inc .", etc.
    # Have to use some more complex logic to find BPE-split 
    # proper nouns like 'C@@ ali@@ f .'
    for i in range(0, len(encoded) - 2):
        k = 1
        while i+k < len(encoded) - 1 and idxs[i] == idxs[i+k]:
            k += 1
        if i+k >= len(encoded) - 1: break
        if encoded[i][0].isupper() and encoded[i+k] == '.':
            abbrev = ''.join(encoded[i : i+k]).replace('@@', '')
            abbrev += '.'
            if abbrev in line:
                for j in range(i+k, len(encoded)):
                    idxs[j] = idxs[j] - 1

    # Handle characters joined by @, like in email addresses
    for i in range(1, len(encoded) - 1):
        if encoded[i] in BRIDGE_CHARS and idxs[i] == idxs[i-1] + 1 and idxs[i] == idxs[i+1] - 1:
            string = encoded[i-1] + encoded[i] + encoded[i+1].replace('@@', '')
            if string in line:
                idxs[i] = idxs[i] - 1
                for j in range(i+1, len(encoded)):
                    idxs[j] = idxs[j] - 2

    # Handle three-char seqs that should be joined
    for i in range(len(encoded) - 2):
        for seq in THREE_CHAR_JOINS:
            if encoded[i] == seq[0] and encoded[i+1] == seq[1] and encoded[i+2] == seq[2]:
                idxs[i+1] -= 1
                for j in range(i+2, len(encoded)):
                    idxs[j] -= 2

    # # Handle % signs prepended to words %uh gets split as % uh, for example
    # for i in range(len(encoded)):
    #     if encoded[i] == '%':
    #         if i != len(encoded) - 1:
    #             k = 1
    #             next_word = encoded[i+k]
    #             while '@@' in encoded[i+k]:
    #                 k += 1
    #                 next_word += encoded[i+k]
    #             next_word = next_word.replace('@@', '')
    #             string = encoded[i] + next_word
    #             print(string)
    #             if string in line:
    #                 for j in range(i+1, len(encoded)):
    #                     idxs[j] = idxs[j] - 1

    #         if i == 0:
    #            continue
    #         prev_word = encoded[i-1]
    #         k = 2
    #         while idxs[i-k] == idxs[i-1]:
    #            prev_word = encoded[i-k] + prev_word
    #            k -= 1
    #         prev_word = prev_word.replace('@@', '')
    #         prev_word = prev_word.replace(' ', '')
    #         string = prev_word + encoded[i]
    #         if string in line:
    #             print(string)
    #             for j in range(i, len(encoded)):
    #                 idxs[j] = idxs[j] - 1

    return idxs


def make_string(encoded, indices):
    encoded = encoded.split(' ')
    i = 0
    res = ""
    while i < len(encoded):
        res += encoded[i]
        if i >= len(indices) - 1:
            break
        elif indices[i] == indices[i+1]:
            i += 1
            continue
        else:
            res += ' '
            i += 1
    res = res.replace('@@', '')
    res = res.replace('&apos;', '\'')
    return res


def make_tokens(lines, task, encode_fn):
    encoded_inputs = []
    indices = []
    for src_str in lines:
        src_str = src_str.replace('’', '\'')
        x, idxs = encode_fn(src_str)
        idxs = adjust_indices(src_str, x, idxs)
        encoded_inputs.append(x)
        indices.append(idxs)
    tokens = [
        task.source_dictionary.encode_line(
            encoded_input, add_if_not_exist=False
        ).long()
        for encoded_input in encoded_inputs
    ]
    return tokens, indices, encoded_inputs


def make_batches(tokens, args, task, max_positions):
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
    rep = rep[-token_length : -1]
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


def map_probs(p, indices):
    '''
    INPUT
    p_tok (n_tokens): attention to each of the tokens
    indices (n_tokens): for each token, give index of word it corresponds to
    OUTPUT
    p_word (n_words): attention to each of the words
    '''
    sentence_length = max(indices) + 1
    p_word = np.zeros(sentence_length, dtype=np.float32)
    for i in range(sentence_length):
        p_word[i] == np.sum(p_tok[indices == i])



def detokenize_attn(attn, indices, token_length):
    '''
    rep: MaxTokens x Dim
    attn: Heads x Tokens x Tokens
    indices: index mapping each token to a word index in the original sentence
    tokenlength: value corresponding to how many tokens the encoded input is, w/o padding 
    '''
    # get rid of padding
    attn = attn[:, -token_length : , -token_length : ]
    # number of tokens in the sentence as tokenized by whatever dataset we are using
    sentence_length = max(indices) + 1

    sentence_attn = np.zeros((attn.shape[0], sentence_length, sentence_length), dtype=np.float32)
    for (i, sent_idx) in enumerate(indices):
        # sentence_attn[:, i]
        continue



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
    t0 = time.time()
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

    # lines = open(args.input, 'r').readlines()
    lines = ["Bob went to the supermarket . He did n't know where it was .",
             # "I have never listened to that kind of music before ."
             ]
    print("Dataset size: %d sentences" % len(lines))
    tokens, indices, encoded_inputs = make_tokens(lines, task, encode_fn)
    count_matched = 0
    count_mismatched = 0
    count_toolong = 0
    keep_lines = []
    for (i, line) in enumerate(lines):
        line = line.strip().split(' ')
        if max(indices[i]) + 1 != len(line):
            # res = make_string(encoded_inputs[i], indices[i])
            # print(' '.join(line))
            # print(res)
            # print(encoded_inputs[i])
            # print(indices[i])
            count_mismatched += 1
        elif len(line) > 1024:
            count_toolong += 1
        else:
            print(line)
            print(encoded_inputs[i].split(' '))
            print(indices[i])
            count_matched += 1
            keep_lines.append(i)
    keep_lines = set(keep_lines)
    print('''%d sentences had matched tokenizations with space-splitting; 
            %d did not and %d were too long''' % (count_matched, count_mismatched, count_toolong))
    # sys.exit(0)
    encoder_reps = {}
    for batch in make_batches(tokens, args, task, max_positions):
        src_tokens = batch.src_tokens
        src_lengths = batch.src_lengths
        if use_cuda:
            src_tokens = src_tokens.cuda()
            src_lengths = src_lengths.cuda()

        enc_outputs = encoder.forward(src_tokens, src_lengths, return_all_hiddens=False, return_all_attns=True)
        enc_self_attns = enc_outputs.encoder_self_attns # List[Batch x Heads x Tokens x Tokens]

        for i in range(len(enc_self_attns)):
            print(enc_self_attns[i].shape)
            if i==5:
                print(enc_self_attns[i][0])
                print(enc_self_attns[i][1])

        # for (i, id) in enumerate(batch.ids.tolist()):
        #     if id not in keep_lines:
        #         continue
        #     mapped_reps = []
        #     for k in range(len(enc_self_attns)):
        #         mapped_rep = map_rep_to_sentence(encoder_states[k][i].cpu().detach().numpy(), 
        #                                          indices[id],
        #                                          src_lengths[i])
        #         mapped_reps.append(mapped_rep)
        #     encoder_reps[id] = np.array(mapped_reps)
    
    # sentence_to_index = {}
    # for (i, line) in enumerate(lines):
    #     if i not in keep_lines:
    #         continue
    #     sentence_to_index[line.strip()] = str(i)

    # make_hdf5_file(sentence_to_index, encoder_reps, args.outfile)
    # print("Precomputing reps took %.2fsec" % (time.time() - t0))


def cli_main():
    parser = options.get_generation_parser(interactive=True)
    args = options.parse_args_and_arch(parser)
    main(args)


if __name__ == '__main__':
    cli_main()
