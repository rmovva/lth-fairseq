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


def buffered_read(input, buffer_size):
    buffer = []
    with fileinput.input(files=[input], openhook=fileinput.hook_encoded("utf-8")) as h:
        for src_str in h:
            buffer.append(src_str.strip())
            if len(buffer) >= buffer_size:
                yield buffer
                buffer = []

    if len(buffer) > 0:
        yield buffer


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


def main(args):
    outfile = '/raj-learn/data/encoder_reps/test_ccg_head.npz'

    utils.import_user_module(args)

    if args.buffer_size < 1:
        args.buffer_size = 1
    if args.max_tokens is None and args.max_sentences is None:
        args.max_sentences = 1

    assert not args.max_sentences or args.max_sentences <= args.buffer_size, \
        '--max-sentences/--batch-size cannot be larger than --buffer-size'

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

    if args.buffer_size > 1:
        logger.info('Sentence buffer size: %s', args.buffer_size)
    logger.info('NOTE: hypothesis and token scores are output in base 2')
    logger.info('Type the input sentence and press return:')
    start_id = 0
    outputs = []
    for inputs in buffered_read(args.input, args.buffer_size):
        tokens, indices = make_tokens(inputs, task, encode_fn)
        for batch in make_batches(tokens, args, task, max_positions):
            src_tokens = batch.src_tokens
            src_lengths = batch.src_lengths
            if use_cuda:
                src_tokens = src_tokens.cuda()
                src_lengths = src_lengths.cuda()

            print(len(src_tokens))
            print(len(src_lengths))

            enc_outputs = encoder.forward(src_tokens, src_lengths, return_all_hiddens=False)
            encoder_data = [[output.encoder_out, output.encoder_out] for output in enc_outputs]
            final_reps = final_reps.transpose(0, 1)
            print(final_reps.shape)

            for (i, id) in enumerate(batch.ids.tolist()):
                encoder_reps.append((start_id + id, final_reps[i]))
        # update running id counter
        start_id += len(inputs)
    encoder_reps = sorted(encoder_reps, key=lambda x: x[0])
    encoder_reps = torch.cat([x[1] for x in encoder_reps]).cpu().detach().numpy()
    np.savez(outfile, encoder_reps)


def cli_main():
    parser = options.get_generation_parser(interactive=True)
    args = options.parse_args_and_arch(parser)
    main(args)


if __name__ == '__main__':
    cli_main()
