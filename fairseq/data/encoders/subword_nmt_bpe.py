# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from fairseq import file_utils
from fairseq.data.encoders import register_bpe


@register_bpe('subword_nmt')
class SubwordNMTBPE(object):

    @staticmethod
    def add_args(parser):
        # fmt: off
        parser.add_argument('--bpe-codes', type=str,
                            help='path to subword NMT BPE')
        parser.add_argument('--bpe-separator', default='@@',
                            help='BPE separator')
        # fmt: on

    def __init__(self, args):
        if args.bpe_codes is None:
            raise ValueError('--bpe-codes is required for --bpe=subword_nmt')
        codes = file_utils.cached_path(args.bpe_codes)
        try:
            from subword_nmt import apply_bpe
            bpe_parser = apply_bpe.create_parser()
            bpe_args = bpe_parser.parse_args([
                '--codes', codes,
                '--separator', args.bpe_separator,
            ])
            self.bpe = apply_bpe.BPE(
                bpe_args.codes,
                bpe_args.merges,
                bpe_args.separator,
                None,
                bpe_args.glossaries,
            )
            self.bpe_symbol = bpe_args.separator + ' '
        except ImportError:
            raise ImportError('Please install subword_nmt with: pip install subword-nmt')

    def encode(self, x: str) -> str:
        return self.bpe.process_line(x)

    def encode_with_mapping(self, line):
        '''
        input is a line with a full sentence
        sentence --> sequence of tokens
        pass sequence of tokens through segment_tokens
        use output to generate indices mapping to token sequence
        then, join token sequence + strip whitespace to get result
        '''
        token_seq = line.strip('\r\n ').split(' ')
        bpe_seq = self.bpe.segment_tokens(token_seq)
        mapping_idxs = []
        i = 0
        for j in range(len(bpe_seq)):
            mapping_idxs.append(i)
            if self.bpe_symbol not in bpe_seq[j]:
                i += 1
        return self.bpe.process_line(line), mapping_idxs

    def decode(self, x: str) -> str:
        return (x + ' ').replace(self.bpe_symbol, '').rstrip()
