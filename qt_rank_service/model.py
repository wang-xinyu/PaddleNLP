# Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import argparse
import os

import paddle
import paddle.nn as nn
import paddlenlp
from paddlenlp.experimental import FasterTokenizer
from paddlenlp.experimental.model_utils import load_vocabulary


class ErnieWithFasterTokenizer(nn.Layer):
    def init_weights(self, layer):
        """ Initialization hook """
        if isinstance(layer, nn.LayerNorm):
            layer._epsilon = 1e-5

    def __init__(self,
                 ernie,
                 vocab_file,
                 do_lower_case=True,
                 is_split_into_words=False,
                 max_seq_len=128,
                 pad_to_max_seq_len=False):
        super(ErnieWithFasterTokenizer, self).__init__()
        self.ernie = ernie
        self.vocab = load_vocabulary(vocab_file)
        self.tokenizer = FasterTokenizer(
            self.vocab,
            do_lower_case=do_lower_case,
            is_split_into_words=is_split_into_words)
        self.max_seq_len = max_seq_len
        self.pad_to_max_seq_len = pad_to_max_seq_len
        self.apply(self.init_weights)

    def forward(self, text, text_pair=None):
        input_ids, token_type_ids = self.tokenizer(
            text=text,
            text_pair=text_pair,
            max_seq_len=self.max_seq_len,
            pad_to_max_seq_len=self.pad_to_max_seq_len)
        logits = self.ernie(input_ids, token_type_ids)
        return logits
