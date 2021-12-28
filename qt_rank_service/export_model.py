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
from functools import partial

import numpy as np
import paddle
import paddle.nn.functional as F
from paddlenlp.data import Stack, Tuple, Pad
from paddlenlp.transformers import ErnieForSequenceClassification

# yapf: disable
parser = argparse.ArgumentParser()
parser.add_argument("--params_dir", type=str, default='./ernie2.0_qt_rank',
    help="The path to model parameters to be loaded.")
parser.add_argument("--output_path", type=str, default='./infer_model/static_graph_params',
    help="The path of model parameter in static graph to be saved.")
args = parser.parse_args()
# yapf: enable

if __name__ == "__main__":
    label_map = {0: 'negative', 1: 'positive'}
    model = ErnieForSequenceClassification.from_pretrained(
        args.params_dir, num_classes=len(label_map))
    model.eval()
    # Convert to static graph with specific input description
    # Just need 2 inputs, the attention_mask and pos_ids can be calculate by model itself.
    model = paddle.jit.to_static(
        model,
        input_spec=[
            paddle.static.InputSpec(
                shape=[None, None], dtype="int64"),  # input_ids
            paddle.static.InputSpec(
                shape=[None, None], dtype="int64"),  # segment_ids
        ])
    # Save in static graph model.
    paddle.jit.save(model, args.output_path)
