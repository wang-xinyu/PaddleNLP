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
import logging
import os
import sys
import random
import time
import math
import distutils.util
import json
import numpy as np
import paddle

from paddlenlp.transformers import ErnieForSequenceClassification
from paddlenlp.experimental import FasterErnieForSequenceClassification, FasterErnieModel


def parse_args():
    parser = argparse.ArgumentParser()

    # Required parameters
    parser.add_argument(
        "--params_dir",
        type=str,
        default='./ernie2.0_qt_rank',
        help="The path to model parameters to be loaded.")
    parser.add_argument(
        "--output_dir",
        default=None,
        type=str,
        required=True,
        help="The output directory where the model predictions and checkpoints will be written.",
    )
    args = parser.parse_args()
    return args


def fused_weight(weight, num_head):
    if paddle.in_dynamic_mode():
        a = paddle.transpose(weight, perm=[1, 0])
        return paddle.reshape(
            a, shape=[1, num_head, int(a.shape[0] / num_head), a.shape[1]])
    else:
        a = weight.transpose(1, 0)
        return a.reshape((1, num_head, int(a.shape[0] / num_head), a.shape[1]))


def fused_qkv(q, k, v, num_head):
    fq = fused_weight(q, num_head)
    fk = fused_weight(k, num_head)
    fv = fused_weight(v, num_head)
    if paddle.in_dynamic_mode():
        return paddle.concat(x=[fq, fk, fv], axis=0)
    else:
        return np.concatenate((fq, fk, fv), axis=0)


def convert_encoder(encoder, fused_encoder, num_heads):
    for i in range(len(encoder.layers)):
        base_layer = encoder.layers[i]
        fused_layer = fused_encoder.layers[i]
        fused_layer.ffn._linear1_weight.set_value(base_layer.linear1.weight)
        fused_layer.ffn._linear1_bias.set_value(base_layer.linear1.bias)
        fused_layer.ffn._linear2_weight.set_value(base_layer.linear2.weight)
        fused_layer.ffn._linear2_bias.set_value(base_layer.linear2.bias)
        fused_layer.ffn._ln1_scale.set_value(base_layer.norm2.weight)
        fused_layer.ffn._ln1_bias.set_value(base_layer.norm2.bias)
        fused_layer.ffn._ln2_scale.set_value(base_layer.norm2.weight)
        fused_layer.ffn._ln2_bias.set_value(base_layer.norm2.bias)

        fused_layer.fused_attn.linear_weight.set_value(
            base_layer.self_attn.out_proj.weight)
        fused_layer.fused_attn.linear_bias.set_value(
            base_layer.self_attn.out_proj.bias)
        fused_layer.fused_attn.pre_ln_scale.set_value(base_layer.norm1.weight)
        fused_layer.fused_attn.pre_ln_bias.set_value(base_layer.norm1.bias)
        fused_layer.fused_attn.ln_scale.set_value(base_layer.norm1.weight)
        fused_layer.fused_attn.ln_bias.set_value(base_layer.norm1.bias)

        q = base_layer.self_attn.q_proj.weight
        q_bias = base_layer.self_attn.q_proj.bias
        k = base_layer.self_attn.k_proj.weight
        k_bias = base_layer.self_attn.k_proj.bias
        v = base_layer.self_attn.v_proj.weight
        v_bias = base_layer.self_attn.v_proj.bias

        qkv_weight = fused_qkv(q, k, v, num_heads)
        fused_layer.fused_attn.qkv_weight.set_value(qkv_weight)

        if paddle.in_dynamic_mode():
            tmp = paddle.concat(x=[q_bias, k_bias, v_bias], axis=0)
            qkv_bias = paddle.reshape(
                tmp, shape=[3, num_heads, int(tmp.shape[0] / 3 / num_heads)])
            fused_layer.fused_attn.qkv_bias.set_value(qkv_bias)
        else:
            qkv_bias = np.concatenate((q, k, v), axis=0)
            fused_layer.fused_attn.qkv_bias.set_value(qkv_bias)


def do_convert(args):
    paddle.set_device("cpu")
    model_config_file = os.path.join(args.params_dir, "model_config.json")
    vocab_file = os.path.join(args.params_dir, "vocab.txt")
    with open(model_config_file, 'r') as f:
        array = json.load(f)
    ernie_seq_cls = ErnieForSequenceClassification.from_pretrained(
        args.params_dir, num_classes=2)

    faster_ernie_model = FasterErnieModel(vocab_file=vocab_file, **array)
    faster_ernie_seq_cls = FasterErnieForSequenceClassification(
        faster_ernie_model, num_classes=2)

    num_heads = faster_ernie_seq_cls.ernie.encoder.layers[
        0].fused_attn.num_heads
    convert_encoder(ernie_seq_cls.ernie.encoder,
                    faster_ernie_seq_cls.ernie.encoder, num_heads)

    model = faster_ernie_seq_cls
    output_dir = os.path.join(args.output_dir, "faster_ernie_qt_service")
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    model.save_pretrained(output_dir)


if __name__ == "__main__":
    args = parse_args()
    do_convert(args)
