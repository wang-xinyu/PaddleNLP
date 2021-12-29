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

from functools import partial
import argparse
import os
import random
import time

import numpy as np
import paddle
from paddle import inference
import paddle.nn.functional as F
from paddlenlp.data import Stack, Tuple, Pad
from paddlenlp.datasets import load_dataset
from paddlenlp.transformers import ErnieTokenizer
from scipy.special import softmax

# yapf: disable
parser = argparse.ArgumentParser()
parser.add_argument("--model_dir", type=str, default='./infer_model/', help="The path to model parameters to be loaded.")
parser.add_argument("--vocab_path", type=str, default='./ernie2.0_qt_rank/vocab.txt', help="The vocab path of tokenizer.")
parser.add_argument("--batch_size", type=int, default=1, help="The number of sequences contained in a mini-batch.")
parser.add_argument("--device", default="gpu", type=str, choices=["cpu", "gpu"], help="The device to select to train the model, is must be cpu/gpu.")
parser.add_argument("--max_seq_length", type=int, default=60, help="The maximum total input sequence length after tokenization. ""Sequences longer than this will be truncated, sequences shorter will be padded.")
parser.add_argument('--use_tensorrt', type=eval, default=False, choices=[True, False], help='Enable to use tensorrt to speed up.')
parser.add_argument("--precision", type=str, default="fp32", choices=["fp32", "fp16", "int8"], help='The tensorrt precision.')
parser.add_argument('--cpu_threads', type=int, default=10, help='Number of threads to predict when using cpu.')
parser.add_argument('--enable_mkldnn', type=eval, default=False, choices=[True, False], help='Enable to use mkldnn to speed up when using cpu.')
parser.add_argument("--benchmark", type=eval, default=False, help="To log some information about environment and running.")
parser.add_argument("--save_log_path", type=str, default="./log_output/", help="The file path to save log.")
parser.add_argument("--test_ds_path", type=str, default="../../qt_rank_service/test_data", help="")
args = parser.parse_args()
# yapf: enable


def convert_example(example,
                    tokenizer,
                    max_seq_length=512,
                    is_test=False,
                    is_pair=False):
    """
    Builds model inputs from a sequence or a pair of sequence for sequence classification tasks
    by concatenating and adding special tokens. And creates a mask from the two sequences passed 
    to be used in a sequence-pair classification task.
        
    A BERT sequence has the following format:

    - single sequence: ``[CLS] X [SEP]``

    It returns the first portion of the mask (0's).


    Args:
        example(obj:`list[str]`): List of input data, containing text and label if it have label.
        tokenizer(obj:`PretrainedTokenizer`): This tokenizer inherits from :class:`~paddlenlp.transformers.PretrainedTokenizer` 
            which contains most of the methods. Users should refer to the superclass for more information regarding methods.
        max_seq_len(obj:`int`): The maximum total input sequence length after tokenization. 
            Sequences longer than this will be truncated, sequences shorter will be padded.
        is_test(obj:`False`, defaults to `False`): Whether the example contains label or not.

    Returns:
        input_ids(obj:`list[int]`): The list of token ids.
        token_type_ids(obj: `list[int]`): List of sequence pair mask.
        label(obj:`numpy.array`, data type of int64, optional): The input label if not is_test.
    """

    if is_pair:
        text = example["text_a"]
        text_pair = example["text_b"]
    else:
        text = example["text"]
        text_pair = None
    encoded_inputs = tokenizer(
        text=text, text_pair=text_pair, max_seq_len=max_seq_length)
    input_ids = encoded_inputs["input_ids"]
    token_type_ids = encoded_inputs["token_type_ids"]
    if is_test:
        return input_ids, token_type_ids
    label = np.array([example["label"]], dtype="int64")
    return input_ids, token_type_ids, label


class Predictor(object):
    def __init__(self,
                 model_dir,
                 device="gpu",
                 max_seq_length=128,
                 batch_size=32,
                 use_tensorrt=False,
                 precision="fp32",
                 cpu_threads=10,
                 enable_mkldnn=False):
        self.max_seq_length = max_seq_length
        self.batch_size = batch_size
        params_file = os.path.join(model_dir, "static_graph_params.pdiparams")
        model_file = os.path.join(model_dir, "static_graph_params.pdmodel")
        if not os.path.exists(model_file):
            raise ValueError("not find model file path {}".format(model_file))
        if not os.path.exists(params_file):
            raise ValueError("not find params file path {}".format(params_file))
        config = paddle.inference.Config(model_file, params_file)
        if device == "gpu":
            # set GPU configs accordingly
            # such as intialize the gpu memory, enable tensorrt
            config.enable_use_gpu(100, 0)
            precision_map = {
                "fp16": inference.PrecisionType.Half,
                "fp32": inference.PrecisionType.Float32,
                "int8": inference.PrecisionType.Int8
            }
            precision_mode = precision_map[precision]

            if use_tensorrt:
                config.enable_tensorrt_engine(
                    max_batch_size=batch_size,
                    min_subgraph_size=30,
                    precision_mode=precision_mode)
        elif device == "cpu":
            # set CPU configs accordingly,
            # such as enable_mkldnn, set_cpu_math_library_num_threads
            config.disable_gpu()
            if enable_mkldnn:
                # cache 10 different shapes for mkldnn to avoid memory leak
                config.set_mkldnn_cache_capacity(10)
                config.enable_mkldnn()
            config.set_cpu_math_library_num_threads(cpu_threads)
        elif device == "xpu":
            # set XPU configs accordingly
            config.enable_xpu(100)

        config.switch_use_feed_fetch_ops(False)
        self.predictor = paddle.inference.create_predictor(config)
        self.input_handles = [
            self.predictor.get_input_handle(name)
            for name in self.predictor.get_input_names()
        ]
        self.output_handle = self.predictor.get_output_handle(
            self.predictor.get_output_names()[0])

    def predict(self, data, tokenizer):
        """
        Predicts the data labels.
        Args:
            data (obj:`List(str)`): The batch data whose each element is a raw text.
            tokenizer(obj:`PretrainedTokenizer`): This tokenizer inherits from :class:`~paddlenlp.transformers.PretrainedTokenizer` 
                which contains most of the methods. Users should refer to the superclass for more information regarding methods.
        Returns:
            results(obj:`dict`): All the predictions labels.
        """
        examples = []
        for text, text_pair in data:
            example = {"text_a": text, "text_b": text_pair}
            input_ids, segment_ids = convert_example(
                example,
                tokenizer,
                max_seq_length=self.max_seq_length,
                is_test=True,
                is_pair=True)
            examples.append((input_ids, segment_ids))

        batchify_fn = lambda samples, fn=Tuple(
            Pad(axis=0, pad_val=tokenizer.pad_token_id),  # input
            Pad(axis=0, pad_val=tokenizer.pad_token_id),  # segment
        ): fn(samples)
        input_ids, segment_ids = batchify_fn(examples)
        self.input_handles[0].copy_from_cpu(input_ids)
        self.input_handles[1].copy_from_cpu(segment_ids)
        self.predictor.run()
        logits = self.output_handle.copy_to_cpu()

        probs = softmax(logits, axis=1)
        idx = np.argmax(probs, axis=1)
        idx = idx.tolist()
        return probs, idx


def read_data(data_file):
    data = []
    labels = []
    with open(data_file, "r") as f:
        for line in f.readlines():
            text_a, text_b, label = line.strip().split("\t")
            data.append((text_a, text_b))
            labels.append(int(label))
    return data, labels


if __name__ == "__main__":
    tokenizer = ErnieTokenizer(args.vocab_path)

    predictor = Predictor(args.model_dir, args.device, args.max_seq_length,
                          args.batch_size, args.use_tensorrt, args.precision,
                          args.cpu_threads, args.enable_mkldnn)

    data, labels = read_data(args.test_ds_path)
    predict_labels = []
    for example in data:
        _, label = predictor.predict([example], tokenizer)
        predict_labels.extend(label)
    correct = (np.array(labels) == np.array(predict_labels)).sum()
    total = len(labels)
    print("Accuracy: {:4f}".format(correct / total))
    print("Running benchmark....")
    start = time.time()
    batches = [
        data[idx:idx + args.batch_size]
        for idx in range(0, len(data), args.batch_size)
    ]
    for batch in batches:
        predictor.predict(batch, tokenizer)
    print("predict time: {} s".format(time.time() - start))
