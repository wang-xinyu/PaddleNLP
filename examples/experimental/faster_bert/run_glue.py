# Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
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
from functools import partial

import numpy as np
import paddle
from paddle.io import DataLoader
from paddle.metric import Metric, Accuracy, Precision, Recall

from paddlenlp.datasets import load_dataset
from paddlenlp.data import Stack, Tuple, Pad, Dict
from paddlenlp.data.sampler import SamplerHelper
#from paddlenlp.transformers import BertForSequenceClassification, BertTokenizer
from paddlenlp.transformers import BertTokenizer
from static.modeling import BertForSequenceClassification
from paddlenlp.transformers import ElectraForSequenceClassification, ElectraTokenizer
from paddlenlp.transformers import ErnieForSequenceClassification, ErnieTokenizer
from paddlenlp.transformers import LinearDecayWithWarmup
from paddlenlp.metrics import AccuracyAndF1, Mcc, PearsonAndSpearman
from static.model_convert_util import convert_base_to_fused

FORMAT = '%(asctime)s-%(levelname)s: %(message)s'
logging.basicConfig(level=logging.INFO, format=FORMAT)
logger = logging.getLogger(__name__)

METRIC_CLASSES = {
    "cola": Mcc,
    "sst-2": Accuracy,
    "mrpc": AccuracyAndF1,
    "sts-b": PearsonAndSpearman,
    "qqp": AccuracyAndF1,
    "mnli": Accuracy,
    "qnli": Accuracy,
    "rte": Accuracy,
}

MODEL_CLASSES = {
    "bert": (BertForSequenceClassification, BertTokenizer),
    "ernie": (ErnieForSequenceClassification, ErnieTokenizer),
}


def parse_args():
    parser = argparse.ArgumentParser()

    # Required parameters
    parser.add_argument(
        "--task_name",
        default=None,
        type=str,
        required=True,
        help="The name of the task to train selected in the list: " +
        ", ".join(METRIC_CLASSES.keys()), )
    parser.add_argument(
        "--model_type",
        default=None,
        type=str,
        required=True,
        help="Model type selected in the list: " +
        ", ".join(MODEL_CLASSES.keys()), )
    parser.add_argument(
        "--model_name_or_path",
        default=None,
        type=str,
        required=True,
        help="Path to pre-trained model or shortcut name selected in the list: "
        + ", ".join(
            sum([
                list(classes[-1].pretrained_init_configuration.keys())
                for classes in MODEL_CLASSES.values()
            ], [])), )
    parser.add_argument(
        "--output_dir",
        default=None,
        type=str,
        required=True,
        help="The output directory where the model predictions and checkpoints will be written.",
    )
    parser.add_argument(
        "--max_seq_length",
        default=128,
        type=int,
        help="The maximum total input sequence length after tokenization. Sequences longer "
        "than this will be truncated, sequences shorter will be padded.", )
    parser.add_argument(
        "--learning_rate",
        default=1e-4,
        type=float,
        help="The initial learning rate for Adam.")
    parser.add_argument(
        "--num_train_epochs",
        default=3,
        type=int,
        help="Total number of training epochs to perform.", )
    parser.add_argument(
        "--logging_steps",
        type=int,
        default=100,
        help="Log every X updates steps.")
    parser.add_argument(
        "--save_steps",
        type=int,
        default=100,
        help="Save checkpoint every X updates steps.")
    parser.add_argument(
        "--batch_size",
        default=32,
        type=int,
        help="Batch size per GPU/CPU for training.", )
    parser.add_argument(
        "--weight_decay",
        default=0.0,
        type=float,
        help="Weight decay if we apply some.")
    parser.add_argument(
        "--warmup_steps",
        default=0,
        type=int,
        help="Linear warmup over warmup_steps. If > 0: Override warmup_proportion"
    )
    parser.add_argument(
        "--warmup_proportion",
        default=0.1,
        type=float,
        help="Linear warmup proportion over total steps.")
    parser.add_argument(
        "--adam_epsilon",
        default=1e-6,
        type=float,
        help="Epsilon for Adam optimizer.")
    parser.add_argument(
        "--max_steps",
        default=-1,
        type=int,
        help="If > 0: set total number of training steps to perform. Override num_train_epochs.",
    )
    parser.add_argument(
        "--seed", default=42, type=int, help="random seed for initialization")
    parser.add_argument(
        "--device",
        default="gpu",
        type=str,
        choices=["cpu", "gpu", "xpu", "npu"],
        help="The device to select to train the model, is must be cpu/gpu/xpu/npu."
    )
    parser.add_argument(
        "--use_amp",
        type=distutils.util.strtobool,
        default=False,
        help="Enable mixed precision training.")
    parser.add_argument(
        "--scale_loss",
        type=float,
        default=2**15,
        help="The value of scale_loss for fp16.")
    args = parser.parse_args()
    return args


def set_seed(args):
    # Use the same data seed(for data shuffle) for all procs to guarantee data
    # consistency after sharding.
    random.seed(args.seed)
    np.random.seed(args.seed)
    # Maybe different op seeds(for dropout) for different procs is better. By:
    # `paddle.seed(args.seed + paddle.distributed.get_rank())`
    paddle.seed(args.seed)


@paddle.no_grad()
def evaluate(model, loss_fct, metric, data_loader):
    model.eval()
    metric.reset()
    for batch in data_loader:
        input_ids, segment_ids, labels = batch
        logits = model(input_ids, segment_ids)
        loss = loss_fct(logits, labels)
        correct = metric.compute(logits, labels)
        metric.update(correct)
    res = metric.accumulate()
    if isinstance(metric, AccuracyAndF1):
        print(
            "eval loss: %f, acc: %s, precision: %s, recall: %s, f1: %s, acc and f1: %s, "
            % (
                loss.numpy(),
                res[0],
                res[1],
                res[2],
                res[3],
                res[4], ),
            end='')
    elif isinstance(metric, Mcc):
        print("eval loss: %f, mcc: %s, " % (loss.numpy(), res[0]), end='')
    elif isinstance(metric, PearsonAndSpearman):
        print(
            "eval loss: %f, pearson: %s, spearman: %s, pearson and spearman: %s, "
            % (loss.numpy(), res[0], res[1], res[2]),
            end='')
    else:
        print("eval loss: %f, acc: %s, " % (loss.numpy(), res), end='')
    model.train()


def convert_example(example,
                    tokenizer,
                    label_list,
                    max_seq_length=512,
                    is_test=False):
    """convert a glue example into necessary features"""
    if not is_test:
        # `label_list == None` is for regression task
        label_dtype = "int64" if label_list else "float32"
        # Get the label
        label = example['labels']
        label = np.array([label], dtype=label_dtype)
    # Convert raw text to feature
    if (int(is_test) + len(example)) == 2:
        example = tokenizer(example['sentence'], max_seq_len=max_seq_length)
    else:
        example = tokenizer(
            example['sentence1'],
            text_pair=example['sentence2'],
            max_seq_len=max_seq_length)

    if not is_test:
        return example['input_ids'], example['token_type_ids'], label
    else:
        return example['input_ids'], example['token_type_ids']

#def fused_weight(weight, num_head):
#    a = paddle.transpose(weight, perm=[1, 0])
#    return paddle.reshape(a, shape=[1, num_head, int(a.shape[0]/num_head), a.shape[1]])
#
#def fused_qkv(qkv_weight, num_head):
#    q = qkv_weight['q']
#    k = qkv_weight['k']
#    v = qkv_weight['v']
#
#    fq = fused_weight(q, num_head)
#    fk = fused_weight(k, num_head)
#    fv = fused_weight(v, num_head)
#    a = paddle.concat(x=[fq, fk, fv], axis=0)
#    return a
#
#def convert_base_to_fused(state_to_load):
#    base_to_fused = dict()
#    base_to_fused["weight"] = "scale"
#    base_to_fused["bias"] = "bias"
#
#    fused_state_to_load = dict()
#    qkv_weight = dict()
#    qkv_bias = dict()
#    qkv_count = 0
#    num_head = 16
#    layer_index = 0
#    for key, value in state_to_load.items():
#        array = key.split('.')
#        fused_array = list(array)
#        if len(array) == 6:#linear or layer_norm
#            if 'linear' in array[4]:
#                #linear1.weight -> ffn._linear1_weight
#                #linear1.bias -> ffn._linear1_bias
#                fused_array[5] = "_" + array[4] + "_" + array[5]
#                fused_array[4] = "ffn"
#                fused_key = '.'.join(fused_array)
#                fused_state_to_load[fused_key] = value
#                #print(key, fused_key)
#                #if array[3] == "0":
#                #    np.savetxt(key+".txt", value)
#
#            elif 'norm' in array[4]:
#                if array[4][-1] == '1':
#                    #norm1.weight -> fused_atten.pre_ln_scale
#                    #norm2.weight -> fused_atten.ln_scale
#                    fused_array[4] = "fused_attn"
#                    fused_array[5] = "ln_" + base_to_fused[array[5]]
#                    fused_key = '.'.join(fused_array)
#                    fused_state_to_load[fused_key] = value
#                    #print(key, fused_key)
#                    #if array[3] == "0":
#                    #    np.savetxt(key+".txt", value)
#                else:
#                    #norm1.weight -> ffn._ln1_scale
#                    fused_array[4] = "ffn"
#                    fused_array[5] = "_ln" + array[4][-1] + "_" + base_to_fused[array[5]]
#                    fused_key = '.'.join(fused_array)
#                    fused_state_to_load[fused_key] = value
#                    #print(key, fused_key)
#                    #if array[3] == "0":
#                    #    np.savetxt(key+".txt", value)
#        elif len(array) == 7:#self_atten
#            if 'q' in array[5]:
#                if array[6] == "weight":
#                    qkv_weight['q'] = value
#                else:
#                    qkv_bias['q'] = value
#                qkv_count += 1
#            elif 'k' in array[5]:
#                if array[6] == "weight":
#                    qkv_weight['k'] = value
#                else:
#                    qkv_bias['k'] = value
#                qkv_count += 1
#            elif 'v' in array[5]:
#                if array[6] == "weight":
#                    qkv_weight['v'] = value
#                else:
#                    qkv_bias['v'] = value
#                qkv_count += 1
#            else:
#                fused_array.pop()
#                fused_array[4] = "fused_attn"
#                if array[6] == "weight":
#                    fused_array[5] = "linear_weight"
#                else:
#                    fused_array[5] = "linear_bias"
#                fused_key = '.'.join(fused_array)
#                fused_state_to_load[fused_key] = value
#                #print(key, fused_key)
#                #if array[3] == "0":
#                #    np.savetxt(key+".txt", value)
#
#            if qkv_count == 6:
#                qkv_count = 0
#                fused_array.pop()
#
#                fused_array[4] = "fused_attn"
#                fused_array[5] = "qkv_weight"
#                fused_key = '.'.join(fused_array)
#                fused_state_to_load[fused_key] = fused_qkv(qkv_weight, num_head)
#                #print(key, fused_key)
#
#                fused_array[4] = "fused_attn"
#                fused_array[5] = "qkv_bias"
#                fused_key = '.'.join(fused_array)
#                a = paddle.concat(x=[qkv_bias['q'], qkv_bias['k'], qkv_bias['v']], axis=0)
#                tmp_bias = paddle.reshape(a, shape=[3, num_head, int(a.shape[0]/3/num_head)])
#                fused_state_to_load[fused_key] = tmp_bias
#                #print(key, fused_key, tmp_bias.numpy().shape)
#                #if array[3] == "0":
#                #    np.savetxt("fused_bias.txt", tmp_bias.numpy().flatten())
#                    #if array[3] == "0":
#
#        else:
#            fused_state_to_load[key] = value
#    return fused_state_to_load



def do_train(args):
    paddle.set_device(args.device)
    if paddle.distributed.get_world_size() > 1:
        paddle.distributed.init_parallel_env()

    set_seed(args)

    args.task_name = args.task_name.lower()
    metric_class = METRIC_CLASSES[args.task_name]
    args.model_type = args.model_type.lower()
    model_class, tokenizer_class = MODEL_CLASSES[args.model_type]

    train_ds = load_dataset('glue', args.task_name, splits="train")
    tokenizer = tokenizer_class.from_pretrained(args.model_name_or_path)

    trans_func = partial(
        convert_example,
        tokenizer=tokenizer,
        label_list=train_ds.label_list,
        max_seq_length=args.max_seq_length)
    train_ds = train_ds.map(trans_func, lazy=True)
    train_batch_sampler = paddle.io.DistributedBatchSampler(
        train_ds, batch_size=args.batch_size, shuffle=True)
    batchify_fn = lambda samples, fn=Tuple(
        Pad(axis=0, pad_val=tokenizer.pad_token_id),  # input
        Pad(axis=0, pad_val=tokenizer.pad_token_type_id),  # segment
        Stack(dtype="int64" if train_ds.label_list else "float32")  # label
    ): fn(samples)
    train_data_loader = DataLoader(
        dataset=train_ds,
        batch_sampler=train_batch_sampler,
        collate_fn=batchify_fn,
        num_workers=0,
        return_list=True)
    if args.task_name == "mnli":
        dev_ds_matched, dev_ds_mismatched = load_dataset(
            'glue', args.task_name, splits=["dev_matched", "dev_mismatched"])

        dev_ds_matched = dev_ds_matched.map(trans_func, lazy=True)
        dev_ds_mismatched = dev_ds_mismatched.map(trans_func, lazy=True)
        dev_batch_sampler_matched = paddle.io.BatchSampler(
            dev_ds_matched, batch_size=args.batch_size, shuffle=False)
        dev_data_loader_matched = DataLoader(
            dataset=dev_ds_matched,
            batch_sampler=dev_batch_sampler_matched,
            collate_fn=batchify_fn,
            num_workers=0,
            return_list=True)
        dev_batch_sampler_mismatched = paddle.io.BatchSampler(
            dev_ds_mismatched, batch_size=args.batch_size, shuffle=False)
        dev_data_loader_mismatched = DataLoader(
            dataset=dev_ds_mismatched,
            batch_sampler=dev_batch_sampler_mismatched,
            collate_fn=batchify_fn,
            num_workers=0,
            return_list=True)
    else:
        dev_ds = load_dataset('glue', args.task_name, splits='dev')
        dhidden_dropout_probev_ds = dev_ds.map(trans_func, lazy=True)
        dev_batch_sampler = paddle.io.BatchSampler(
            dev_ds, batch_size=args.batch_size, shuffle=False)
        dev_data_loader = DataLoader(
            dataset=dev_ds,
            batch_sampler=dev_batch_sampler,
            collate_fn=batchify_fn,
            num_workers=0,
            return_list=True)

    num_classes = 1 if train_ds.label_list == None else len(train_ds.label_list)
    base_model, base_state_to_load = model_class.from_pretrained(
        args.model_name_or_path, num_classes=num_classes)
    fused_model, fused_state_to_load  = model_class.from_pretrained(
        args.model_name_or_path, num_classes=num_classes)
####convert model to fused model
    fused_state_to_load = convert_base_to_fused(base_state_to_load)
    fused_model.set_state_dict(fused_state_to_load)
####convert model to fused model
    model = fused_model
    #model = base_model
    #model.set_state_dict(base_state_to_load)

    if paddle.distributed.get_world_size() > 1:
        model = paddle.DataParallel(model)

    num_training_steps = args.max_steps if args.max_steps > 0 else (
        len(train_data_loader) * args.num_train_epochs)
    warmup = args.warmup_steps if args.warmup_steps > 0 else args.warmup_proportion

    lr_scheduler = LinearDecayWithWarmup(args.learning_rate, num_training_steps,
                                         warmup)

    # Generate parameter names needed to perform weight decay.
    # All bias and LayerNorm parameters are excluded.
    decay_params = [
        p.name for n, p in model.named_parameters()
        if not any(nd in n for nd in ["bias", "norm"])
    ]
    optimizer = paddle.optimizer.AdamW(
        learning_rate=lr_scheduler,
        beta1=0.9,
        beta2=0.999,
        epsilon=args.adam_epsilon,
        parameters=model.parameters(),
        weight_decay=args.weight_decay,
        apply_decay_param_fun=lambda x: x in decay_params)

    loss_fct = paddle.nn.loss.CrossEntropyLoss(
    ) if train_ds.label_list else paddle.nn.loss.MSELoss()

    metric = metric_class()
    if args.use_amp:
        scaler = paddle.amp.GradScaler(init_loss_scaling=args.scale_loss)

    global_step = 0
    tic_train = time.time()
    for epoch in range(args.num_train_epochs):
        for step, batch in enumerate(train_data_loader):
            global_step += 1

            input_ids, segment_ids, labels = batch
            with paddle.amp.auto_cast(
                    args.use_amp,
                    custom_white_list=["layer_norm", "softmax", "gelu", "fused_attention"]):
                logits = model(input_ids, segment_ids)
                loss = loss_fct(logits, labels)
            if args.use_amp:
                scaler.scale(loss).backward()
                scaler.minimize(optimizer, loss)
            else:
                loss.backward()
                optimizer.step()
            lr_scheduler.step()
            optimizer.clear_grad()
            if global_step % args.logging_steps == 0:
                print(
                    "global step %d/%d, epoch: %d, batch: %d, rank_id: %s, loss: %f, lr: %.10f, speed: %.4f step/s"
                    % (global_step, num_training_steps, epoch, step,
                       paddle.distributed.get_rank(), loss, optimizer.get_lr(),
                       args.logging_steps / (time.time() - tic_train)))
                tic_train = time.time()
            if global_step % args.save_steps == 0 or global_step == num_training_steps:
                tic_eval = time.time()
                if args.task_name == "mnli":
                    evaluate(model, loss_fct, metric, dev_data_loader_matched)
                    evaluate(model, loss_fct, metric,
                             dev_data_loader_mismatched)
                    print("eval done total : %s s" % (time.time() - tic_eval))
                else:
                    evaluate(model, loss_fct, metric, dev_data_loader)
                    print("eval done total : %s s" % (time.time() - tic_eval))
                if paddle.distributed.get_rank() == 0:
                    output_dir = os.path.join(args.output_dir,
                                              "%s_ft_model_%d.pdparams" %
                                              (args.task_name, global_step))
                    if not os.path.exists(output_dir):
                        os.makedirs(output_dir)
                    # Need better way to get inner model of DataParallel
                    model_to_save = model._layers if isinstance(
                        model, paddle.DataParallel) else model
                    model_to_save.save_pretrained(output_dir)
                    tokenizer.save_pretrained(output_dir)
            if global_step >= num_training_steps:
                return


def print_arguments(args):
    """print arguments"""
    print('-----------  Configuration Arguments -----------')
    for arg, value in sorted(vars(args).items()):
        print('%s: %s' % (arg, value))
    print('------------------------------------------------')


if __name__ == "__main__":
    args = parse_args()
    print_arguments(args)
    do_train(args)
