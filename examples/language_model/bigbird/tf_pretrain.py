import warnings
warnings.filterwarnings('ignore')

from bigbird.core import flags
from bigbird.core import modeling, optimization
from bigbird.core import utils
from bigbird.pretrain import run_pretraining
import tensorflow.compat.v2 as tf
import tensorflow_datasets as tfds
from tqdm import tqdm
import sys
import os
import numpy as np
tf.enable_v2_behavior()

FLAGS = flags.FLAGS
if not hasattr(FLAGS, "f"): flags.DEFINE_string("f", "", "")
FLAGS(sys.argv)

#FLAGS.init_checkpoint = "/root/projects/bigbird/ckpt/bigbird-transformer/pretrain/bigbr_base/model.ckpt-0"
FLAGS.init_checkpoint = "/root/projects/bigbird_weight/model.ckpt-0"
FLAGS.attention_probs_dropout_prob = 0.0
FLAGS.hidden_dropout_prob = 0.0
#FLAGS.train_batch_size = 1
FLAGS.save_checkpoints_steps = 1
FLAGS.learning_rate = 1e-4
bert_config = flags.as_dictionary()

model = modeling.BertModel(bert_config)
masked_lm = run_pretraining.MaskedLMLayer(
    bert_config["hidden_size"],
    bert_config["vocab_size"],
    model.embeder,
    initializer=utils.create_initializer(bert_config["initializer_range"]),
    activation_fn=utils.get_activation(bert_config["hidden_act"]))
next_sentence = run_pretraining.NSPLayer(
    bert_config["hidden_size"],
    initializer=utils.create_initializer(bert_config["initializer_range"]))
stop_gradient_weights = [
    # "bert/encoder/layer_0/attention/self/query/kernel:0",
    # "bert/encoder/layer_0/attention/self/query/bias:0",
    # "bert/encoder/layer_0/attention/self/key/kernel:0",
    # "bert/encoder/layer_0/attention/self/key/bias:0",
    # "bert/encoder/layer_0/attention/self/value/kernel:0",
    # "bert/encoder/layer_0/attention/self/value/bias:0",
    # "bert/embeddings/word_embeddings:0",
    # "bert/embeddings/token_type_embeddings:0",
    # "bert/embeddings/position_embeddings:0",
    # "bert/encoder/layer_0/attention/output/dense/kernel:0",
    # "bert/encoder/layer_0/attention/output/dense/bias:0",
]


@tf.function(experimental_compile=True)
def fwd_bwd(features):
    with tf.GradientTape() as g:
        sequence_output, pooled_output = model(
            features["input_ids"],
            training=True,
            token_type_ids=features.get("segment_ids"))

        masked_lm_loss, masked_lm_log_probs = masked_lm(
            sequence_output,
            label_ids=features.get("masked_lm_ids"),
            label_weights=features.get("masked_lm_weights"),
            masked_lm_positions=features.get("masked_lm_positions"))

        next_sentence_loss, next_sentence_log_probs = next_sentence(
            pooled_output, features.get("next_sentence_labels"))
        #total_loss = masked_lm_loss
        #if bert_config["use_nsp"]:
        total_loss = masked_lm_loss + next_sentence_loss
    weight_list = model.trainable_weights + masked_lm.trainable_weights + next_sentence.trainable_weights
    weight_list = [
        weight for weight in weight_list
        if weight.name not in stop_gradient_weights
    ]
    grads = g.gradient(total_loss, weight_list + [sequence_output])
    return total_loss, masked_lm_log_probs, next_sentence_log_probs, grads, sequence_output  #, per_example_loss


train_input_fn = run_pretraining.input_fn_builder(
    data_dir=FLAGS.data_dir,
    vocab_model_file=FLAGS.vocab_model_file,
    masked_lm_prob=FLAGS.masked_lm_prob,
    max_encoder_length=FLAGS.max_encoder_length,
    max_predictions_per_seq=FLAGS.max_predictions_per_seq,
    preprocessed_data=FLAGS.preprocessed_data,
    substitute_newline=FLAGS.substitute_newline,
    tmp_dir=os.path.join(FLAGS.output_dir, "tfds"),
    is_training=True)

dataset = train_input_fn({'batch_size': FLAGS.train_batch_size})

np.random.seed(0)
for ex in dataset.take(1):
    fwd_bwd(ex)

#################### load params #######################################
ckpt_path = FLAGS.init_checkpoint
ckpt_reader = tf.compat.v1.train.NewCheckpointReader(ckpt_path)
model.set_weights([
    ckpt_reader.get_tensor(v.name[:-2])
    for v in tqdm(
        model.trainable_weights, position=0)
])
masked_lm.set_weights([
    ckpt_reader.get_tensor(v.name[:-2])
    for v in tqdm(
        masked_lm.trainable_weights, position=0)
])
next_sentence.set_weights([
    ckpt_reader.get_tensor(v.name[:-2])
    for v in tqdm(
        next_sentence.trainable_weights, position=0)
])

# learning_rate = optimization.get_linear_warmup_linear_decay_lr(
#     init_lr=bert_config["learning_rate"],
#     num_train_steps=bert_config["num_train_steps"],
#     num_warmup_steps=bert_config["num_warmup_steps"])
learning_rate = bert_config["learning_rate"]
opt = optimization.get_optimizer(bert_config, learning_rate)
# opt = tf.keras.optimizers.Adam(learning_rate, epsilon=1e-8)

train_loss = tf.keras.metrics.Mean(name='train_loss')
np.random.seed(0)

input_ids = []
segment_ids = []
masked_lm_positions = []
masked_lm_ids = []
masked_lm_weights = []
next_sentence_labels = []

weight_list = model.trainable_weights + masked_lm.trainable_weights + next_sentence.trainable_weights
weight_list = [
    weight for weight in weight_list if weight.name not in stop_gradient_weights
]
loss_list = []
from visualdl import LogWriter
with LogWriter(logdir="./vdl_log/tf_bigbird") as writer:
    sequence_output_list = []
    for step, ex in enumerate(
            tqdm(
                dataset.take(FLAGS.num_train_steps), position=0)):
        total_loss, masked_lm_log_probs, next_sentence_log_probs, grads, sequence_output = fwd_bwd(
            ex)
        # print("grads:{}".format(grads),flush=True)
        opt.apply_gradients(zip(grads, weight_list))
        # train_loss(total_loss)
        # print('ex: {}'.format(ex))
        # save input data
        input_ids.extend(ex['input_ids'].numpy().tolist())
        segment_ids.extend(ex['segment_ids'].numpy().tolist())
        masked_lm_positions.extend(ex['masked_lm_positions'].numpy().tolist())
        masked_lm_ids.extend(ex['masked_lm_ids'].numpy().tolist())
        masked_lm_weights.extend(ex['masked_lm_weights'].numpy().tolist())
        next_sentence_labels.extend(ex['next_sentence_labels'].numpy().tolist())
        # for i in range(0, 3):
        #     np.save("tf_grad_{}.npy".format(i), np.array(grads[i].numpy()))
        # for i, grad in enumerate(grads[:-1]):
        #     np_grad = grad.numpy()
        #     print("grad name: {} shape: {} sum:{} abs_sum: {}".format(weight_list[i].name, 
        #         np_grad.shape, np.sum(np_grad), np.sum(np.abs(np_grad))), flush=True)
        # print("grad name: seq_out shape: {} sum:{} abs_sum: {}".format(sequence_output.shape, 
        #     np.sum(grads[-1]), np.sum(np.abs(grads[-1])), flush=True))

        print('step = {}, loss = {}'.format(step, total_loss), flush=True)
        sequence_output_list.append(sequence_output.numpy())
        np.random.seed(0)
        loss_list.append(total_loss)
        writer.add_scalar(tag="loss", step=step, value=total_loss.numpy())
    sequence_output_list = np.concatenate(sequence_output_list, axis=0)
    np.save("tf_seq_out.npy", sequence_output_list)
np.save('tf_loss_np.npy', loss_list)

np.savez(
    'wiki1000.npz',
    input_ids=input_ids,
    segment_ids=segment_ids,
    masked_lm_positions=masked_lm_positions,
    masked_lm_ids=masked_lm_ids,
    masked_lm_weights=masked_lm_weights,
    next_sentence_labels=next_sentence_labels)
