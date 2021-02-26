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

FLAGS = flags.FLAGS
if not hasattr(FLAGS, "f"): flags.DEFINE_string("f", "", "")
FLAGS(sys.argv)

tf.enable_v2_behavior()

#FLAGS.init_checkpoint = "/root/projects/bigbird/ckpt/bigbird-transformer/pretrain/bigbr_base/model.ckpt-0"
FLAGS.init_checkpoint = "/root/projects/bigbird_weight/model.ckpt-0"
FLAGS.attention_probs_dropout_prob = 0.0
FLAGS.hidden_dropout_prob = 0.0
FLAGS.train_batch_size = 1
FLAGS.save_checkpoints_steps = 1
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
    total_loss = masked_lm_loss
    if bert_config["use_nsp"]:
        total_loss += next_sentence_loss
    grads = g.gradient(total_loss,
                       model.trainable_weights + masked_lm.trainable_weights +
                       next_sentence.trainable_weights)
    return total_loss, masked_lm_log_probs, next_sentence_log_probs, grads  #, per_example_loss, sequence_output


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

dataset = train_input_fn({'batch_size': 1})

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

opt = optimization.get_optimizer(bert_config, FLAGS.learning_rate)

train_loss = tf.keras.metrics.Mean(name='train_loss')
np.random.seed(0)

input_ids = []
segment_ids = []
masked_lm_positions = []
masked_lm_ids = []
masked_lm_weights = []
next_sentence_labels = []

loss_list = []
for i, ex in enumerate(tqdm(dataset.take(FLAGS.num_train_steps), position=0)):
    total_loss, masked_lm_log_probs, next_sentence_log_probs, grads = fwd_bwd(
        ex)
    opt.apply_gradients(
        zip(grads, model.trainable_weights + masked_lm.trainable_weights +
            next_sentence.trainable_weights))
    # train_loss(total_loss)
    # print('ex: {}'.format(ex))

    # save input data
    input_ids.append(ex['input_ids'].numpy())
    segment_ids.append(ex['segment_ids'].numpy())
    masked_lm_positions.append(ex['masked_lm_positions'].numpy())
    masked_lm_ids.append(ex['masked_lm_ids'].numpy())
    masked_lm_weights.append(ex['masked_lm_weights'].numpy())
    next_sentence_labels.append(ex['next_sentence_labels'].numpy())

    print('step = {}, loss = {}'.format(i, total_loss), flush=True)
    np.random.seed(0)
    loss_list.append(total_loss)
np.save('tf_loss_np.npy', loss_list)

np.savez(
    'wiki1000.npz',
    input_ids=input_ids,
    segment_ids=segment_ids,
    masked_lm_positions=masked_lm_positions,
    masked_lm_ids=masked_lm_ids,
    masked_lm_weights=masked_lm_weights,
    next_sentence_labels=next_sentence_labels)
