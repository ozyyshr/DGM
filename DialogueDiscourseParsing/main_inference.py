import tensorflow as tf
import numpy as np
import os, random, time
import json
import sys
reload(sys)
sys.setdefaultencoding('utf8')

from Model import Model
from utils import load_data, build_vocab, preview_data, get_batches

if not os.environ.has_key('CUDA_VISIBLE_DEVICES'): 
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
FLAGS = tf.flags.FLAGS

tf.flags.DEFINE_boolean('train', False, 'train model')
tf.flags.DEFINE_integer('display_interval', 500, 'step interval to display information')
tf.flags.DEFINE_boolean('show_predictions', False, 'show predictions in the test stage')
tf.flags.DEFINE_string('word_vector', 'DialogueDiscourseParsing/glove/glove.6B.100d.txt', 'word vector')
tf.flags.DEFINE_string('prefix', 'dev', 'prefix for storing model and log')
tf.flags.DEFINE_integer('vocab_size', 1000, 'vocabulary size')
tf.flags.DEFINE_integer('max_edu_dist', 20, 'maximum distance between two related edus') 
tf.flags.DEFINE_integer('dim_embed_word', 100, 'dimension of word embedding')
tf.flags.DEFINE_integer('dim_embed_relation', 100, 'dimension of relation embedding')
tf.flags.DEFINE_integer('dim_feature_bi', 4, 'dimension of binary features')
tf.flags.DEFINE_boolean('use_structured', True, 'use structured encoder')
tf.flags.DEFINE_boolean('use_speaker_attn', True, 'use speaker highlighting mechanism')
tf.flags.DEFINE_boolean('use_shared_encoders', False, 'use shared encoders')
tf.flags.DEFINE_boolean('use_random_structured', False, 'use random structured repr.')
tf.flags.DEFINE_integer('num_epochs', 50, 'number of epochs')
tf.flags.DEFINE_integer('num_units', 256, 'number of hidden units')
tf.flags.DEFINE_integer('num_layers', 1, 'number of RNN layers in encoders')
tf.flags.DEFINE_integer('num_relations', 16, 'number of relation types')
tf.flags.DEFINE_integer('batch_size', 4, 'batch size')
tf.flags.DEFINE_float('keep_prob', 0.5, 'probability to keep units in dropout')
tf.flags.DEFINE_float('learning_rate', 0.1, 'learning rate')
tf.flags.DEFINE_float('learning_rate_decay', 0.98, 'learning rate decay factor')
    
def get_summary_sum(s, length):
    loss_bi, loss_multi = s[0] / length, s[1] / length
    prec_bi, recall_bi = s[4] * 1. / s[3], s[4] * 1. / s[2]
    f1_bi = 2 * prec_bi * recall_bi / (prec_bi + recall_bi)
    prec_multi, recall_multi = s[5] * 1. / s[3], s[5] * 1. / s[2]
    f1_multi = 2 * prec_multi * recall_multi / (prec_multi + recall_multi)
    return [loss_bi, loss_multi, f1_bi, f1_multi]    
    
map_relations = {}
data_train = load_data('DialogueDiscourseParsing/data/processed_data/train.json', map_relations)
with open('dev_for_train.json') as f:
    data_test = json.load(f)

vocab, embed = build_vocab(data_train)
model_dir, log_dir = 'DialogueDiscourseParsing/' + FLAGS.prefix + '_model', 'DialogueDiscourseParsing/' + FLAGS.prefix + '_log'

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.Session(config=config)
with sess.as_default():
    model = Model(sess, FLAGS, embed, data_train)

    global_step = tf.Variable(0, name='global_step', trainable=False)
    global_step_inc_op = global_step.assign(global_step + 1)    
    epoch = tf.Variable(0, name='epoch', trainable=False)
    epoch_inc_op = epoch.assign(epoch + 1)

    saver = tf.train.Saver(write_version=tf.train.SaverDef.V2, max_to_keep=None, pad_step_number=True)
    
    summary_list = ['loss_bi', 'loss_multi', 'f1_bi', 'f1_multi']
    summary_num = len(summary_list)
    len_output_feed = 6

    print 'Reading model parameters from %s' % model_dir 
    saver.restore(sess, tf.train.latest_checkpoint(model_dir))

    test_batches = get_batches(data_test, 1, sort=False)

    s = np.zeros(len_output_feed)
    random.seed(0)
    mapping = []
    for k, batch in enumerate(test_batches):
        if len(batch[0]['edus']) == 1: 
            continue    
        ops = model.step(batch)
        # for i in range(len_output_feed):
        #     s[i] += ops[i]

        idx = preview_data(batch, ops[-1], map_relations, vocab) 

        mapping.append({'edus': batch[0]['edus'], 'id': batch[0]['id'], 'relations':[{'x': item[0], 'y': item[1], 'type': item[2]} for item in idx]})
    
    with open(out_file, 'wt') as f:     ## TODO: change out_file
        json.dump(mapping, f, indent=2)
    summary_sum = get_summary_sum(s, len(test_batches))

    print 'Test:'
    for k in range(summary_num):
        print '  test %s: %.5lf' % (summary_list[k], summary_sum[k])   
