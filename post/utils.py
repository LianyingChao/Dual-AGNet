import os
import tensorflow as tf

CKPT_DIR = './Checkpoints/'
GRAPH_DIR = './Graphs/'

def initialize(sess):
    saver = tf.train.Saver()
    writer = tf.summary.FileWriter(GRAPH_DIR, sess.graph)
    if not os.path.exists(CKPT_DIR):
        os.makedirs(CKPT_DIR)
    ckpt = tf.train.get_checkpoint_state(os.path.dirname(CKPT_DIR))
    if ckpt and ckpt.model_checkpoint_path:
        saver.restore(sess, ckpt.model_checkpoint_path)
    return saver
