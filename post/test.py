import tensorflow as tf
import numpy as np
from utils import *
from model import *
import imageio


def test():
    tf.reset_default_graph()
    global_step = tf.Variable(0, dtype=tf.int32, trainable=False, name='global_step')
    gen_in = tf.placeholder(shape=[None, 448, 448, 9, 1], dtype=tf.float32,
                            name='generated_image')
    Gz = generator(gen_in)
    init = tf.global_variables_initializer()
    with tf.Session() as sess:
        sess.run(init)
        saver = initialize(sess)
        for k in range(3):
            idx=k+19
            for i in range(200):
                print('idx: %d  slice: '%idx + '%d'%(i+1))
                image = np.expand_dims(np.load('/mnt/data1/KBS/chaolianying/reconstruction/walnut_%d/post_test/'%idx+'%d'%(i+1)+'.npy'),axis=0)
                image_recon = sess.run(Gz, feed_dict={gen_in: image})
                image_recon=np.resize(image_recon,[448,448,9])
                imageio.imsave('/mnt/data1/KBS/chaolianying/reconstruction/walnut_%d/post_re/'%idx +'%d.tif'%(i+1), image_recon[:,:,4])    
if __name__ == '__main__':
    test()