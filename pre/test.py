import tensorflow as tf
import numpy as np
import imageio
from utils import *
from model import *




def test():
    tf.reset_default_graph()
    global_step = tf.Variable(0, dtype=tf.int32, trainable=False, name='global_step')
    gen_in = tf.placeholder(shape=[None, 544, 576, 5, 1], dtype=tf.float32,
                            name='generated_image')
    Gz = generator(gen_in)
    init = tf.global_variables_initializer()
    with tf.Session() as sess:
        sess.run(init)
        saver = initialize(sess)
        for k in range(3):
            idx=k+19
            print('idx'*20)
            print(idx)
            print('idx'*20)
            for i in range(500):
                print('idx: %d  '%idx+'slice: %d'%(i+1))
                image = np.expand_dims(np.load('/mnt/data1/KBS/chaolianying/reconstruction/walnut_%d/pre_test/'%idx+'%d'%(i+1)+'.npy'),axis=0)
                image_recon = sess.run(Gz, feed_dict={gen_in: image})
                image_recon=np.resize(image_recon,[544,576,5])
                imageio.imsave('/mnt/data1/KBS/chaolianying/reconstruction/walnut_%d/pre_proj/'%idx +'%d.tif'%(i+1), image_recon[:,:,2])
            
    
if __name__ == '__main__':
    test()
