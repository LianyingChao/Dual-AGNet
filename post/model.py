import tensorflow as tf
from utils import *


def generator(input):

    w, h, d = input.shape[1], input.shape[2], input.shape[3]

    ##########################===============Encoder====================################################
    output1 = tf.layers.conv3d(input, 32, 3, padding='SAME', kernel_initializer=tf.contrib.layers.xavier_initializer(), name="G_conv1")
    output1 = tf.nn.relu(output1)  
    output1 = tf.layers.conv3d(output1, 32, 3, padding='SAME', kernel_initializer=tf.contrib.layers.xavier_initializer(), name="G_conv2")
    output1 = tf.nn.relu(output1)
    cal1 = output1

    output1 = tf.layers.conv3d(output1, 32, 3, padding='SAME', kernel_initializer=tf.contrib.layers.xavier_initializer(), name="G_conv3")
    output1 = tf.nn.relu(output1)
    feature = tf.reduce_mean(output1, axis=4, keepdims=True)
    feature = tf.layers.conv3d(feature, 1, 1, padding='SAME', kernel_initializer=tf.contrib.layers.xavier_initializer(), name="G_convAtt1")
    feature = tf.reshape(feature, [-1, w*h*d])
    feature = tf.nn.sigmoid(feature)
    feature = tf.reshape(feature, [-1, w, h, d, 1])
    feature = tf.layers.conv3d(feature, 1, 1, padding='SAME', kernel_initializer=tf.contrib.layers.xavier_initializer(), name="G_convAtt2")
    feature = tf.nn.relu(feature)
    output1 = output1*feature
    output1 = tf.layers.conv3d(output1, 32, 3, padding='SAME', kernel_initializer=tf.contrib.layers.xavier_initializer(), name="G_conv4")
    output1 = tf.nn.relu(output1)
    cal2 = output1

    output1 = tf.layers.conv3d(output1, 32, 3, padding='SAME', kernel_initializer=tf.contrib.layers.xavier_initializer(), name="G_conv5")
    output1 = tf.nn.relu(output1)
    feature = tf.reduce_mean(output1, axis=4, keepdims=True)
    feature = tf.layers.conv3d(feature, 1, 1, padding='SAME', kernel_initializer=tf.contrib.layers.xavier_initializer(), name="G_convAtt3")
    feature = tf.reshape(feature, [-1, w*h*d])
    feature = tf.nn.sigmoid(feature)
    feature = tf.reshape(feature, [-1, w, h, d, 1])
    feature = tf.layers.conv3d(feature, 1, 1, padding='SAME', kernel_initializer=tf.contrib.layers.xavier_initializer(), name="G_convAtt4")
    feature = tf.nn.relu(feature)
    output1 = output1*feature
    output1 = tf.layers.conv3d(output1, 32, 3, padding='SAME', kernel_initializer=tf.contrib.layers.xavier_initializer(), name="G_conv6")
    output1 = tf.nn.relu(output1)

    ##########################===============Decoder====================################################
    output1 = tf.layers.conv3d_transpose(output1, 32, 3, padding='SAME', kernel_initializer=tf.contrib.layers.xavier_initializer(), name="G_deconv1")
    output1 = tf.nn.relu(output1)
    feature = tf.reduce_mean(output1, axis=4, keepdims=True)
    feature = tf.layers.conv3d(feature, 1, 1, padding='SAME', kernel_initializer=tf.contrib.layers.xavier_initializer(), name="G_convAtt5")
    feature = tf.reshape(feature, [-1, w*h*d])
    feature = tf.nn.sigmoid(feature)
    feature = tf.reshape(feature, [-1, w, h, d, 1])
    feature = tf.layers.conv3d(feature, 1, 1, padding='SAME', kernel_initializer=tf.contrib.layers.xavier_initializer(), name="G_convAtt6")
    feature = tf.nn.relu(feature)
    output1 = output1*feature
    output1 = tf.layers.conv3d_transpose(output1, 32, 3, padding='SAME', kernel_initializer=tf.contrib.layers.xavier_initializer(), name="G_deconv2")
    output1 = tf.nn.relu(output1)
    output1=tf.concat([cal2,output1],axis=-1)

    output1 = tf.layers.conv3d_transpose(output1, 32, 3, padding='SAME', kernel_initializer=tf.contrib.layers.xavier_initializer(), name="G_deconv3")
    output1 = tf.nn.relu(output1)
    feature = tf.reduce_mean(output1, axis=4, keepdims=True)
    feature = tf.layers.conv3d(feature, 1, 1, padding='SAME', kernel_initializer=tf.contrib.layers.xavier_initializer(), name="G_convAtt7")
    feature = tf.reshape(feature, [-1, w*h*d])
    feature = tf.nn.sigmoid(feature)
    feature = tf.reshape(feature, [-1, w, h, d, 1])
    feature = tf.layers.conv3d(feature, 1, 1, padding='SAME', kernel_initializer=tf.contrib.layers.xavier_initializer(), name="G_convAtt8")
    feature = tf.nn.relu(feature)
    output1 = output1*feature
    output1 = tf.layers.conv3d_transpose(output1, 32, 3, padding='SAME', kernel_initializer=tf.contrib.layers.xavier_initializer(), name="G_deconv4")
    output1 = tf.nn.relu(output1)
    output1=tf.concat([cal1,output1],axis=-1)

    output1 = tf.layers.conv3d_transpose(output1, 32, 3, padding='SAME', kernel_initializer=tf.contrib.layers.xavier_initializer(), name="G_deconv5")
    output1 = tf.nn.relu(output1)
    output1 = tf.layers.conv3d_transpose(output1, 32, 3, padding='SAME', kernel_initializer=tf.contrib.layers.xavier_initializer(), name="G_deconv6")
    output1 = tf.nn.relu(output1)

    # reshape layer
    output1 = tf.layers.conv3d(output1, 1, 3, padding='SAME', kernel_initializer=tf.contrib.layers.xavier_initializer(), name="G_conv7")
    output1 = tf.nn.tanh(output1)

    # obtain clean image
    output = input -output1

    return output
