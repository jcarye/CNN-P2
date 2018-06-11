# Street View House Number (SVHN) DataSet

import tensorflow as tf
import math
import os
import urllib
import numpy as np
import collections

from scipy.io import loadmat
from datetime import datetime
# from tensorflow.python.framework import dtypes
# import tensorflow.examples.tutorials.mnist as mnist
# from tensorflow.python.framework import random_seed

###############
# Global Vars #
###############
print("Tensorflow version " + tf.__version__)
# Size of inputs

DIM = 32
layers = 3

############
# GET DATA #
############

# This area is a work in progress
# TODO: get the data into one of the TF classes that let's us easily generate batches
# Possible COAs: DataSet method defined in tensorflow.examples.tutorials.mnist or
#       maybe tf.data.Dataset
# Data currently auto downloads and imports if not already present
# Labels are currently in One-Hot format but may need to be in integer format depending
#       on how we shoehorn the dataset class

# Dataset = collections.namedtuple('Dataset', ['data', 'target'])
# Datasets = collections.namedtuple('Datasets', ['train', 'validation', 'test'])

DOWNLOADS_DIR = 'data'

if not os.path.exists(DOWNLOADS_DIR):
    try:
        print('Making Dir')
        os.mkdir(DOWNLOADS_DIR)
    except OSError as exc:
        raise

url="http://ufldl.stanford.edu/housenumbers/"

filenames = ["train_32x32.mat", "test_32x32.mat"]

for f in filenames:
    link = url + f
    filename = os.path.join(DOWNLOADS_DIR, f)
    if not os.path.isfile(filename):
        print('Downloading: ' + filename)
        try:
            urllib.request.urlretrieve(link, filename)
        except Exception as inst:
            print(inst)
            print('   Encounterd unknown error. Continuing.')

testdata = loadmat('./data/test_32x32.mat')
traindata = loadmat('./data/train_32x32.mat')

test_images = testdata['X']
test_labels = tf.one_hot((testdata['y'])[:, 0], 10)

train_images = traindata['X']
train_labels = tf.one_hot((traindata['y'])[:, 0], 10)

##################
# NORMALIZE DATA #
##################


def norm_images(images):
    rows = images.shape[0]
    columns = images.shape[1]
    channels = images.shape[2]
    num = images.shape[3]
    norm_array = np.empty(shape=(num, rows, columns, channels), dtype=np.float32)
    for x in range(0, num):
        image = images[:, :, :, x]
        norm_vec = (255-image)/255.0
        norm_vec -= np.mean(norm_vec, axis=0)  # Are we losing skew data here?  Look into this.
        norm_array[x] = norm_vec
    return norm_array


############################
# TRAINING HYPERPARAMETERS #
############################
mbatch_size = 100
num_of_batches = 6000
init_lr = 0.02 # Initial learning rate
final_lr = 0.0001 # Final decayed learning rate
lr_decay = 2000 # Number of steps to decay over

########################################################
# Create the shape of our Convolutional Neural Network #
########################################################
# input X: 28x28 grayscale images, the first dimension (None) will index the images in the mini-batch
X = tf.placeholder(tf.float32, [None, DIM, DIM, layers])
# correct answers will go here
Y_ = tf.placeholder(tf.float32, [None, 10])
# test flag for batch norm
tst = tf.placeholder(tf.bool)
iter = tf.placeholder(tf.int32)
# dropout probability
pkeep = tf.placeholder(tf.float32)
pkeep_conv = tf.placeholder(tf.float32)

def batchnorm(Ylogits, is_test, iteration, offset, convolutional=False):
    exp_moving_avg = tf.train.ExponentialMovingAverage(0.999, iteration) # adding the iteration prevents from averaging across non-existing iterations
    bnepsilon = 1e-5
    if convolutional:
        mean, variance = tf.nn.moments(Ylogits, [0, 1, 2])
    else:
        mean, variance = tf.nn.moments(Ylogits, [0])
    update_moving_averages = exp_moving_avg.apply([mean, variance])
    m = tf.cond(is_test, lambda: exp_moving_avg.average(mean), lambda: mean)
    v = tf.cond(is_test, lambda: exp_moving_avg.average(variance), lambda: variance)
    Ybn = tf.nn.batch_normalization(Ylogits, m, v, offset, None, bnepsilon)
    return Ybn, update_moving_averages

def no_batchnorm(Ylogits, is_test, iteration, offset, convolutional=False):
    return Ylogits, tf.no_op()

def compatible_convolutional_noise_shape(Y):
    noiseshape = tf.shape(Y)
    noiseshape = noiseshape * tf.constant([1,0,0,1]) + tf.constant([0,1,1,0])
    return noiseshape

# three convolutional layers with their channel counts, and a
# fully connected layer (tha last layer has 10 softmax neurons)
K = 24  # first convolutional layer output depth
L = 48  # second convolutional layer output depth
M = 64  # third convolutional layer
N = 200  # fully connected layer

W1 = tf.Variable(tf.truncated_normal([6, 6, layers, K], stddev=0.1))  # 6x6 patch, 1 input channel, K output channels
B1 = tf.Variable(tf.constant(0.1, tf.float32, [K]))
W2 = tf.Variable(tf.truncated_normal([5, 5, K, L], stddev=0.1))
B2 = tf.Variable(tf.constant(0.1, tf.float32, [L]))
W3 = tf.Variable(tf.truncated_normal([4, 4, L, M], stddev=0.1))
B3 = tf.Variable(tf.constant(0.1, tf.float32, [M]))

W4 = tf.Variable(tf.truncated_normal([8 * 8 * M, N], stddev=0.1))
B4 = tf.Variable(tf.constant(0.1, tf.float32, [N]))
W5 = tf.Variable(tf.truncated_normal([N, 10], stddev=0.1))
B5 = tf.Variable(tf.constant(0.1, tf.float32, [10]))

# The model
# batch norm scaling is not useful with relus
# batch norm offsets are used instead of biases
stride = 1  # output is 32x32
Y1l = tf.nn.conv2d(X, W1, strides=[1, stride, stride, 1], padding='SAME')
Y1bn, update_ema1 = batchnorm(Y1l, tst, iter, B1, convolutional=True)
Y1r = tf.nn.relu(Y1bn)
Y1 = tf.nn.dropout(Y1r, pkeep_conv, compatible_convolutional_noise_shape(Y1r))
stride = 2  # output is 16x16
Y2l = tf.nn.conv2d(Y1, W2, strides=[1, stride, stride, 1], padding='SAME')
Y2bn, update_ema2 = batchnorm(Y2l, tst, iter, B2, convolutional=True)
Y2r = tf.nn.relu(Y2bn)
Y2 = tf.nn.dropout(Y2r, pkeep_conv, compatible_convolutional_noise_shape(Y2r))
stride = 2  # output is 8x8
Y3l = tf.nn.conv2d(Y2, W3, strides=[1, stride, stride, 1], padding='SAME')
Y3bn, update_ema3 = batchnorm(Y3l, tst, iter, B3, convolutional=True)
Y3r = tf.nn.relu(Y3bn)
Y3 = tf.nn.dropout(Y3r, pkeep_conv, compatible_convolutional_noise_shape(Y3r))

# reshape the output from the third convolution for the fully connected layer
YY = tf.reshape(Y3, shape=[-1, 8 * 8 * M])

Y4l = tf.matmul(YY, W4)
Y4bn, update_ema4 = batchnorm(Y4l, tst, iter, B4)
Y4r = tf.nn.relu(Y4bn)
Y4 = tf.nn.dropout(Y4r, pkeep)
Ylogits = tf.matmul(Y4, W5) + B5
Y = tf.nn.softmax(Ylogits)

update_ema = tf.group(update_ema1, update_ema2, update_ema3, update_ema4)

# cross-entropy loss function (= -sum(Y_i * log(Yi)) ), normalised for batches of 100  images
# TensorFlow provides the softmax_cross_entropy_with_logits function to avoid numerical stability
# problems with log(0) which is NaN
cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=Ylogits, labels=Y_)
cross_entropy = tf.reduce_mean(cross_entropy)*100

# accuracy of the trained model, between 0 (worst) and 1 (best)
correct_prediction = tf.equal(tf.argmax(Y, 1), tf.argmax(Y_, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

# matplotlib visualisation
allweights = tf.concat([tf.reshape(W1, [-1]), tf.reshape(W2, [-1]), tf.reshape(W3, [-1]), tf.reshape(W4, [-1]), tf.reshape(W5, [-1])], 0)
allbiases  = tf.concat([tf.reshape(B1, [-1]), tf.reshape(B2, [-1]), tf.reshape(B3, [-1]), tf.reshape(B4, [-1]), tf.reshape(B5, [-1])], 0)
conv_activations = tf.concat([tf.reshape(tf.reduce_max(Y1r, [0]), [-1]), tf.reshape(tf.reduce_max(Y2r, [0]), [-1]), tf.reshape(tf.reduce_max(Y3r, [0]), [-1])], 0)
dense_activations = tf.reduce_max(Y4r, [0])
# I = tensorflowvisu.tf_format_mnist_images(X, Y, Y_)
# It = tensorflowvisu.tf_format_mnist_images(X, Y, Y_, 1000, lines=25)
# datavis = tensorflowvisu.MnistDataVis(title4="batch-max conv activation", title5="batch-max dense activations", histogram4colornum=2, histogram5colornum=2)

# training step
# the learning rate is: # 0.0001 + 0.03 * (1/e)^(step/1000)), i.e. exponential decay from 0.03->0.0001
lr = 0.0001 +  tf.train.exponential_decay(0.02, iter, 1600, 1/math.e)
train_step = tf.train.AdamOptimizer(lr).minimize(cross_entropy)


def main():
    test_inputs = norm_images(test_images)
    train_inputs = norm_images(train_images)

    # validation_inputs = train_inputs[:13000]
    # validation_labels = train_labels[:13000]
    # train_inputs = train_inputs[13000:]
    # train_labels = train_labels[13000:]

    train_dataset = tf.data.Dataset.from_tensor_slices((train_inputs, train_labels))
    train_batch = train_dataset.repeat().batch(mbatch_size)
    train_iterator = train_batch.make_one_shot_iterator()
    next_train = train_iterator.get_next()

    # validation_dataset = tf.data.Dataset.from_tensor_slices((validation_inputs, validation_labels))
    test_dataset = tf.data.Dataset.from_tensor_slices((test_inputs, test_labels))
    test_batch = test_dataset.repeat().batch(mbatch_size)
    test_iterator = test_batch.make_one_shot_iterator()
    next_test = test_iterator.get_next()

    # Initialization
    # init
    init = tf.global_variables_initializer()
    sess = tf.Session()
    sess.run(init)

    now = datetime.utcnow().strftime("%Y%m%d%H%M%S")
    root_logdir = "tf_logs"
    logdir = "{}/run-{}".format(root_logdir, ("Codelab" + now))

    acc_summary_train = tf.summary.scalar('Accuracy_Training', accuracy)
    acc_summary_test = tf.summary.scalar('Accuracy_Test', accuracy)
    loss_summary_train = tf.summary.scalar('Loss_Training', cross_entropy)
    loss_summary_test = tf.summary.scalar('Loss_Test', cross_entropy)
    # weights_summary = tf.summary.histogram('Weights', allweights)
    # biases_summary = tf.summary.histogram('Biases', allbiases)
    # conv_act_summary = tf.summary.histogram('Convolutional Activations', conv_activations)
    # dens_act_summary = tf.summary.histogram('Dense Activations', dense_activations)

    file_writer = tf.summary.FileWriter(logdir, tf.get_default_graph())

    for i in range(num_of_batches):
        batch_X, batch_Y = sess.run(next_train)

        # Run training analytics every so many batches
        if i%25 == 0:
            a, c, l = sess.run([acc_summary_train,
                                loss_summary_train,
                                lr],
                               feed_dict={X: batch_X,
                                          Y_: batch_Y,
                                          iter: i,
                                          tst: False,
                                          pkeep: 1.0,
                                          pkeep_conv: 1.0})
            file_writer.add_summary(c, i)
            file_writer.add_summary(a, i)
            # file_writer.add_summary(w_str, i)
            # file_writer.add_summary(b_str, i)
            # file_writer.add_summary(ca, i)
            # file_writer.add_summary(da, i)
            print(str(i) + ": accuracy:" + str(a) + " loss: " + str(c) + " (lr:" + str(l) + ")")
        # Run test analytics every so many batches
        if i%100 == 0:
            TEST_DATA, TEST_LABELS = sess.run(next_test)
            a, c = sess.run([acc_summary_test,
                             loss_summary_test],
                            feed_dict={X: TEST_DATA,
                                       Y_: TEST_LABELS,
                                       tst: True,
                                       pkeep: 1.0,
                                       pkeep_conv: 1.0})
            file_writer.add_summary(c, i)
            file_writer.add_summary(a, i)
            print(str(i) + ": ********* epoch " +
                  str(i) +
                  " ********* test accuracy:" +
                  str(a) +
                  " test loss: " +
                  str(c))
            # datavis.append_test_curves_data(i, a, c)
            # datavis.update_image2(im)
        sess.run(train_step, {X: batch_X, Y_: batch_Y, tst: False, iter: i, pkeep: 0.75, pkeep_conv: 1.0})
        sess.run(update_ema, {X: batch_X, Y_: batch_Y, tst: False, iter: i, pkeep: 1.0, pkeep_conv: 1.0})
    # After training loops are complete, execute a final validation step
    # All of the training we plan to do is now complete
    print("Final NN Test")
    # acc_t, loss_t = sess.run([accuracy, loss], feed_dict:{X: TEST_DATA, Y_: TEST_LABELS})


if __name__ == "__main__":
    main()
