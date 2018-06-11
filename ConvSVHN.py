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

# Patch size at each convolutional layer
P1 = 6
P2 = 5
P3 = 4

# Stride at each convolutional layer
S1 = 1
S2 = 2
S3 = 2

# Output Channel Controls (Number of output channels per layer)
C1 = 12
C2 = 24
C3 = 48

# Fully connected layers neurons per layer
N1 = 100
N2 = 10 # Output layer

# Input images are 28x28x3 RBG images
X = tf.placeholder(tf.float32, [None, DIM, DIM, layers])

# Output format has yet to be determined, CIFAR-10 placeholder
Y_ = tf.placeholder(tf.float32, [None, N2])

# Step placeholder to feed back to learning rate
step = tf.placeholder(tf.int32)

# Declare our xavier initializer used for weights layers
weights_init = tf.contrib.layers.xavier_initializer()

# First convolutional layer
W_conv1 = tf.Variable(weights_init([P1,P1,3,C1]))
B_conv1 = tf.Variable(tf.zeros([C1]))
Y_conv1 = tf.nn.relu(tf.nn.conv2d(X, W_conv1, strides=[1, S1, S1, 1], padding='SAME') + B_conv1)

# Second convolutional layer
W_conv2 = tf.Variable(weights_init([P2,P2,C1,C2]))
B_conv2 = tf.Variable(tf.zeros([C2]))
Y_conv2 = tf.nn.relu(tf.nn.conv2d(Y_conv1, W_conv2, strides=[1, S2, S2, 1], padding='SAME') + B_conv2)

# Third convolutional layer
W_conv3 = tf.Variable(weights_init([P3,P3,C2,C3]))
B_conv3 = tf.Variable(tf.zeros([C3]))
Y_conv3 = tf.nn.relu(tf.nn.conv2d(Y_conv2, W_conv3, strides=[1, S3, S3, 1], padding='SAME') + B_conv3)

# First FA layer
reduced_size = int(((DIM / S1) / S2) / S3) # I changed this from (((28 / S1) / S2) / S3) because I figured the '28' was left over from MNIST
Y_conv2fc = tf.reshape(Y_conv3, shape=[-1, reduced_size * reduced_size * C3])
W_fa1 = tf.Variable(weights_init([reduced_size * reduced_size * C3, N1]))
B_fa1 = tf.Variable(tf.zeros([N1]))
Y_fa1 = tf.nn.relu(tf.matmul(Y_conv2fc, W_fa1) + B_fa1)

# Output FA layer
W_fa2 = tf.Variable(weights_init([N1, N2]))
B_fa2 = tf.Variable(tf.zeros([N2]))
Y_fa2_logits = tf.matmul(Y_fa1, W_fa2) + B_fa2
Y = tf.nn.softmax(Y_fa2_logits)

# Learning Rate and Optimizer
lr = final_lr + tf.train.exponential_decay(init_lr, step, lr_decay, 1/math.e)
loss = tf.nn.softmax_cross_entropy_with_logits(logits=Y_fa2_logits, labels=Y_)
optimizer = tf.train.AdamOptimizer(learning_rate=lr).minimize(loss)

# Checking accuracy
correct_pred = tf.equal(tf.argmax(Y, 1), tf.argmax(Y_, 1))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

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
    sess = tf.InteractiveSession()
    sess.run(tf.global_variables_initializer())

    now = datetime.utcnow().strftime("%Y%m%d%H%M%S")
    root_logdir = "tf_logs"
    logdir = "{}/run-{}".format(root_logdir, ("Codelab" + now))

    acc_summary_train = tf.summary.scalar('Accuracy_Training', accuracy)
    acc_summary_test = tf.summary.scalar('Accuracy_Test', accuracy)
    loss_summary_train = tf.summary.scalar('Loss_Training', loss)
    loss_summary_test = tf.summary.scalar('Loss_Test', loss)
    # weights_summary = tf.summary.histogram('Weights', allweights)
    # biases_summary = tf.summary.histogram('Biases', allbiases)
    # conv_act_summary = tf.summary.histogram('Convolutional Activations', conv_activations)
    # dens_act_summary = tf.summary.histogram('Dense Activations', dense_activations)

    file_writer = tf.summary.FileWriter(logdir, tf.get_default_graph())

    for i in range(num_of_batches):
        batch_X, batch_Y = sess.run(next_train)

        # Run training analytics every so many batches
        if i%25 == 0:
            acc, los, lr_now = sess.run([acc_summary_train, loss_summary_train, lr], {X: batch_X, Y_: batch_Y, step: i})
            file_writer.add_summary(los, i)
            file_writer.add_summary(acc, i)
            print("Training Analytics")
            print("LR: " + str(lr_now))
            print("Loss: " + str(los))
            print("Accuracy: " + str(acc))
        # Run test analytics every so many batches
        if i%100 == 0:
            TEST_DATA, TEST_LABELS = sess.run(next_test)
            a, l = sess.run([acc_summary_test, loss_summary_test], {X: TEST_DATA, Y_: TEST_LABELS})
            file_writer.add_summary(l, i)
            file_writer.add_summary(a, i)
            print("Testing Analytics")
            print("Loss: " + str(l))
            print("Accuracy: " + str(a))
            # acc_t, loss_t, lr_t = sess.run([accuracy, loss, lr], feed_dict:{X: TEST_DATA, Y_: TEST_LABELS})
        sess.run(optimizer, {X: batch_X, Y_:batch_Y, step: i})
    # After training loops are complete, execute a final validation step
    # All of the training we plan to do is now complete
    print("Final NN Test")
    # acc_t, loss_t = sess.run([accuracy, loss], feed_dict:{X: TEST_DATA, Y_: TEST_LABELS})


if __name__ == "__main__":
    main()
