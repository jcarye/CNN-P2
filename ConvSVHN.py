# Street View House Number (SVHN) DataSet

import tensorflow as tf
import math

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
X = tf.placeholder(tf.float32, [None, 28, 28, 3])

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
reduced_size = (((28 / S1) / S2) / S3)
Y_conv2fc = tf.reshape(Y_conv3, shape=[-1,int(reduced_size)])
W_fa1 = tf.Variable(weights_init([int(reduced_size), N1]))
B_fa1 = tf.Variable(tf.zeros([N1]))
Y_fa1 = tf.nn.relu(tf.matmul(Y_conv2fc, W_fa1) + B_fa1)

# Output FA layer
W_fa2 = tf.Variable(weights_init([N1, N2]))
B_fa2 = tf.Variable(tf.zeros([N2]))
Y_fa2_logits = tf.matmul(Y_fa1, W_fa2) + B_fa2
Y = tf.nn.softmax(Y_fa2_logits)

# Learning Rate and Optimizer
lr = final_lr + tf.train.exponential_decay(init_lr, step, lr_decay, 1/math.e)
loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits = Y_fa2_logits, labels = Y_))
optimizer = tf.train.AdamOptimizer(learning_rate=lr).minimize(loss)

# Checking accuracy
correct_pred = tf.equal(tf.argmax(Y, 1), tf.argmax(Y_, 1))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

def main():
    # Initialization
    sess = tf.InteractiveSession()
    sess.run(tf.global_variables_initializer())

    for i in range(num_of_batches):
        # Calculate new learning rate (if needed, may not be)
        # ## MAYBE NOT NEEDED## lr = final_lr + tf.train.exponential_decay(init_lr, step, lr_decay, 1/math.e)
        # print("Learning rate is: {}".format(lr))
        # _, lr_now = sess.run([optimizer, lr], {X: batch_X, Y_: batch_Y, step: i})
        # Run training analytics every so many batches
        if i%25 == 0:
            print("Training Analytics")
            # print("LR: "+str(lr_now))
        # Run test analytics every so many batches
        if i%100 == 0:
            print("Testing Analytics")
            # acc_t, loss_t, lr_t = sess.run([accuracy, loss, lr], feed_dict:{X: TEST_DATA, Y_: TEST_LABELS})
    # After training loops are complete, execute a final validation step
    # All of the training we plan to do is now complete
    print("Final NN Test")
    # acc_t, loss_t = sess.run([accuracy, loss], feed_dict:{X: TEST_DATA, Y_: TEST_LABELS})


if __name__ == "__main__":
    main()
