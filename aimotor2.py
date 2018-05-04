import tensorflow as tf
import pandas as pd
import numpy as np
import random
import os
import matplotlib.pyplot as plt


"""
For training data of Maxwell RMxprt by using Neural-network Supervisor
"""

data = pd.read_csv('test_y_1.csv')
#Read data that you want to train

def MaxMinNor(x, Max, Min):
    """
    function of max min normalization
    """
    x = (x - Min) / (Max - Min)
    return x


for i in range(0, 8):
    if i != 5 and i != 6:
        data.loc[:, data.columns[i]] = MaxMinNor(data.loc[:, data.columns[i]], np.max(data.loc[:, data.columns[i]]),np.min(data.loc[:, data.columns[i]]))


# prepare training data(70%) and test data(30%) by random sampling
index_1000_10000 = []
for i in range(0, len(data)):
    if data.iloc[i, 8] > 1000 and data.iloc[i, 8] <= 10000:
        index_1000_10000.append(i)
k = random.sample(index_1000_10000, int(len(index_1000_10000)*0.7))
m = []
for i in range(0, len(index_1000_10000)):
    if not index_1000_10000[i] in k:
        m.append(index_1000_10000[i])
data_train_x = data.iloc[k, [1, 3, 4, 7]].values
data_train_y = data.iloc[k, [8]].values
data_test_x = data.iloc[m, [1, 3, 4, 7]].values
data_test_y = data.iloc[m, [8]].values
#####################################################################

x_feeds = tf.placeholder(tf.float32, shape = [None, 4])
y_feeds = tf.placeholder(tf.float32, shape = [None, 1])

w = tf.Variable(tf.random_normal([4, 4]))
b = tf.Variable(tf.random_normal([4]))
hidden_layer = tf.add(tf.matmul(x_feeds, w), b)
hidden_layer_ac = tf.nn.relu(hidden_layer)

w1 = tf.Variable(tf.random_normal([4, 1]))
b1 = tf.Variable(tf.random_normal([1]))
outer_layer = tf.add(tf.matmul(hidden_layer_ac, w1), b1)

loss = tf.sqrt(tf.reduce_mean(tf.square(outer_layer - y_feeds)))
tra_res = tf.train.GradientDescentOptimizer(0.05).minimize(loss)
init = tf.global_variables_initializer()
l = []

batch_size = 100
training_epoch = 4000
display_step = 20
avg_set = []
epoch_set = []

with tf.Session() as sess:
    sess.run(init)
    for epoch in range(training_epoch):
        avg_cost = 0.
        total_batch = int(len(data_train_x) / batch_size)
        for i in range(total_batch):
            start = i * batch_size
            end = min(start + batch_size, len(data_train_x))
            batch_xs = data_train_x[start:end, :]
            batch_ys = data_train_y[start:end, :]
            sess.run(tra_res, feed_dict= {x_feeds: batch_xs, y_feeds: batch_ys})
            avg_cost += sess.run(loss, feed_dict={x_feeds: batch_xs, y_feeds: batch_ys}) / total_batch
        if epoch % display_step == 0:
            print ("epoch:", '%04d' % (epoch+1),\
                   "loss=", "{:.9f}".format(avg_cost))
            avg_set.append(avg_cost)
            epoch_set.append(epoch + 1)
        if (epoch+1) == 4000:
            print ("training phase finished")
            plt.plot(epoch_set, avg_set, 'o',\
                     label='logistic regression training phase')
            plt.ylabel('loss')
            plt.xlabel('epoch')
            plt.legend()
            plt.show()
    train_error_percent = abs(
    data_train_y - sess.run(outer_layer, feed_dict={x_feeds: data_train_x})) / data_train_y
    train_accuracy = 1 - np.mean(train_error_percent)
    print(train_accuracy)

    test_error_percent = abs(data_test_y - sess.run(outer_layer, feed_dict={x_feeds: data_test_x})) / data_test_y
    test_accuracy = 1 - np.mean(test_error_percent)
    print(test_accuracy)
#Start training and draw loss function diagram 



#Calculus the accuracy of test data

