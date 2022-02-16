import sys
import os
import numpy as np
import math
from operator import itemgetter, attrgetter, methodcaller
import tensorflow as tf
from tensorflow.contrib import rnn
from tensorflow.contrib.rnn import DropoutWrapper
import random
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import f1_score
from sklearn.metrics import cohen_kappa_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.utils import shuffle
from sklearn.metrics import confusion_matrix
import time
import scipy.io as scio
import mat73


def MSCNN(x):
    conv1 = tf.layers.conv2d(
        inputs=x,
        filters=256,
        kernel_size=[3, 3],
        padding="valid",
        activation=tf.nn.relu)

    conv1 = tf.layers.batch_normalization(conv1)

    conv2 = tf.layers.conv2d(
        inputs=conv1,
        filters=512,
        kernel_size=[3, 3],
        padding="valid",
        activation=tf.nn.relu)

    conv2 = tf.layers.batch_normalization(conv2)

    conv3 = tf.layers.conv2d(
        inputs=conv2,
        filters=1024,
        kernel_size=[3, 3],
        padding="valid",
        activation=tf.nn.relu)

    conv3 = tf.layers.batch_normalization(conv3)

    cnn = tf.reduce_max(conv3, [1, 2])

    return cnn


def PCNN(x):
    conv1 = tf.layers.conv2d(
        inputs=x,
        filters=128,
        kernel_size=[3, 3],
        padding="valid",
        activation=tf.nn.relu)

    conv1 = tf.layers.batch_normalization(conv1)
    conv1 = tf.layers.max_pooling2d(inputs=conv1, pool_size=[2, 2], strides=2)

    conv2 = tf.layers.conv2d(
        inputs=conv1,
        filters=256,
        kernel_size=[3, 3],
        padding="valid",
        activation=tf.nn.relu)

    conv2 = tf.layers.batch_normalization(conv2)
    conv2 = tf.layers.max_pooling2d(inputs=conv2, pool_size=[2, 2], strides=2)

    conv3 = tf.layers.conv2d(
        inputs=conv2,
        filters=512,
        kernel_size=[3, 3],
        padding="valid",
        activation=tf.nn.relu)

    conv3 = tf.layers.batch_normalization(conv3)

    cnn = tf.reduce_max(conv3, [1, 2])

    return cnn


def getBatch(X, Y, i, batch_size):
    start_id = i * batch_size
    t = (i + 1) * batch_size
    end_id = min((i + 1) * batch_size, X.shape[0])
    batch_x = X[start_id:end_id]
    batch_y = Y[start_id:end_id]
    return batch_x, batch_y


def getLabelFormat(Y):
    new_Y = []
    vals = np.unique(np.array(Y))
    for el in Y:
        t = np.zeros(len(vals))
        t[el] = 1.0
        new_Y.append(t)
    return np.array(new_Y)


def classifier(vec_feat, nclasses):
    cl = tf.layers.dense(vec_feat, 512, activation=tf.nn.relu)
    cl = tf.layers.dense(cl, 512, activation=tf.nn.relu)
    cl = tf.layers.dense(cl, nclasses, activation=None)
    return cl


def getPrediction(x_cnn_1, x_cnn_2, nclasses, dropout, is_training):
    features_learnt = None
    prediction = None

    vec_cnn_1 = PCNN(x_cnn_1)
    vec_cnn_1 = tf.layers.dropout(vec_cnn_1, rate=dropout, training=is_training)

    vec_cnn_2 = MSCNN(x_cnn_2)
    vec_cnn_2 = tf.layers.dropout(vec_cnn_2, rate=dropout, training=is_training)

    features_learnt = tf.concat([vec_cnn_1, vec_cnn_2], axis=1, name="features")
    pred_full = tf.layers.dense(features_learnt, nclasses, activation=None)

    return [pred_full, features_learnt]


def data_augmentation(label_train, vhsr_train_1, vhsr_train_2):
    new_label_train = []
    new_vhsr_train_1 = []
    new_vhsr_train_2 = []
    for i in range(vhsr_train_1.shape[0]):
        img_1 = vhsr_train_1[i]
        img_2 = vhsr_train_2[i]
        # ROTATE
        new_vhsr_train_1.append(img_1)
        new_vhsr_train_2.append(img_2)
        new_label_train.append(label_train[i])

        for j in range(1, 4):
            if (random.random() < 0.3):
                img_rotate_1 = np.rot90(img_1, k=j, axes=(0, 1))
                img_rotate_2 = np.rot90(img_2, k=j, axes=(0, 1))
                new_label_train.append(label_train[i])
                new_vhsr_train_1.append(img_rotate_1)
                new_vhsr_train_2.append(img_rotate_2)

        # FLIPPING
        for j in range(2):
            if (random.random() < 0.3):
                img_flip_1 = np.rot90(img_1, j)
                img_flip_2 = np.rot90(img_2, j)
                new_label_train.append(label_train[i])
                # new_ts_train.append( ts_train[i])
                new_vhsr_train_1.append(img_flip_1)
                new_vhsr_train_2.append(img_flip_2)

        # TRANSPOSE
        if (random.random() < 0.3):
            t_img_1 = np.transpose(img_1, (1, 0, 2))
            t_img_2 = np.transpose(img_2, (1, 0, 2))
            new_label_train.append(label_train[i])
            new_vhsr_train_1.append(t_img_1)
            new_vhsr_train_2.append(t_img_2)

    return np.array(new_label_train), np.array(new_vhsr_train_1), np.array(new_vhsr_train_2)


# Model parameters
batchsz = 100
hm_epochs = 100
n_levels_lstm = 1
# dropout = 0.2


# Data INformation
patch_window_1 = 26  # P-CNN分支（即image分支）
n_channels_1 = 1

patch_window_2 = 14  # MS-CNN分支（即glcm分支）
n_channels_2 = 1


data1 = mat73.loadmat('D:\\Z_CD\\Two_Branch_CNN\\Guangzhou\\data preparation\\Guangzhou_image_non_trn.mat')
data2 = mat73.loadmat('D:\\Z_CD\\Two_Branch_CNN\\Guangzhou\\data preparation\\Guangzhou_3DGLCM_non_trn_pad0.mat')
label = mat73.loadmat('D:\\Z_CD\\Two_Branch_CNN\\Guangzhou\\data preparation\\Guangzhou_Pseudo_label.mat')
vhsr_train_pan = data1['image_trn']
vhsr_train_ms = data2['glcm_trn']

vhsr_train_pan = vhsr_train_pan.transpose(2, 0, 1)
vhsr_train_ms = vhsr_train_ms.transpose(2, 0, 1)
vhsr_train_pan = vhsr_train_pan[:, :, :, np.newaxis]
vhsr_train_ms = vhsr_train_ms[:, :, :, np.newaxis]

label_train = label['label']
label_train = label_train.astype(np.int)
nclasses = len(np.unique(label_train))

# print
# "DATA AUGMENTATION:"
# label_train, vhsr_train_pan, vhsr_train_ms = data_augmentation(label_train, vhsr_train_pan, vhsr_train_ms)
# label_train = label_train.astype('int')

x_cnn_1 = tf.placeholder("float", [None, patch_window_1, patch_window_1, n_channels_1], name="x_cnn_1")
x_cnn_2 = tf.placeholder("float", [None, patch_window_2, patch_window_2, n_channels_2], name="x_cnn_2")
y = tf.placeholder("float", [None, nclasses], name="y")

learning_rate = tf.placeholder(tf.float32, shape=[], name="learning_rate")
is_training = tf.placeholder(tf.bool, shape=(), name="is_training")
dropout = tf.placeholder(tf.float32, shape=(), name="drop_rate")

sess = tf.InteractiveSession()

pred_full, features_learnt = getPrediction(x_cnn_1, x_cnn_2, nclasses, dropout, is_training)
pred_tot = pred_full
testPrediction = tf.argmax(pred_tot, 1, name="prediction")
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y, logits=pred_full))
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)
correct = tf.equal(tf.argmax(pred_tot, 1), tf.argmax(y, 1))
accuracy = tf.reduce_mean(tf.cast(correct, tf.float64))

tf.global_variables_initializer().run()
saver = tf.train.Saver()

classes = label_train
train_y = getLabelFormat(classes)

vhsr_train_1 = vhsr_train_pan
iterations = vhsr_train_1.shape[0] / batchsz
iterations = int(iterations)
print("iterations %d" % iterations)

if vhsr_train_1.shape[0] % batchsz != 0:
    iterations += 1

best_loss = sys.float_info.max

print("开始训练...")
for e in range(hm_epochs):
    ee = e+1
    print("epoch %d / " % ee, "epoch %d" % hm_epochs)
    lossi = 0
    accS = 0

    vhsr_train_pan, vhsr_train_ms, train_y = shuffle(vhsr_train_pan, vhsr_train_ms, train_y, random_state=0)
    start = time.time()
    for ibatch in range(iterations):
        batch_cnn_x_1, _ = getBatch(vhsr_train_pan, train_y, ibatch, batchsz)
        batch_cnn_x_2, batch_y = getBatch(vhsr_train_ms, train_y, ibatch, batchsz)
        acc, _, loss = sess.run([accuracy, optimizer, cost], feed_dict={x_cnn_1: batch_cnn_x_1,
                                                                        x_cnn_2: batch_cnn_x_2,
                                                                        y: batch_y,
                                                                        is_training: True,
                                                                        dropout: 0.4,
                                                                        learning_rate: 0.0002})

        lossi += loss
        accS += acc

        del batch_cnn_x_1
        del batch_cnn_x_2
        del batch_y
    c_loss = lossi / iterations
    end = time.time()
    print("cost=", "{:.9f}".format(c_loss))
    print("此epoch的运行时间为：", end - start)

    if c_loss < best_loss:
        save_path = saver.save(sess, "./models_2non_pad0/model_%d" % ee)
        print("Model saved in path: %s" % save_path)
        best_loss = c_loss

print("训练结束")

print("开始测试...")
ckpt = tf.train.get_checkpoint_state("models_2non_pad0/checkpoint")
if ckpt and ckpt.model_checkpoint_path:
    saver.restore(sess, ckpt.model_checkpoint_path)

test1 = mat73.loadmat('D:\\Z_CD\\Two_Branch_CNN\\Guangzhou\\data preparation\\Guangzhou_image_non_tst.mat')
test2 = mat73.loadmat('D:\\Z_CD\\Two_Branch_CNN\\Guangzhou\\data preparation\\Guangzhou_3DGLCM_non_tst_pad0.mat')
test1 = test1['image_tst']
test2 = test2['glcm_tst']
test1 = test1.transpose(2, 0, 1)
test2 = test2.transpose(2, 0, 1)
test1 = test1[:, :, :, np.newaxis]
test2 = test2[:, :, :, np.newaxis]


test_batch = 100
test_iterations = test1.shape[0] / test_batch
test_iterations = math.ceil(test_iterations)
print("test_iterations %d" % test_iterations)
test_start = time.time()
data_vlist = []
for ibatch1 in range(test_iterations):
    test_cnn_x_1, _ = getBatch(test1, train_y, ibatch1, test_batch)
    test_cnn_x_2, batch_y = getBatch(test2, train_y, ibatch1, test_batch)
    res1 = sess.run(pred_tot, feed_dict={x_cnn_1: test_cnn_x_1,
                                         x_cnn_2: test_cnn_x_2,
                                         is_training: False,
                                         dropout: 0})  # res1为每个test_batch的测试结果  # 测试时将dropout设置为0
    del test_cnn_x_1  # 删除
    del test_cnn_x_2
    del batch_y
    res1 = np.argmax(res1, 1)
    data_vlist.append(res1)

res = np.hstack(data_vlist)

test_end = time.time()

scio.savemat('Guangzhou_test_result_2non_pad0.mat', {"test_result": res})

print("测试时间为：", test_end - test_start)
print("测试结束！！！")

