#!/usr/bin/python

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys
sys.dont_write_bytecode = True

import tensorflow as tf
import numpy as np
import argparse
import papl
import os

import scipy.sparse as sp

argparser = argparse.ArgumentParser()
argparser.add_argument("-train", "--train", action="store_true", help="Train with 20000 iterations")
argparser.add_argument("-test", "--test", action="store_true", help="Run test")
argparser.add_argument("-d", "--deploy", action="store_true", help="Run deploy with seven.png")
args = argparser.parse_args()

from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets('/tmp/data/', one_hot=True)
if (args.train or args.test or args.deploy) == False:
	argparser.print_help()
	sys.exit()

#sess = tf.InteractiveSession()
sess = tf.InteractiveSession(config=tf.ConfigProto(device_count={'GPU':0}))

def load_weight(dense_w):
	for wl in dense_w.keys():
		fi = wl + "_p.dat"
		print(fi)
		w = np.genfromtxt(fi)
		weight_obj = dense_w[wl]
		weight_arr = np.reshape(w,weight_obj.eval().shape)
		sess.run(weight_obj.assign(weight_arr))
		print(weight_arr.shape)

conv1 = int(sum(1 for line in open("w_conv1_p.dat","r"))/(5*5))
print(conv1)
conv2 = int(sum(1 for line in open("w_conv2_p.dat","r"))/(5*5*conv1))
print(conv2)
fc1 = int(sum(1 for line in open("w_fc1_p.dat","r"))/(7*7*conv2))
print(fc1)
#fc2 = int(sum(1 for line in open("w_fc2_p.dat","r"))/10)
#print(fc2)

dense_w={
	"w_conv1": tf.Variable(tf.truncated_normal([5,5,1,conv1],stddev=0.1), name="w_conv1"),
	"b_conv1": tf.Variable(tf.constant(0.1,shape=[conv1]), name="b_conv1"),
	"w_conv2": tf.Variable(tf.truncated_normal([5,5,conv1,conv2],stddev=0.1), name="w_conv2"),
	"b_conv2": tf.Variable(tf.constant(0.1,shape=[conv2]), name="b_conv2"),
	"w_fc1": tf.Variable(tf.truncated_normal([7*7*conv2,fc1],stddev=0.1), name="w_fc1"),
	"b_fc1": tf.Variable(tf.constant(0.1,shape=[fc1]), name="b_fc1"),
	"w_fc2": tf.Variable(tf.truncated_normal([fc1,10],stddev=0.1), name="w_fc2"),
	"b_fc2": tf.Variable(tf.constant(0.1,shape=[10]), name="b_fc2")
}


def dense_cnn_model(weights):
	def conv2d(x, W):
		return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

	def max_pool_2x2(x):
		return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
							  strides=[1, 2, 2, 1], padding='SAME')

	x_image = tf.reshape(x, [-1,28,28,1])
	h_conv1 = tf.nn.relu(conv2d(x_image, weights["w_conv1"]) + weights["b_conv1"])
	tf.add_to_collection("in_conv1", x_image)
	h_pool1 = max_pool_2x2(h_conv1)
	tf.add_to_collection("in_conv2", h_pool1)
	h_conv2 = tf.nn.relu(conv2d(h_pool1, weights["w_conv2"]) + weights["b_conv2"])
	h_pool2 = max_pool_2x2(h_conv2)
	h_pool2_flat = tf.reshape(h_pool2, [-1, 7*7*conv2])
	tf.add_to_collection("in_fc1", h_pool2_flat)
	h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, weights["w_fc1"]) + weights["b_fc1"])
	h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)
	tf.add_to_collection("in_fc2", h_fc1_drop)
	y_conv=tf.nn.softmax(tf.matmul(h_fc1_drop, weights["w_fc2"]) + weights["b_fc2"])
	return y_conv

def test(y_infer, message="None."):
	correct_prediction = tf.equal(tf.argmax(y_infer,1), tf.argmax(y_,1))
	accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))

	# To avoid OOM, run validation with 500/10000 test dataset
	result = 0
	for i in range(20):
		batch = mnist.test.next_batch(500)
		result += accuracy.eval(feed_dict={x: batch[0],
										  y_: batch[1],
										  keep_prob: 1.0})
	result /= 20

	print(message+" %g\n" % result)
	return result

def imgread(path):
    tmp = papl.imread(path)
    img = np.zeros((28,28,1))
    img[:,:,0]=tmp[:,:,0]
    img = np.reshape(img, img.size)
    return img

# Construct a dense model
x = tf.placeholder("float", shape=[None, 784], name="x")
y_ = tf.placeholder("float", shape=[None, 10], name="y_")
keep_prob = tf.placeholder("float", name="keep_prob")

y_conv = dense_cnn_model(dense_w)
tf.add_to_collection("y_conv", y_conv)

saver = tf.train.Saver()

if args.train == True:
	# First round: Train baseline dense model
	cross_entropy = -tf.reduce_sum(y_*tf.log(tf.clip_by_value(y_conv,1e-10,1.0)))
	train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
	correct_prediction = tf.equal(tf.argmax(y_conv,1), tf.argmax(y_,1))
	accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
	tf.add_to_collection("accuracy", accuracy)

	sess.run(tf.initialize_all_variables())

	load_weight(dense_w)

	for i in range(100):
		batch = mnist.train.next_batch(50)
		if i%100 == 0:
			train_accuracy = accuracy.eval(feed_dict={
				x:batch[0], y_: batch[1], keep_prob: 1.0})
			print("step %d, training accuracy %g"%(i, train_accuracy))
		train_step.run(feed_dict={x: batch[0], y_: batch[1], keep_prob: 0.5})

	# Test
	score = test(y_conv, message="First-round prune-only test accuracy")
	papl.log("baseline_accuracy.log", score)
	
	# Save model objects to readable format
	papl.print_weight_vars(dense_w, papl.config.all_layer, papl.config.all_p_dat, show_zero=papl.config.show_zero)
	# Save model objects to serialized format
	saver.save(sess, os.path.join(os.getcwd(), "model_ckpt_sparse"))

if args.test == True:
	import time

	sess.run(tf.initialize_all_variables())
	load_weight(dense_w)

	y_conv = tf.get_collection("y_conv")[0]
	correct_prediction = tf.equal(tf.argmax(y_conv,1), tf.argmax(y_,1))
	accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
	b = time.time()
	result = 0
	for i in range(1000):
		batch = mnist.test.next_batch(10)
		result += sess.run(accuracy, feed_dict={"x:0": batch[0],
				"y_:0": batch[1],
				"keep_prob:0": 1.0})
	result /= 1000
	a = time.time()
	print("Test accuracy: %g" % result)
	print("Time: ",(a-b))

if args.deploy == True:
	import time
	sess.run(tf.initialize_all_variables())
	load_weight(dense_w)

	img = imgread('seven.png')
	y_conv = tf.get_collection("y_conv")[0]

	b = time.time()
	for i in range(100):
		result = sess.run(tf.argmax(y_conv,1), feed_dict={"x:0":[img], "y_:0":mnist.test.labels, "keep_prob:0": 1.0})
	a = time.time()

	print("Output: %s" % result)
	print("Time: %s s" % ((a-b)/100))
