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
argparser.add_argument("-1", "--first_round", action="store_true",
	help="Run 1st-round: train with 20000 iterations")
argparser.add_argument("-2", "--second_round", action="store_true",
	help="Run 2nd-round: apply pruning and additional retraining")
argparser.add_argument("-3", "--third_round", action="store_true",
	help="Run 3rd-round: apply iterative pruning and additional retraining")
argparser.add_argument("-4", "--fourth_round", action="store_true",
	help="Run 4th-round: apply iterative pruning for each layer and retraining")
argparser.add_argument("-5", "--fifth_round", action="store_true",
	help="Run 5th-round: structure pruning and retraining")
argparser.add_argument("-6", "--sixth_round", action="store_true",
	help="Run 6th-round: transform model to a sparse format and save it")
argparser.add_argument("-m", "--checkpoint", default="./model_ckpt_dense",
	help="Target checkpoint model file for 2nd, 3rd, 4th, 5th round")
argparser.add_argument("r", type=float, help="Pruning ratio")
args = argparser.parse_args()

from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets('/tmp/data/', one_hot=True)
if (args.first_round or args.second_round or args.third_round or args.fourth_round or args.fifth_round or args.sixth_round) == False:
	argparser.print_help()
	sys.exit()

sess = tf.InteractiveSession()

def apply_prune(weights):
	dict_nzidx = {}

	for target in papl.config.target_layer:
		wl = "w_" + target
		print(weights[wl].eval().shape)
		w_= np.abs(weights[wl].eval().reshape(-1))
		w_sort = np.sort(w_)
		th = w_sort[int(len(w_sort)*args.r)]
		print(wl + " threshold:\t" + str(th))

		weight_obj = weights[wl]
		weight_arr = weight_obj.eval()

		weight_arr, w_nzidx, w_nnz = papl.prune_dense(weight_arr, name=wl,thresh=th)

		dict_nzidx[wl] = w_nzidx
		sess.run(weight_obj.assign(weight_arr))
	return dict_nzidx

def apply_prune_grad(grads,weights):
	dict_nzidx = {}

	for target in papl.config.target_layer:
		wl = "w_" + target
		print(grads[wl].shape)
		grads[wl] = np.random.random_sample((grads[wl].shape)) # random pruning
		w_= np.abs(grads[wl].reshape(-1))
		w_sort = np.sort(w_)
		th = w_sort[int(len(w_sort)*args.r)]
		print(wl + " threshold:\t" + str(th))

		grad_arr = grads[wl]
		under_threshold = abs(grad_arr) < th

		weight_obj = weights[wl]
		weight_arr = weight_obj.eval()
		weight_arr[under_threshold] = 0
		count = np.sum(under_threshold)
		print("Non-zero count (%s): %s" % (wl, weight_arr.size - count))

		dict_nzidx[wl] = ~under_threshold
		sess.run(weight_obj.assign(weight_arr))
	return dict_nzidx

def apply_prune_channel(weights):
	dict_nzidx = {}

	target = "conv1"
	wl = "w_" + target
	w_= np.abs(weights[wl].eval())
	node_num = w_.shape[3]
	prune_num = int(node_num*args.r)
	w_ = np.reshape(w_,(w_.shape[0]*w_.shape[1],w_.shape[3]))
	s = np.sum(w_,axis=0)
	idx = np.argsort(s)
	weight_obj = weights[wl]
	weight_arr = weight_obj.eval()
	for i in range(prune_num):
		weight_arr[:,:,0,idx[i]] = 0
	print("Non-zero count (%s): %s" % (wl, weight_arr.size - 25*prune_num))
	dict_nzidx[wl] = abs(weight_arr) > 0
	sess.run(weight_obj.assign(weight_arr))

	target = "conv1"
	wl = "b_" + target
	weight_obj = weights[wl]
	weight_arr = weight_obj.eval()
	for i in range(prune_num):
		weight_arr[idx[i]] = 0
	print("Non-zero count (%s): %s" % (wl, weight_arr.size - prune_num))
	dict_nzidx[wl] = abs(weight_arr) > 0
	sess.run(weight_obj.assign(weight_arr))

	target = "conv2"
	wl = "w_" + target
	weight_obj = weights[wl]
	weight_arr = weight_obj.eval()
	for i in range(prune_num):
		weight_arr[:,:,idx[i],:] = 0
	w_ = np.reshape(weight_arr,(weight_arr.shape[0]*weight_arr.shape[1]*weight_arr.shape[2],weight_arr.shape[3]))
	s = np.sum(w_,axis=0)
	idx = np.argsort(s)
	node_num = weight_arr.shape[3]
	prune_num2 = int(node_num*args.r)
	for i in range(prune_num2):
		weight_arr[:,:,:,idx[i]] = 0
	print("Non-zero count (%s): %s" % (wl, 5*5*(32-prune_num)*(64-prune_num2)))
	dict_nzidx[wl] = abs(weight_arr) > 0
	sess.run(weight_obj.assign(weight_arr))
	
	target = "conv2"
	wl = "b_" + target
	weight_obj = weights[wl]
	weight_arr = weight_obj.eval()
	for i in range(prune_num2):
		weight_arr[idx[i]] = 0
	print("Non-zero count (%s): %s" % (wl, weight_arr.size - prune_num2))
	dict_nzidx[wl] = abs(weight_arr) > 0
	sess.run(weight_obj.assign(weight_arr))

	target = "fc1"
	wl = "w_" + target
	weight_obj = weights[wl]
	weight_arr = weight_obj.eval()
	for i in range(prune_num2):
		weight_arr[idx[i]*49:(idx[i]+1)*49,:] = 0
	node_num = weight_arr.shape[1]
	prune_num = int(node_num*args.r)
	s = np.sum(weight_arr,axis=0)
	idx = np.argsort(s)
	for i in range(prune_num):
		weight_arr[:,idx[i]] = 0
	print("Non-zero count (%s): %s" % (wl, 7*7*(64-prune_num2)*(1024-prune_num)))
	dict_nzidx[wl] = abs(weight_arr) > 0
	sess.run(weight_obj.assign(weight_arr))

	target = "fc1"
	wl = "b_" + target
	w_= np.abs(weights[wl].eval())
	weight_obj = weights[wl]
	weight_arr = weight_obj.eval()
	for i in range(prune_num):
		weight_arr[idx[i]] = 0
	print("Non-zero count (%s): %s" % (wl, weight_arr.size - prune_num))
	dict_nzidx[wl] = abs(weight_arr) > 0
	sess.run(weight_obj.assign(weight_arr))

	target = "fc2"
	wl = "w_" + target
	w_= np.abs(weights[wl].eval())
	weight_obj = weights[wl]
	weight_arr = weight_obj.eval()
	for i in range(prune_num):
		weight_arr[idx[i],:] = 0
	print("Non-zero count (%s): %s" % (wl, weight_arr.size - 10*prune_num))
	dict_nzidx[wl] = abs(weight_arr) > 0
	sess.run(weight_obj.assign(weight_arr))
	return dict_nzidx

def apply_prune_iter_channel(weights,r):
	dict_nzidx = {}

	target = "conv1"
	wl = "w_" + target
	w_= np.abs(weights[wl].eval())
	node_num = w_.shape[3]
	prune_num = int(node_num*r)
	w_ = np.reshape(w_,(w_.shape[0]*w_.shape[1],w_.shape[3]))
	s = np.sum(w_,axis=0)
	idx = np.argsort(s)
	weight_obj = weights[wl]
	weight_arr = weight_obj.eval()
	for i in range(prune_num):
		weight_arr[:,:,0,idx[i]] = 0
	print("Non-zero count (%s): %s" % (wl, weight_arr.size - 25*prune_num))
	dict_nzidx[wl] = abs(weight_arr) > 0
	sess.run(weight_obj.assign(weight_arr))

	target = "conv1"
	wl = "b_" + target
	weight_obj = weights[wl]
	weight_arr = weight_obj.eval()
	for i in range(prune_num):
		weight_arr[idx[i]] = 0
	print("Non-zero count (%s): %s" % (wl, weight_arr.size - prune_num))
	dict_nzidx[wl] = abs(weight_arr) > 0
	sess.run(weight_obj.assign(weight_arr))

	target = "conv2"
	wl = "w_" + target
	weight_obj = weights[wl]
	weight_arr = weight_obj.eval()
	for i in range(prune_num):
		weight_arr[:,:,idx[i],:] = 0
	w_ = np.reshape(weight_arr,(weight_arr.shape[0]*weight_arr.shape[1]*weight_arr.shape[2],weight_arr.shape[3]))
	s = np.sum(w_,axis=0)
	idx = np.argsort(s)
	node_num = weight_arr.shape[3]
	prune_num2 = int(node_num*r)
	for i in range(prune_num2):
		weight_arr[:,:,:,idx[i]] = 0
	print("Non-zero count (%s): %s" % (wl, 5*5*(32-prune_num)*(64-prune_num2)))
	dict_nzidx[wl] = abs(weight_arr) > 0
	sess.run(weight_obj.assign(weight_arr))
	
	target = "conv2"
	wl = "b_" + target
	weight_obj = weights[wl]
	weight_arr = weight_obj.eval()
	for i in range(prune_num2):
		weight_arr[idx[i]] = 0
	print("Non-zero count (%s): %s" % (wl, weight_arr.size - prune_num2))
	dict_nzidx[wl] = abs(weight_arr) > 0
	sess.run(weight_obj.assign(weight_arr))

	target = "fc1"
	wl = "w_" + target
	weight_obj = weights[wl]
	weight_arr = weight_obj.eval()
	for i in range(prune_num2):
		weight_arr[idx[i]*49:(idx[i]+1)*49,:] = 0
	node_num = weight_arr.shape[1]
	prune_num = int(node_num*r)
	s = np.sum(weight_arr,axis=0)
	idx = np.argsort(s)
	for i in range(prune_num):
		weight_arr[:,idx[i]] = 0
	print("Non-zero count (%s): %s" % (wl, 7*7*(64-prune_num2)*(1024-prune_num)))
	dict_nzidx[wl] = abs(weight_arr) > 0
	sess.run(weight_obj.assign(weight_arr))

	target = "fc1"
	wl = "b_" + target
	w_= np.abs(weights[wl].eval())
	weight_obj = weights[wl]
	weight_arr = weight_obj.eval()
	for i in range(prune_num):
		weight_arr[idx[i]] = 0
	print("Non-zero count (%s): %s" % (wl, weight_arr.size - prune_num))
	dict_nzidx[wl] = abs(weight_arr) > 0
	sess.run(weight_obj.assign(weight_arr))

	target = "fc2"
	wl = "w_" + target
	w_= np.abs(weights[wl].eval())
	weight_obj = weights[wl]
	weight_arr = weight_obj.eval()
	for i in range(prune_num):
		weight_arr[idx[i],:] = 0
	print("Non-zero count (%s): %s" % (wl, weight_arr.size - 10*prune_num))
	dict_nzidx[wl] = abs(weight_arr) > 0
	sess.run(weight_obj.assign(weight_arr))
	return dict_nzidx

def apply_prune_node(weights):
	dict_nzidx = {}

	target = "fc1"
	wl = "w_" + target
	w_= np.abs(weights[wl].eval())
	node_num = w_.shape[1]
	prune_num = int(node_num*args.r)
	s = np.sum(w_,axis=0)
	idx = np.argsort(s)
	weight_obj = weights[wl]
	weight_arr = weight_obj.eval()
	for i in range(prune_num):
		weight_arr[:,idx[i]] = 0
	print("Non-zero count (%s): %s" % (wl, weight_arr.size - 7*7*64*prune_num))
	dict_nzidx[wl] = abs(weight_arr) > 0
	sess.run(weight_obj.assign(weight_arr))
	target = "fc1"
	wl = "b_" + target
	w_= np.abs(weights[wl].eval())
	weight_obj = weights[wl]
	weight_arr = weight_obj.eval()
	for i in range(prune_num):
		weight_arr[idx[i]] = 0
	print("Non-zero count (%s): %s" % (wl, weight_arr.size - prune_num))
	dict_nzidx[wl] = abs(weight_arr) > 0
	sess.run(weight_obj.assign(weight_arr))
	target = "fc2"
	wl = "w_" + target
	w_= np.abs(weights[wl].eval())
	weight_obj = weights[wl]
	weight_arr = weight_obj.eval()
	for i in range(prune_num):
		weight_arr[idx[i],:] = 0
	print("Non-zero count (%s): %s" % (wl, weight_arr.size - 10*prune_num))
	dict_nzidx[wl] = abs(weight_arr) > 0
	sess.run(weight_obj.assign(weight_arr))
	return dict_nzidx

def apply_iter_prune_layer(weights,r,dict_nzidx,target):
	wl = "w_" + target
	weight_obj = weights[wl]
	weight_arr = weight_obj.eval()
	w_sort = np.sort(np.abs(weight_arr.reshape(-1)))
	th = w_sort[int(len(w_sort)*r)]
	print(wl + " threshold:\t" + str(th))

	# Apply pruning
	weight_arr, w_nzidx, w_nnz = papl.prune_dense(weight_arr, name=wl, thresh=th)

	dict_nzidx[wl] = w_nzidx
	sess.run(weight_obj.assign(weight_arr))

	return dict_nzidx

def apply_prune_on_grads(grads_and_vars, dict_nzidx):
	for key, nzidx in dict_nzidx.items():
		count = 0
		for grad, var in grads_and_vars:
			if var.name == key+":0":
				nzidx_obj = tf.cast(tf.constant(nzidx), tf.float32)
				grads_and_vars[count] = (tf.multiply(nzidx_obj, grad), var)
			count += 1
	return grads_and_vars

def gen_sparse_dict(dense_w):
	sparse_w = dense_w
	for target in papl.config.target_w_layer:
		target_arr = np.transpose(dense_w[target].eval())
		sparse_arr = papl.prune_tf_sparse(target_arr, name=target)
		sparse_w[target+"_idx"]=tf.Variable(tf.constant(sparse_arr[0],dtype=tf.int32),
			name=target+"_idx")
		sparse_w[target]=tf.Variable(tf.constant(sparse_arr[1],dtype=tf.float32),
			name=target)
		sparse_w[target+"_shape"]=tf.Variable(tf.constant(sparse_arr[2],dtype=tf.int32),
			name=target+"_shape")
	return sparse_w

dense_w={
	"w_conv1": tf.Variable(tf.truncated_normal([5,5,1,32],stddev=0.1), name="w_conv1"),
	"b_conv1": tf.Variable(tf.constant(0.1,shape=[32]), name="b_conv1"),
	"w_conv2": tf.Variable(tf.truncated_normal([5,5,32,64],stddev=0.1), name="w_conv2"),
	"b_conv2": tf.Variable(tf.constant(0.1,shape=[64]), name="b_conv2"),
	"w_fc1": tf.Variable(tf.truncated_normal([7*7*64,1024],stddev=0.1), name="w_fc1"),
	"b_fc1": tf.Variable(tf.constant(0.1,shape=[1024]), name="b_fc1"),
	"w_fc2": tf.Variable(tf.truncated_normal([1024,10],stddev=0.1), name="w_fc2"),
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
	h_pool2_flat = tf.reshape(h_pool2, [-1, 7*7*64])
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
		result += accuracy.eval(feed_dict={x: batch[0], y_: batch[1], keep_prob: 1.0})
	result /= 20

	print(message+" %g\n" % result)
	return result

def check_file_exists(key):
	import os
	fileList = os.listdir(".")
	count = 0
	for elem in fileList:
		if elem.find(key) >= 0:
			count += 1
	return key + ("-"+str(count) if count>0 else "")

# Construct a dense model
x = tf.placeholder("float", shape=[None, 784], name="x")
y_ = tf.placeholder("float", shape=[None, 10], name="y_")
keep_prob = tf.placeholder("float", name="keep_prob")

y_conv = dense_cnn_model(dense_w)
tf.add_to_collection("y_conv", y_conv)

saver = tf.train.Saver()

if args.first_round == True:
	cross_entropy = -tf.reduce_sum(y_*tf.log(tf.clip_by_value(y_conv,1e-10,1.0)))
	train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
	correct_prediction = tf.equal(tf.argmax(y_conv,1), tf.argmax(y_,1))
	accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
	tf.add_to_collection("accuracy", accuracy)

	sess.run(tf.initialize_all_variables())

	for i in range(20000):
		batch = mnist.train.next_batch(50)
		if i%100 == 0:
			train_accuracy = accuracy.eval(feed_dict={x:batch[0], y_: batch[1], keep_prob: 1.0})
			print("step %d, training accuracy %g"%(i, train_accuracy))
		train_step.run(feed_dict={x: batch[0], y_: batch[1], keep_prob: 0.5})

	score = test(y_conv, message="First-round testing accuracy")
	papl.log("baseline_accuracy.log", score)
	
	papl.print_weight_vars(dense_w, papl.config.all_layer, papl.config.all_dat, show_zero=papl.config.show_zero)
	saver.save(sess, os.path.join(os.getcwd(), "model_ckpt_dense"))

if args.second_round == True:
	saver.restore(sess, args.checkpoint)

	# Apply pruning on this context
	dict_nzidx = apply_prune(dense_w)

	# save model objects to readable format
	papl.print_weight_vars(dense_w, papl.config.all_layer, papl.config.all_p_dat, show_zero=papl.config.show_zero)

	# Test prune-only networks
	score = test(y_conv, message="Second-round after-prune testing accuracy")

	# save model objects to serialized format
	saver.save(sess, os.path.join(os.getcwd(), "model_ckpt_dense_pruned"))

	# Retrain networks
	cross_entropy = -tf.reduce_sum(y_*tf.log(tf.clip_by_value(y_conv,1e-10,1.0)))
	trainer = tf.train.AdamOptimizer(1e-4)
	grads_and_vars = trainer.compute_gradients(cross_entropy)
	grads_and_vars = apply_prune_on_grads(grads_and_vars, dict_nzidx)
	train_step = trainer.apply_gradients(grads_and_vars)

	correct_prediction = tf.equal(tf.argmax(y_conv,1), tf.argmax(y_,1))
	accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))

	# Initialize firstly touched variables (mostly from accuracy calc.)
	for var in tf.global_variables():
		if tf.is_variable_initialized(var).eval() == False:
			sess.run(tf.variables_initializer([var]))

	# Train x epochs additionally
	for i in range(papl.config.retrain_iterations):
		batch = mnist.train.next_batch(50)
		train_step.run(feed_dict={x: batch[0], y_: batch[1], keep_prob: 0.5})
	train_accuracy = accuracy.eval(feed_dict={x:batch[0], y_: batch[1], keep_prob: 1.0})
	print("After retrain, training accuracy %g"%(train_accuracy))

	saver.save(sess, os.path.join(os.getcwd(), "model_ckpt_dense_retrained"))
	score = test(y_conv, message="Second-round after retrain testing accuracy")

if args.third_round == True:
	saver.restore(sess, args.checkpoint)
	dict_nzidx = {}
	# Apply pruning on this context
	for target in papl.config.target_layer:
		wl = "w_" + target
		dict_nzidx[wl] = abs(dense_w[wl].eval()) > 0
	prune_r = 0.05
	for i in range(int(round(args.r/prune_r))):
		print("Iteration (%s)"%(i+1))
		for target in papl.config.target_layer:
			dict_nzidx = apply_iter_prune_layer(dense_w,prune_r*(i+1),dict_nzidx,target)

		score = test(y_conv, message="Third-round iter prune testing accuracy")

		# Retrain networks
		cross_entropy = -tf.reduce_sum(y_*tf.log(tf.clip_by_value(y_conv,1e-10,1.0)))
		trainer = tf.train.AdamOptimizer(1e-4)
		grads_and_vars = trainer.compute_gradients(cross_entropy)
		grads_and_vars = apply_prune_on_grads(grads_and_vars, dict_nzidx)
		train_step = trainer.apply_gradients(grads_and_vars)

		correct_prediction = tf.equal(tf.argmax(y_conv,1), tf.argmax(y_,1))
		accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))

		# Initialize firstly touched variables (mostly from accuracy calc.)
		for var in tf.global_variables():
			if tf.is_variable_initialized(var).eval() == False:
				sess.run(tf.variables_initializer([var]))

		# Train x epochs additionally
		for i in range(papl.config.retrain_iterations):
			batch = mnist.train.next_batch(50)
			train_step.run(feed_dict={x: batch[0], y_: batch[1], keep_prob: 0.5})
		train_accuracy = accuracy.eval(feed_dict={x:batch[0], y_: batch[1], keep_prob: 1.0})
		print("After retrain, training  accuracy: %g"%(train_accuracy))
		score = test(y_conv, message="Retrain test accuracy: ")

	saver.save(sess, os.path.join(os.getcwd(), "model_ckpt_dense_retrained"))
	papl.print_weight_vars(dense_w, papl.config.all_layer, papl.config.all_p_dat, show_zero=papl.config.show_zero)
	score = test(y_conv, message="Third-round final testing accuracy")

if args.fourth_round == True:
	saver.restore(sess, args.checkpoint)
	dict_nzidx = {}
	# Apply pruning on this context
	for target in papl.config.target_layer:
		wl = "w_" + target
		dict_nzidx[wl] = abs(dense_w[wl].eval()) > 0
	prune_r = 0.05
	for itr in range(int(round(args.r/prune_r))):
		for target in papl.config.target_layer:
			print("Iteration (%s)"%(itr+1))
			dict_nzidx = apply_iter_prune_layer(dense_w,prune_r,dict_nzidx,target)

			# Test prune-only networks
			score = test(y_conv, message="Fourth-round iter-layer prune testing accuracy")

			# Retrain networks
			cross_entropy = -tf.reduce_sum(y_*tf.log(tf.clip_by_value(y_conv,1e-10,1.0)))
			trainer = tf.train.AdamOptimizer(1e-4)
			grads_and_vars = trainer.compute_gradients(cross_entropy)
			grads_and_vars = apply_prune_on_grads(grads_and_vars, dict_nzidx)
			train_step = trainer.apply_gradients(grads_and_vars)

			correct_prediction = tf.equal(tf.argmax(y_conv,1), tf.argmax(y_,1))
			accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))

			# Initialize firstly touched variables (mostly from accuracy calc.)
			for var in tf.global_variables():
				if tf.is_variable_initialized(var).eval() == False:
					sess.run(tf.variables_initializer([var]))

			# Train x epochs additionally
			for i in range(papl.config.retrain_iterations):
				batch = mnist.train.next_batch(50)
				train_step.run(feed_dict={x: batch[0], y_: batch[1], keep_prob: 0.5})
			train_accuracy = accuracy.eval(feed_dict={x:batch[0], y_: batch[1], keep_prob: 1.0})
			print("After retrain, training  accuracy: %g"%(train_accuracy))
			score = test(y_conv, message="Retrain test accuracy: ")

	saver.save(sess, os.path.join(os.getcwd(), "model_ckpt_dense_retrained"))
	papl.print_weight_vars(dense_w, papl.config.all_layer, papl.config.all_p_dat, show_zero=papl.config.show_zero)
	# Test the retrained model
	score = test(y_conv, message="Fourth-round final testing accuracy")

if args.fifth_round == True:
	saver.restore(sess, args.checkpoint)
	# Apply pruning on this context
	prune_r = 0.05
	grad_w = {}
	for i in range(int(round(args.r/prune_r))):
		print("Iteration (%s)"%(i+1))
		''' # gradient-based pruning, but worse 
		cross_entropy = -tf.reduce_sum(y_*tf.log(tf.clip_by_value(y_conv,1e-10,1.0)))
		trainer = tf.train.AdamOptimizer(1e-4)
		grads_and_vars = trainer.compute_gradients(cross_entropy)
	
		#print(grads_and_vars)
		#print(grads_and_vars[4])
		#print(grads_and_vars[6])
		train_step = trainer.apply_gradients(grads_and_vars)
		batch = mnist.train.next_batch(10000)
		grad_val = sess.run(grads_and_vars,feed_dict={x: batch[0], y_: batch[1], keep_prob: 1})
		grad_w["w_fc1"] = grad_val[4][0]
		grad_w["w_fc2"] = grad_val[6][0]
		#print(grad_val[4][0].shape)
		#print(grad_val[4][0])
		#print(grad_val[4][1])
		#print(grad_val[6])
		'''
		dict_nzidx = apply_prune_iter_channel(dense_w,prune_r*(i+1))
		score = test(y_conv, message="Fifth-round iter-layer prune testing accuracy")

		# Retrain networks
		cross_entropy = -tf.reduce_sum(y_*tf.log(tf.clip_by_value(y_conv,1e-10,1.0)))
		trainer = tf.train.AdamOptimizer(1e-4)
		grads_and_vars = trainer.compute_gradients(cross_entropy)
		grads_and_vars = apply_prune_on_grads(grads_and_vars, dict_nzidx)
		train_step = trainer.apply_gradients(grads_and_vars)

		correct_prediction = tf.equal(tf.argmax(y_conv,1), tf.argmax(y_,1))
		accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))

		# Initialize firstly touched variables (mostly from accuracy calc.)
		for var in tf.global_variables():
			if tf.is_variable_initialized(var).eval() == False:
				sess.run(tf.variables_initializer([var]))

		# Train x epochs additionally
		for i in range(papl.config.retrain_iterations):
			batch = mnist.train.next_batch(50)
			train_step.run(feed_dict={x: batch[0], y_: batch[1], keep_prob: 0.5})
		train_accuracy = accuracy.eval(feed_dict={x:batch[0], y_: batch[1], keep_prob: 1.0})
		print("After retrain, training  accuracy: %g"%(train_accuracy))
		score = test(y_conv, message="Retrain test accuracy: ")

	saver.save(sess, os.path.join(os.getcwd(), "model_ckpt_dense_retrained"))
	papl.print_weight_vars(dense_w, papl.config.all_layer, papl.config.all_p_dat, show_zero=papl.config.show_zero)
	# Test the retrained model
	score = test(y_conv, message="Fifth-round final testing accuracy")

if args.sixth_round == True:
	if args.second_round == False and args.third_round == False and args.fourth_round == False and args.fifth_round == False:
		saver.restore(sess, os.path.join(os.getcwd(), "model_ckpt_dense_retrained"))

	# Transform final weights to a sparse form
	sparse_w = gen_sparse_dict(dense_w)

	# Initialize new variables in a sparse form
	for var in tf.global_variables():
		if tf.is_variable_initialized(var).eval() == False:
			sess.run(tf.variables_initializer([var]))

	# Save model objects to readable format
	papl.print_weight_vars(dense_w, papl.config.all_layer, papl.config.all_tp_dat, show_zero=papl.config.show_zero)
	# Save model objects to serialized format
	final_saver = tf.train.Saver(sparse_w)
	final_saver.save(sess, os.path.join(os.getcwd(), "model_ckpt_sparse_retrained")) 
